# import os
# os.environ["HF_DATASETS_CACHE"] = "/dev/shm/cache/datasets"
# os.environ["HF_HUB_CACHE"] = "/dev/shm/cache/hub"
# os.environ["TRANSFORMERS_CACHE"] = "/dev/shm/cache/transformers"
import torch
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import BitsAndBytesConfig, get_constant_schedule, get_cosine_schedule_with_warmup, TrainingArguments
from datasets import load_dataset
from tqdm import tqdm
import wandb
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset as TorchDataset
from qwen_omni_utils import process_mm_info
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import re

try:
    # Try relative import (works when run as package)
    from .my_sft_trainer import SFTTrainer
except ImportError:
    from my_sft_trainer import SFTTrainer


class ExpressoDataset(TorchDataset):
    """Dataset for Expresso audio transcription training."""

    def __init__(self, raw_dataset, processor):
        self.raw_dataset = raw_dataset
        self.processor = processor

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        """Process a single audio transcription example."""
        example = self.raw_dataset[idx]
        
        # Create the message format for transcription
        messages = [
            {"role": "system", "content":[{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role": "user", "content": [{"type": "audio", "audio": example["audio"]["array"]}, {"type": "text", "text": "Transcribe the given audio."}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"({example['style']}) {example['text']}"}]}
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # Process multimedia info
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        
        # Process with the processor
        inputs = self.processor(
            text=text, 
            images=images, 
            videos=videos, 
            audio=audios, 
            padding=False, 
            return_tensors="pt"
        )

        # Remove batch dimension
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # --- Filter labels: everything before assistant must be -100 ---
        # Find where the assistant's response starts in the tokenized input
        # Tokenize up to and including the user message
        user_messages = messages[:2]
        user_text = self.processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
        user_inputs = self.processor(
            text=user_text,
            images=images, 
            videos=videos, 
            audio=audios, 
            padding=False,
            return_tensors="pt"
        )
        user_input_ids = user_inputs["input_ids"].squeeze(0)
        # Set all tokens before assistant's response to -100
        labels[:len(user_input_ids)] = -100
        # --- End filter labels ---
        
        # Remove debug print/quit and add audio features and feature_attention_mask
        processed_example = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": inputs["input_features"].squeeze(0),
            "feature_attention_mask": inputs["feature_attention_mask"].squeeze(0),
        }

        return processed_example

    def select(self, indices):
        """Select a subset of the dataset by indices."""
        if isinstance(indices, int):
            indices = list(range(min(indices, len(self.raw_dataset))))

        # Create a new dataset with selected indices
        new_raw_dataset = self.raw_dataset.select(indices)
        return ExpressoDataset(new_raw_dataset, self.processor)

    def shuffle(self, seed=None):
        """Shuffle the dataset."""
        shuffled_raw_dataset = self.raw_dataset.shuffle(seed=seed)
        return ExpressoDataset(shuffled_raw_dataset, self.processor)


@dataclass
class ExpressoDataCollator:
    """Custom data collator for Expresso audio data."""

    def __init__(self, processor):
        self.processor = processor
        # Get the actual pad token ID from the tokenizer
        self.pad_token_id = processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = processor.tokenizer.eos_token_id
        print(f"Using pad_token_id: {self.pad_token_id}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Handle input_ids, attention_mask, and labels
        input_ids = []
        attention_mask = []
        labels = []
        input_features = []
        feature_attention_masks = []

        for f in features:
            input_ids.append(f["input_ids"])
            attention_mask.append(f["attention_mask"])
            labels.append(f["labels"])
            input_features.append(f["input_features"])
            feature_attention_masks.append(f["feature_attention_mask"])

        # Use simple right padding with correct pad token ID
        input_ids = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = rnn_utils.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = rnn_utils.pad_sequence(labels, batch_first=True, padding_value=-100)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": torch.stack(input_features),
            "feature_attention_mask": torch.stack(feature_attention_masks),
        }

        return batch


def prepare_models(model_name="KE-Team/Ke-Omni-R-3B", cache_dir="cache"):
    print("Loading:", model_name)

    # Configure quantization for QLoRA
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Load model
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        cache_dir=cache_dir,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # Set use_cache=False for gradient checkpointing compatibility

    # Dynamically extract target_modules for LoRA from state_dict
    layer_proj_pattern = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|down_proj|up_proj)\.weight")
    layer_indices = set()
    module_names = set()
    
    for k in model.state_dict().keys():
        m = layer_proj_pattern.match(k)
        if m:
            layer_indices.add(int(m.group(1)))
            module_names.add(f"{m.group(2)}.{m.group(3)}")

    if not layer_indices:
        raise NotImplementedError("No matching layers found in state_dict. Please check the model architecture.")
        # # Fallback target modules if pattern matching fails
        # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    else:
        # Use last 10 layers similar to orpheus approach
        last_n = 10
        selected_layers = sorted(layer_indices)[-last_n:]
        target_modules = []
        for i in selected_layers:
            for module in module_names:
                target_modules.append(f"model.layers.{i}.{module}")
                
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=16,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        lora_dropout=0.0,
        bias="none",
        use_rslora=True,
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # Required for gradient checkpointing with PEFT
    model.print_trainable_parameters()

    # Load processor
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    # Set padding
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    return model, processor


def prepare_optimizer(model, total_training_steps, optimizer=None, scheduler=None):
    new_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    if optimizer is not None:
        new_optimizer.load_state_dict(optimizer.state_dict())

    # Use cosine schedule with 100 warmup steps
    new_scheduler = get_cosine_schedule_with_warmup(
        new_optimizer,
        num_warmup_steps=10,
        num_training_steps=total_training_steps
    )
    if scheduler is not None:
        new_scheduler.load_state_dict(scheduler.state_dict())

    return new_optimizer, new_scheduler


def run_training(model, processor, processed_dataset, num_inner_steps=4, batch_size=4, optimizer=None,
                 scheduler=None, output_dir="output", eval_dataset=None):
    # Shuffle with seed by time
    seed = int(time.time())
    processed_dataset = processed_dataset.shuffle(seed=seed)

    # Create custom data collator first
    data_collator = ExpressoDataCollator(processor)

    # Create DataLoader with specified batch size and custom collator
    dataloader = DataLoader(
        processed_dataset,
        batch_size=batch_size,
        shuffle=False,  # Dataset is already shuffled
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=16,
        collate_fn=data_collator,  # Add the custom collator here
    )

    gradient_accumulation_steps = 4
    per_device_train_batch_size = 1

    # Add evaluation strategy if eval_dataset is provided
    eval_strategy = "steps" if eval_dataset is not None else "no"
    eval_steps = 40

    training_args = TrainingArguments(
        output_dir=f"{output_dir}",
        dataloader_drop_last=True,
        save_strategy="no",
        max_steps=num_inner_steps,
        logging_steps=2,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        warmup_steps=0,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        weight_decay=0.01,
        push_to_hub=False,
        include_tokens_per_second=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    # Create eval DataLoader if eval_dataset is provided
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            collate_fn=data_collator,  # Add the custom collator here too
        )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataloader,
        eval_dataset=eval_dataloader,
        args=training_args,
        data_collator=data_collator,
        processing_class=processor.tokenizer,
        optimizers=(optimizer, scheduler) if optimizer is not None and scheduler is not None else (None, None),
    )

    trainer_stats = trainer.train()
    
    # Merge LoRA with base model directly (no reloading needed)
    merged_model = model.merge_and_unload()
    
    torch.cuda.empty_cache()

    return trainer_stats, merged_model, optimizer, scheduler


def load_data(processor, cache_dir="cache", split="train", subset_size=None):
    """Load Expresso dataset."""
    print(f"Loading Expresso dataset, split: {split}")
    
    # Load the Expresso dataset
    raw_dataset = load_dataset("ylacombe/expresso", split=split, cache_dir=cache_dir)
    
    # Create Expresso dataset
    expresso_dataset = ExpressoDataset(raw_dataset, processor)
    
    # Select subset if specified
    if subset_size is not None:
        expresso_dataset = expresso_dataset.select(subset_size)
    
    return expresso_dataset


def run_evaluation(model, processor, eval_dataset=None, batch_size=4):
    """Evaluate the model on validation dataset."""
    import math

    model.eval()

    # Create DataLoader for evaluation
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    eval_args = TrainingArguments(
        output_dir="cache",
        per_device_eval_batch_size=1,
        fp16=False,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    # Create custom data collator for evaluation
    data_collator = ExpressoDataCollator(processor)

    # Initialize trainer for evaluation
    eval_trainer = SFTTrainer(
        model=model,
        args=eval_args,
        train_dataset=None,
        eval_dataset=eval_dataloader,
        data_collator=data_collator,
    )

    # Run evaluation
    eval_results = eval_trainer.evaluate()

    # Extract metrics and calculate perplexity
    eval_loss = eval_results.get("eval_loss", float('inf'))
    perplexity = math.exp(eval_loss) if eval_loss < 100 else float('inf')

    model.train()  # Reset to training mode

    eval_metrics = {
        "eval_loss": eval_loss,
        "eval_perplexity": perplexity,
        "eval_samples": len(eval_dataset)
    }

    return eval_metrics


if __name__ == "__main__":
    # Configuration
    model_name = "KE-Team/Ke-Omni-R-3B"
    cache_dir = "autodl-tmp/cache"
    output_dir = "autodl-tmp/result"
    num_training_steps = 1000
    batch_size = 1
    
    print("Starting Expresso QLoRA fine-tuning...")
    
    # Prepare model and processor
    print("Loading model and processor...")
    model, processor = prepare_models(model_name=model_name, cache_dir=cache_dir)
    
    # Load datasets
    print("Loading training dataset...")
    train_dataset = load_data(processor, cache_dir=cache_dir, split="train", subset_size=None)
    
    # Prepare optimizer and scheduler
    print("Preparing optimizer and scheduler...")
    optimizer, scheduler = prepare_optimizer(model, num_training_steps)
    
    # Run training
    print(f"Starting training for {num_training_steps} steps...")
    trainer_stats, trained_model, final_optimizer, final_scheduler = run_training(
        model=model,
        processor=processor,
        processed_dataset=train_dataset,
        num_inner_steps=num_training_steps,
        batch_size=batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        eval_dataset=None
    )
    
    print("Training completed!")
    print(f"Training stats: {trainer_stats}")
    
    # Save the merged model directly
    print(f"Saving merged model to {output_dir}..")
    trained_model.save_pretrained(f"{output_dir}")
    processor.save_pretrained(f"{output_dir}")

