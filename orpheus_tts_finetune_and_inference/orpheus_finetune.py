from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import wandb
from torch.nn.utils.rnn import pad_sequence
from datasets import concatenate_datasets

model_name = "/student/jian1034/Desktop/TTS/orpheus-3b-0.1-kara-ft-kara2-kara-again"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")
tokenizer.pad_token = tokenizer.eos_token # Set pad token to eos token
tokenizer.padding_side = "left" # Set padding side to left

# train_dataset1 = load_dataset("BarryFutureman/kara-v2-tokenized-snac",
#                               split="train", cache_dir="cache").select(range(400))
train_dataset2 = load_dataset("BarryFutureman/kara-tokenized-snac",
                              split="train", cache_dir="cache")
train_dataset3 = load_dataset("BarryFutureman/kara-filtered-tokenized-snac",
                             split="train", cache_dir="cache")
train_dataset = concatenate_datasets([train_dataset2, train_dataset3])

train_dataset = train_dataset.shuffle(seed=42) # Shuffle the dataset

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             torch_dtype=torch.bfloat16,
                                             cache_dir="cache",
                                             device_map="auto",
                                             quantization_config=bnb_config)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False # Set use_cache=False for gradient checkpointing compatibility

# Dynamically extract target_modules for the last 10 layers from state_dict
import re
layer_proj_pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.(q_proj|k_proj|v_proj|o_proj|gate_proj|down_proj|up_proj)\.weight")
layer_indices = set()
proj_names = set()
for k in model.state_dict().keys():
    m = layer_proj_pattern.match(k)
    if m:
        layer_indices.add(int(m.group(1)))
        proj_names.add(m.group(2))

if not layer_indices:
    raise RuntimeError("No matching layers found in state_dict.")

last_n = 10
selected_layers = sorted(layer_indices)[-last_n:]
target_modules = []
for i in selected_layers:
    for proj in proj_names:
        target_modules.append(f"model.layers.{i}.mlp.{proj}")

lora_config = LoraConfig(
        r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        lora_alpha=16,
        target_modules=target_modules,
        # modules_to_save=["lm_head", "embed_tokens"],
        task_type="CAUSAL_LM",
        lora_dropout=0,
        bias="none",
        use_rslora=True,
    )

model = get_peft_model(model, lora_config)
model.enable_input_require_grads() # Recommended for gradient checkpointing with PEFT
model.print_trainable_parameters()

wandb.login(key="")
wandb.init(project="tts")

class FixedGradRequiringDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False) # Assuming causal LM, so mlm=False

    def __call__(self, features): # Changed 'call' to '__call__' to match parent class
        batch = {}

        first_item = features[0]
        # Determine if input_ids is already a tensor or a list of integers
        # This check might need adjustment based on how datasets library yields features
        are_already_tensors = isinstance(first_item.get("input_ids"), torch.Tensor)

        for key in ["input_ids", "attention_mask", "labels"]: # Process only relevant keys
            if key not in first_item: # Skip if key is missing in features
                if key == "labels" and "input_ids" in first_item: # Auto-create labels from input_ids
                    pass # Will be handled later
                else:
                    continue
            
            if key == "labels" and key not in first_item and "input_ids" in first_item:
                 # If labels are not present, create them from input_ids
                values = [f["input_ids"] for f in features]
            else:
                values = [f[key] for f in features]

            # Ensure values are lists of numbers for padding
            processed_values = []
            for v_list in values:
                if isinstance(v_list, torch.Tensor):
                    processed_values.append(v_list.tolist())
                elif isinstance(v_list, list):
                    processed_values.append(v_list)
                else:
                    # Handle cases where v_list might be a single number or other unexpected type
                    # This part might need more robust error handling or type checking
                    raise TypeError(f"Unexpected type for feature {key}: {type(v_list)}")


            if key == "input_ids":
                padding_value = self.tokenizer.pad_token_id
            elif key == "labels":
                padding_value = -100 # Standard padding value for labels in Hugging Face
            elif key == "attention_mask":
                padding_value = 0
            else:
                # Should not happen with the filtered keys
                raise ValueError(f"Unexpected key for padding: {key}")

            # Pad sequences
            # pad_sequence expects a list of Tensors. Convert lists to Tensors first.
            # Or, pad manually as in the original snippet. Let's stick to manual padding for now.
            max_length = max(len(v) for v in processed_values)
            
            padded_batch_values = []
            for v_list in processed_values:
                padding_needed = max_length - len(v_list)
                padded_batch_values.append(v_list + [padding_value] * padding_needed)
            
            batch[key] = torch.tensor(padded_batch_values, dtype=torch.long)

        # Ensure labels are present, cloning from input_ids if necessary
        # and applying the ignore_index where input_ids are padded.
        if "labels" not in batch and "input_ids" in batch:
            batch["labels"] = batch["input_ids"].clone()
            # If input_ids were padded with pad_token_id, labels should be -100 there
            if self.tokenizer.pad_token_id is not None:
                 batch["labels"][batch["input_ids"] == self.tokenizer.pad_token_id] = -100
        
        # The print statement for debugging shape
        # print(f"input_ids_padded in collator: {batch['input_ids'].shape}")
        return batch

data_collator_instance = FixedGradRequiringDataCollator(tokenizer=tokenizer)

output_dir = "orpheus-3b-0.1-kara-ft"
base_model_path = model_name

trainer = Trainer(
    model = model,
    train_dataset = train_dataset,
    data_collator = data_collator_instance,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 16,
        gradient_checkpointing = True,
        warmup_steps = 0,
        num_train_epochs = 4,
        learning_rate = 4e-4,
        bf16 = True,
        # optim = "adamw_torch",
        logging_steps = 1,
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        output_dir = output_dir,
        report_to = "wandb",
    ),
)

trainer_stats = trainer.train()
trainer.save_model(f"{output_dir}_adapter")

# Delete models and free up memory
del model
torch.cuda.empty_cache()

def merge_lora(base_model_path, tokenizer_to_save, lora_adapter_path): # Added tokenizer_to_save
    # Load the base model in bf16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=False,
        torch_dtype=torch.bfloat16,
        cache_dir="cache",
    )

    # Apply the LoRA adapter
    peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path, local_files_only=True)
    model = peft_model.merge_and_unload()

    # Save the merged model
    model.save_pretrained(f"{output_dir}")
    if tokenizer_to_save: # Save tokenizer
        tokenizer_to_save.save_pretrained(f"{output_dir}")

    torch.cuda.empty_cache()
    return

# Merge and save the model
merge_lora(base_model_path, tokenizer, f"{output_dir}_adapter")