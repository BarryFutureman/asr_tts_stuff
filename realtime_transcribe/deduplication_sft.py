import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def prepare_models(model_name, cache_dir="cache"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def run_training(model, base_model_path, tokenizer, processed_dataset, output_dir, optimizer=None,
                 scheduler=None):
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import wandb
    import os

    wandb.login(key="a2eae9bb4b212c39378ce78b9850053f54b9eb3b")
    wandb.init(project="deduplicator", name=output_dir)
    training_args = TrainingArguments(
        output_dir=f"{output_dir}",
        dataloader_drop_last=True,
        save_strategy="no",
        num_train_epochs=1,
        # max_steps=100,
        logging_steps=1,
        per_device_train_batch_size=4,
        learning_rate=4e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        weight_decay=0.1,
        push_to_hub=False,
        include_tokens_per_second=True,
        report_to="wandb",
        # report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_dataset,
        args=training_args,
        optimizers=(optimizer, scheduler) if optimizer is not None and scheduler is not None else (None, None),
    )

    trainer_stats = trainer.train()
    trainer.save_model(f"{output_dir}")
    tokenizer.save_pretrained(f"{output_dir}")

    # Save optimizer state
    if optimizer is not None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(optimizer.state_dict(), f"{output_dir}/optimizer.pt")

    if scheduler is not None:
        torch.save(scheduler.state_dict(), f"{output_dir}/scheduler.pt")
        print(f"saving scheduler state dict at {output_dir}/scheduler.pt")
        

def load_deduplication_data(tokenizer, cache_dir="cache"):
    """Load and process the text-based-deduplication dataset"""
    
    def process_batch(batch):
        texts = []
        for chunks, full_text in zip(batch["chunks"], batch["full_text"]):
            # Format as: chunk1\n->chunk2\n->chunk3\n->chunk4\n=full_text
            prompt = "<Transcription Deduplication>"
            chunk_text = "\n->".join(chunks)
            formatted_text = f"{prompt}\n{chunk_text}\n={full_text}\n<|endoftext|>"
            texts.append(formatted_text)
        return {"text": texts}

    # Load the deduplication dataset
    raw_dataset = load_dataset(
        "BarryFutureman/text-based-deduplication-long",
        split="train",
        cache_dir=cache_dir
    )
    
    raw_dataset = raw_dataset.shuffle()

    # Map processing
    processed_dataset = raw_dataset.map(
        process_batch,
        remove_columns=raw_dataset.column_names,
        desc="Processing deduplication dataset",
        batched=True,
        batch_size=64,
        num_proc=1
    )

    return processed_dataset


if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    OUTPUT_DIR = "deduplicator-135M"
    model, tokenizer = prepare_models(model_name, cache_dir="cache")

    # Load the deduplication dataset
    processed_dataset = load_deduplication_data(tokenizer, cache_dir="cache")

    run_training(model, model_name, tokenizer, processed_dataset, OUTPUT_DIR)