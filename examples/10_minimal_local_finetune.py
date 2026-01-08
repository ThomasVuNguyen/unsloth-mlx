"""Minimal sanity-check fine-tuning run for 16 GB Apple Silicon machines."""

from datasets import load_dataset
from unsloth_mlx import (
    FastLanguageModel,
    SFTTrainer,
    SFTConfig,
    get_chat_template,
)


def format_sample(example, tokenizer):
    """Convert chat messages into a single training string."""
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


def main():
    model_name = "unsloth/SmolLM2-1.7B"
    dataset = load_dataset("json", data_files="sample_train.jsonl", split="train")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        load_in_4bit=True,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="auto")
    dataset = dataset.map(lambda ex: format_sample(ex, tokenizer))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="model",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=100,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            logging_steps=5,
            save_steps=25,
        ),
    )

    trainer.train()
    model.save_pretrained("model/lora_adapters")
    model.save_pretrained_merged("model/merged_model", tokenizer)


if __name__ == "__main__":
    main()
