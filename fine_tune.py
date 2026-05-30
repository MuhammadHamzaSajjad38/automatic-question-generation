"""
Fine-tuning script for T5-small on SQuAD 1.1 for Question Generation.

This script:
  1. Loads SQuAD 1.1 from Hugging Face datasets
  2. Formats data as:  input  = "answer: <ans> context: <ctx>"
                        target = "<question>"
  3. Fine-tunes T5-small using Hugging Face Trainer

Usage:
    python fine_tune.py

The fine-tuned model will be saved to ./models/t5-qg-finetuned/
"""
import os
import torch
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# ── Configuration ───────────────────────────────────────────────────────────
BASE_MODEL = "t5-small"
OUTPUT_DIR = "./models/t5-qg-finetuned"
HUB_MODEL_ID = "YOUR_HF_USERNAME/t5-small-qg"  # <-- Change this to your Hugging Face username!
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 64
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 3e-4
SEED = 42


def prepare_dataset():
    """Load SQuAD 1.1 and format for question generation."""
    print("[1/4] Loading SQuAD 1.1 dataset ...")
    dataset = load_dataset("squad", trust_remote_code=True)

    def format_example(example):
        answer = example["answers"]["text"][0]
        context = example["context"]
        question = example["question"]

        # Highlight the answer in context
        highlighted = context.replace(answer, f"<hl> {answer} <hl>", 1)
        input_text = f"generate question: {highlighted}"

        return {
            "input_text": input_text,
            "target_text": question,
        }

    print("[2/4] Formatting examples ...")
    dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)
    return dataset


def tokenize_dataset(dataset, tokenizer):
    """Tokenize the formatted dataset."""
    def tokenize_fn(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            examples["target_text"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("[3/4] Tokenizing ...")
    return dataset.map(tokenize_fn, batched=True, remove_columns=["input_text", "target_text"])


def main():
    """Run the full fine-tuning pipeline."""
    # Load tokenizer & model
    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)

    # Prepare data
    dataset = prepare_dataset()
    tokenized = tokenize_dataset(dataset, tokenizer)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        seed=SEED,
        report_to="none",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    print("[4/4] Starting fine-tuning ...")
    trainer.train()

    # Save & Push
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Model saved locally to {os.path.abspath(OUTPUT_DIR)}")
    
    print(f"\n🚀 Pushing model to Hugging Face Hub: {HUB_MODEL_ID} ...")
    trainer.push_to_hub(commit_message="Fine-tuned T5 on SQuAD for Question Generation")
    print(f"\n✅ Successfully pushed to {HUB_MODEL_ID} on the Hub!")


if __name__ == "__main__":
    main()
