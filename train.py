import time
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import os
from utils import GradientCollector
from config import MODEL_NAME, DATASET_NAME, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_SAVE_PATH
import torch

def fine_tune():
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset(DATASET_NAME)
    print(f"Dataset features: {dataset['train'].features}")

    # Tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize(examples):
        return tokenizer(examples['sentence'], padding="max_length", truncation=True)
    dataset = dataset.map(tokenize, batched=True)

    # Load model and move to the correct device
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2  # SST-2 has 2 classes
    ).to(device)  # Move model to the correct device

    # Set up Trainer with optimizations
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=True,  # Enable mixed precision
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation']
    )

    # Fine-tune
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    model.save_pretrained(MODEL_SAVE_PATH)

    # Collect gradients after training
    collector = GradientCollector(model)
    sample = dataset['validation'][0]  # Use a single sample for gradient computation

    # Tokenize the sample and move inputs to the correct device
    inputs = tokenizer(sample['sentence'], return_tensors="pt").to(device)  # Move inputs to the correct device
    labels = torch.tensor([sample['label']]).to(device)  # Move labels to the correct device

    # Forward pass and backward pass
    outputs = model(**inputs, labels=labels)
    outputs.loss.backward()

    # Save gradients
    torch.save(collector.gradients, os.path.join(MODEL_SAVE_PATH, "layer_grads.pt"))
    collector.remove_hooks()

if __name__ == "__main__":
    fine_tune()
