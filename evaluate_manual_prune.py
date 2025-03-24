import time
import os
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from config import MODEL_SAVE_PATH, DATASET_NAME, BATCH_SIZE, MODEL_NAME

def compute_accuracy(predictions, labels):
    """Compute accuracy manually."""
    predictions = np.argmax(predictions, axis=-1)  # Get predicted class labels
    return np.mean(predictions == labels)  # Calculate accuracy

def evaluate_model(model_path):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load dataset
    dataset = load_dataset(DATASET_NAME)
    
    # Tokenize dataset
    def tokenize(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True)
    dataset = dataset.map(tokenize, batched=True)
    
    # Set up Trainer for evaluation
    training_args = TrainingArguments(
        output_dir="./eval_results",
        per_device_eval_batch_size=BATCH_SIZE,
        do_train=False,
        do_predict=False  # We only use evaluate, not predict
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset['validation']
    )
    
    # Measure inference time and compute validation loss
    start_time = time.time()
    results = trainer.evaluate()
    inference_time = time.time() - start_time
    
    # Get predictions for accuracy calculation
    predictions = trainer.predict(dataset['validation'])
    logits = predictions.predictions  # Model outputs (logits)
    labels = predictions.label_ids  # Ground truth labels
    accuracy = compute_accuracy(logits, labels)
    
    # Print results
    print(f"\nEvaluation results for {model_path}:")
    print(f"Model size: {get_model_size(model):.2f}MB")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Validation loss: {results['eval_loss']:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")

def get_model_size(model):
    """Calculate the size of the model in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024**2)

if __name__ == "__main__":
    # Evaluate original model
    print("=== Original Model ===")
    evaluate_model(MODEL_SAVE_PATH)
    
    # Evaluate pruned model
    print("\n=== Pruned Model ===")
    evaluate_model(os.path.join(MODEL_SAVE_PATH, "manual_pruned_0.3_2"))