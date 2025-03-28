import os
import torch
from transformers import AutoModelForSequenceClassification
from config import MODEL_SAVE_PATH, PRUNE_PERCENTAGE

from utils import GradientCollector


def prune_transformer_layers(model, layers_to_prune):
    """Remove specified layers from transformer"""
    old_layers = model.distilbert.transformer.layer
    new_layers = torch.nn.ModuleList()
    
    for i, layer in enumerate(old_layers):
        if i not in layers_to_prune:
            new_layers.append(layer)
    
    model.distilbert.transformer.layer = new_layers
    model.config.num_hidden_layers = len(new_layers)
    return model

def prune_model():
    # Load the fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)

    # Load the gradients
    gradients = torch.load(os.path.join(MODEL_SAVE_PATH, "layer_grads.pt"))

    # Convert gradients to layer indices
    layer_indices = {}
    for name, grad in gradients.items():
        layer_idx = int(name.split('.')[3])  # Extract layer number (e.g., 'distilbert.transformer.layer.0')
        layer_indices[layer_idx] = grad

    # Sort layers by gradient norms (ascending order)
    sorted_layers = sorted(layer_indices.items(), key=lambda x: x[1])

    
    num_layers = len(sorted_layers)
    num_prune = int(num_layers * PRUNE_PERCENTAGE) 
    layers_to_prune = [idx for idx, _ in sorted_layers[:num_prune]]

    # Prune the least relevant layers
    pruned_model = prune_transformer_layers(model, layers_to_prune)

    # Save the pruned model
    pruned_model.save_pretrained(os.path.join(MODEL_SAVE_PATH, "manual_pruned_0.3_2"))

    print(f"Pruned {num_prune} layers: {layers_to_prune}")

if __name__ == "__main__":
    prune_model()
