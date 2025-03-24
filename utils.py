import torch

class GradientCollector:
    def __init__(self, model):
        self.model = model
        self.gradients = {}
        self.hooks = []
        
        # Register full backward hooks on transformer layers
        for name, module in self.model.named_modules():
            if 'transformer.layer' in name:
                hook = module.register_full_backward_hook(
                    self._store_grad_hook(name)
                )
                self.hooks.append(hook)
    
    def _store_grad_hook(self, name):
        def hook(module, grad_input, grad_output):
            # Store the L2 norm of the output gradients
            self.gradients[name] = grad_output[0].norm().item()
        return hook
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def prune_transformer_layers(model, layers_to_prune):
    """Remove specified layers from transformer"""
    old_layers = model.distilbert.transformer.layer
    new_layers = torch.nn.ModuleList()
    
    for i, layer in enumerate(old_layers):
        if i not in layers_to_prune:
            new_layers.append(layer)
    
    model.distilbert.transformer.layer = new_layers
    return model
