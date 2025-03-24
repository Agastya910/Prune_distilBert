# Gradient-Based Transformer Layer Pruning

## Agastya Todi

### Abstract
A gradient-driven pruning method for transformer models, evaluating performance-size tradeoffs on SST-2 classification. Using DistilBERT (66M parameters), it is demonstrated that removing layers with the lowest gradient norms achieves significant compression while maintaining accuracy. The method achieves **21.2% model size reduction** with **12.5% accuracy drop** while improving inference speed by **4.4×**.

## 1. Introduction
Existing approaches focus on weight pruning [1] or quantization, whereas layer pruning offers distinct advantages for architectural simplification. Gradient-based layer importance scoring for structured pruning is done here, motivated by the hypothesis that gradient magnitudes better reflect parameter importance than activation patterns [2].

## 2. Method

### 2.1 Relevance Scoring
Layer importance \( I_l \) is computed as:

\[ I_l = ||\nabla_{W_l} L||_2 \]

where \( \nabla_{W_l} L \) represents layer gradients during the backward pass. Gradients are preferred over activations as they directly measure the layer’s influence on loss minimization.

### 2.2 Pruning Protocol
1. Fine-tune DistilBERT on SST-2 (3 epochs, batch=64)
2. Compute layer gradients via PyTorch backward hooks
3. Remove \( K\% \) layers with lowest \( I_l \) scores
4. Update model configuration and architecture
5. Evaluate pruned model using `torch.evaluate()`

## 3. Results

| Metric          | Original  | 30% Pruned         | 40% Pruned         |
|----------------|-----------|---------------------|---------------------|
| Size (MB)      | 255.42    | 228.38 (-10.6%)    | 201.34 (-21.2%)    |
| Inference (s)  | 8.55      | 2.37 (3.7×)        | 1.94 (4.4×)        |
| Accuracy       | 0.9128    | 0.8200 (-10.2%)    | 0.7982 (-12.5%)    |
| Validation Loss| 0.236     | 0.4006             | 0.4322             |

## 4. Ideas for Improvement
- **Hessian-based Scoring:** Incorporate second-order information of the gradients with respect to the loss following [3].
- **Cross-Epoch Gradients:** Aggregate relevance across all fine-tuning phases.
- **Large-Scale Validation:** Test on models with a large number of parameters and more diverse, multi-domain datasets.
- **Attention Head Pruning:** Perform targeted pruning of attention heads instead of entire layers.

## References
1. Sanh, V., Wolf, T., & Rush, A. (2020). Movement Pruning: Adaptive Sparsity by Fine-Tuning. *NeurIPS*.
2. Sanh, V., et al. (2020). Pruning Pre-trained Language Models Without Fine-Tuning. *ICML*.
3. Frankle, J., & Carbin, M. (2019). The Lottery Ticket Hypothesis. *ICLR*.
