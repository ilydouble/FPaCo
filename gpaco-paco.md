
# Comparison: PaCo vs GPaCo

This document outlines the core differences between **PaCo** (Parametric Contrastive Learning) and **GPaCo** (Generalized PaCo) as implemented in this codebase.

## 1. Loss Function Definitions

### PaCo Loss (Parametric Contrastive Learning)

PaCo combines supervised contrastive learning with a parametric classifier. To handle long-tail distributions, it explicitly adjusts the supervised logits using **class frequency weights**.

$$
\mathcal{L}_{PaCo} = - \sum_{i \in B} \log \frac{\exp(z_i \cdot z_i^+ / \tau + \mathbb{I}_{sup} \cdot \log \pi_{y_i})}{\sum_{j \in A} \exp(z_i \cdot z_j / \tau + \mathbb{I}_{sup} \cdot \log \pi_{y_j})}
$$

**In Code Implementation (`paco/train_paco.py`):**

The supervised logits ($W \cdot z_i$) are adjusted by the class frequency prior $\log(\pi_c)$:

```python
# sup_logits: Output of the classifier (Parametric branch)
# weight: Class frequency probability vector (pi)
sup_part = (sup_logits + torch.log(self.weight + 1e-9)) / self.supt
anchor_dot_contrast = torch.cat((sup_part, anchor_dot_contrast), dim=1)
```

The target mask is a **Hard One-Hot** encoding:
```python
one_hot_label = torch.nn.functional.one_hot(labels)
mask = torch.cat((one_hot_label * beta, mask * alpha), dim=1)
```

### GPaCo Loss (Generalized PaCo)

GPaCo generalizes the framework by introducing **Label Smoothing** to the supervised signal, preventing overfitting to head classes and learning more robust features for tail classes. In this specific implementation, it removes the explicit class-frequency re-weighting term found in PaCo.

**In Code Implementation (`gpaco/train_gpaco.py`):**

The supervised logits are **not** adjusted by class frequency:

```python
# No log(weight) term
anchor_dot_contrast = torch.cat((sup_logits / self.supt, anchor_dot_contrast), dim=1)
```

The target mask uses **Label Smoothing**:

```python
# Soft Label Generation
one_hot_label = torch.nn.functional.one_hot(labels)
one_hot_label = smooth / (num_classes - 1) * (1 - one_hot_label) + (1 - smooth) * one_hot_label

# Composite Mask
mask = torch.cat((one_hot_label * beta, mask * alpha), dim=1)
```

## 2. Variable Definitions

| Variable | Symbol | Definition & Function |
| :--- | :---: | :--- |
| **Features** | $z$ | The normalized feature vector output by the backbone encoder. |
| **Sup Logits** | $W \cdot z$ | The classification scores output by the linear classifier (Parametric Branch). |
| **Queue** | $Q$ | A memory bank storing historical features (keys) for contrastive learning. |
| **Temperature** | $\tau$ | Scales the logits to control the sharpness of the probability distribution. |
| **Alpha** | $\alpha$ | Weight for the **Contrastive** part of the target mask (usually small, e.g., 0.05). |
| **Beta** | $\beta$ | Weight for the **Supervised** part of the target mask (usually large, e.g., 1.0). |
| **Gamma** | $\gamma$ | Weight for the **Negative Samples** in the denominator of the loss (often 1.0). |
| **Sup Temperature** | `supt` | Scaling factor specifically for the supervised logits branch. |
| **Class Weight** | $\pi_c$ | In PaCo: Frequency of class $c$ in training set. Used to debias the classifier. |
| **Smooth** | $\epsilon$ | In GPaCo: The smoothing factor (e.g., 0.1). Allocates some probability to non-target classes. |

## 3. Core Differences Summary

| Feature | PaCo (`paco/train_paco.py`) | GPaCo (`gpaco/train_gpaco.py`) |
| :--- | :--- | :--- |
| **Core Mechanism** | Re-weighted Parametric Contrastive Learning | Generalized Label-Smoothed Contrastive Learning |
| **Handling Imbalance** | Explicit Logit Adjustment: $\log(\pi_c)$ | Implicit Regularization via **Label Smoothing** |
| **Target Distribution** | **Hard** One-Hot Labels (0/1) | **Soft** Smoothed Labels ($\epsilon, 1-\epsilon$) |
| **Modules Used** | Encoder (Q/K), Queue, Classifier | Encoder (Q/K), Queue, Classifier |
| **Loss Input** | Features + $\log(\text{ClassFreq})$ | Features (No Frequency Prior) |
| **Key Parameter** | `class_freq` (calculated from dataset) | `smooth` (hyperparameter, e.g., 0.1) |

### Why this matters?

*   **PaCo** assumes that the classifier bias is caused strictly by the frequency of samples. By adding $\log(\pi_c)$, it theoretically restores the boundary for balanced classification.
*   **GPaCo** assumes that simply re-weighting might not be enough or optimal. By using Label Smoothing, it forces the model to learn less "confident" boundaries for head classes, effectively reserving feature space for tail classes and preventing feature collapse.
