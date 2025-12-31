
# Comparison: BPaCo vs PaCo

This document outlines the core differences between **BPaCo** (Balanced PaCo) and the original **PaCo** (Parametric Contrastive Learning).

## 1. Loss Function Definitions

### PaCo Loss (Parametric Contrastive Learning)

PaCo addresses long-tail recognition by introducing a set of parametric class centers (classifier weights) into the contrastive learning framework. It explicitly debiases the learning process by adding a log-frequency adjustment to the supervised logits.

$$
\mathcal{L}_{PaCo} = - \sum_{i \in B} \log \frac{\exp(z_i \cdot z_i^+ / \tau + \mathbb{I}_{sup} \cdot \log \pi_{y_i})}{\sum_{j \in A} \exp(z_i \cdot z_j / \tau + \mathbb{I}_{sup} \cdot \log \pi_{y_j})}
$$

**Key Mechanism:**
*   **Parametric Classifiers**: The classifier weights $W$ serve as additional positive samples (centers) for each class.
*   **Logit Adjustment**: The term $\log \pi_c$ (where $\pi_c$ is the class frequency) is added to the logit $z \cdot W_c$ to down-weight head classes and up-weight tail classes during training.

### BPaCo Loss (Balanced PaCo)

BPaCo extends PaCo by checking a "Balanced Contrastive" term. It is a composite loss function consisting of the original PaCo loss plus an additional **Balanced Supervised Contrastive Loss (BalSCL)** term that explicitly incorporates class centers and class-averaging normalization.

$$
\mathcal{L}_{BPaCo} = \mathcal{L}_{PaCo} + \mathcal{L}_{Balanced}
$$

#### component 1: PaCo Term ($\mathcal{L}_{PaCo}$)
Identical to the standard PaCo loss described above, calculating contrast between features and (Queue + Classifier Matrix).

#### component 2: Balanced Term ($\mathcal{L}_{Balanced}$)
This term performs contrastive learning between the batch features and the **Explicit Class Centers**, utilizing a **Class-Averaging** normalization strategy to further mitigate imbalance.

**In Code Implementation (`bpaco_original/losses.py`):**

```python
# 1. Balanced Process (L_Balanced)
# Concatenate Features and Centers
features1 = torch.cat([features[:batch_size], centers], dim=0)

# Compute Logits
logits1 = features1[:batch_size].mm(features1.T)

# Class-Averaging Denominator (per_ins_weight)
# Divides the exponentiated logits by the number of instances of that class in the batch
exp_logits_sum1 = exp_logits1.div(per_ins_weight1).sum(dim=1, keepdim=True)

loss1 = - mean_log_prob_pos1

# 2. Total Loss
loss = loss1 + loss_paco
```

## 2. Variable Definitions

| Variable | Symbol | Definition & Function |
| :--- | :---: | :--- |
| **Features** | $z$ | Normalized output vector from the image encoder. |
| **Centers** | $C$ | Explicit class centers passed to the loss (distinct from, but related to classifier weights). |
| **Class Freq** | $\pi_c$ | Frequency of class $c$ in the training set. |
| **Logit Adj** | $\log \pi_c$ | Bias added to classifier logits to debias prediction. Used in **both** PaCo and BPaCo components. |
| **Queue** | $Q$ | Memory bank of historical features. |
| **Per-Instance Weight** | $N_c$ | Used in $\mathcal{L}_{Balanced}$. The count of samples of class $c$ in the current batch/set. Used to normalize the denominator. |
| **Temperature** | $\tau$ | Scalar temperature parameter (e.g., 0.2). |
| **Sup Temperature** | `supt` | Temperature scaling specifically for the supervised parametric branch. |
| **Alpha** | $\alpha$ | Weight for contrastive vs supervised targets. |

## 3. Comparison Table

| Feature | PaCo | BPaCo |
| :--- | :--- | :--- |
| **Loss Components** | Single PaCo Loss | Composite: PaCo Loss + Balanced SCL Loss |
| **Center Utilization** | Implicit (as Classifier Weights) | Explicit (Concatenated to features in $\mathcal{L}_{Balanced}$) |
| **Normalization** | Standard Softmax Normalization | **Class-Averaging** Normalization (in $\mathcal{L}_{Balanced}$) |
| **Logit Adjustment** | Yes ($\log \pi_c$) | Yes ($\log \pi_c$) |
| **Complexity** | Moderate | High (Dual loss calculation) |
| **Focus** | Re-weighting classifier logits | Dual-focus: Re-weighting classifiers + Balanced Feature Representation |

### Why BPaCo?

While PaCo effectively debiases the classifier boundaries using $\log \pi_c$, **BPaCo** aims to improve the **feature space structure** itself. By adding the $\mathcal{L}_{Balanced}$ term, it explicitly enforces that features from the same class cluster tightly around their centers, and it normalizes this clustering objective by the class size (Class-Averaging) to ensure that tail classes (with fewer samples) contribute equally to the gradient updates as head classes.
