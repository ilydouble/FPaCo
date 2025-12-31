# Experiment Results

![Accuracy Chart](results_chart.png)

## Detailed metrics

| Method                      | Dataset     | Accuracy   |   F1 Score |      AUC |
|:----------------------------|:------------|:-----------|-----------:|---------:|
| BioMedCLIP                  | aptos       | 0.3644     |     0.3594 |   0.7342 |
| CoOp                        | aptos       | 0.5589     |     0.5447 |   0.8277 |
| Tip-Adapter (Zero-Shot)     | aptos       | 0.3644     |     0.3594 |   0.7342 |
| Tip-Adapter (Training-Free) | aptos       | 0.4329     |     0.4168 |   0.764  |
| Tip-Adapter (Fine-Tuned)    | aptos       | 0.4411     |     0.4273 |   0.7684 |
| DPE                         | aptos       | 0.2959     |     0.268  |   0.7051 |
| ce_loss                     | aptos       | 0.6219     |     0.6038 |   0.9175 |
| focal_loss                  | aptos       | 0.6877     |     0.6865 |   0.9277 |
| paco                        | aptos       | 0.6466     |     0.5859 |   0.9288 |
| gpaco                       | aptos       | **0.7452** |     0.743  |   0.9274 |
| bpaco_original              | aptos       | 0.5753     |     0.5365 |   0.8775 |
| FPaCo (Heatmap)             | aptos       | 0.7014     |     0.6967 |   0.9023 |
| BioMedCLIP                  | fingerA     | 0.0491     |     0.0084 |   0.5135 |
| CoOp                        | fingerA     | 0.0427     |     0.0216 |   0.4954 |
| Tip-Adapter (Zero-Shot)     | fingerA     | 0.0513     |     0.0129 |   0.5116 |
| Tip-Adapter (Training-Free) | fingerA     | 0.0513     |     0.0129 |   0.5186 |
| Tip-Adapter (Fine-Tuned)    | fingerA     | 0.0791     |     0.0676 |   0.5335 |
| DPE                         | fingerA     | 0.0556     |     0.0061 |   0.5231 |
| ce_loss                     | fingerA     | 0.6959     |     0.505  |   0.9724 |
| focal_loss                  | fingerA     | 0.7297     |     0.6136 |   0.9778 |
| paco                        | fingerA     | **0.7365** |     0.6531 |   0.9805 |
| gpaco                       | fingerA     | 0.6453     |     0.6258 |   0      |
| bpaco_original              | fingerA     | 0.7162     |     0.5832 |   0.9772 |
| FPaCo (Heatmap)             | fingerA     | nan        |   nan      | nan      |
| BioMedCLIP                  | fingerB     | nan        |   nan      | nan      |
| CoOp                        | fingerB     | nan        |   nan      | nan      |
| Tip-Adapter (Zero-Shot)     | fingerB     | 0.0513     |     0.0129 |   0.5116 |
| Tip-Adapter (Training-Free) | fingerB     | 0.0513     |     0.0129 |   0.5186 |
| Tip-Adapter (Fine-Tuned)    | fingerB     | 0.0791     |     0.0676 |   0.5335 |
| DPE                         | fingerB     | nan        |   nan      | nan      |
| ce_loss                     | fingerB     | nan        |   nan      | nan      |
| focal_loss                  | fingerB     | nan        |   nan      | nan      |
| paco                        | fingerB     | nan        |   nan      | nan      |
| gpaco                       | fingerB     | **0.5812** |     0.5616 |   0.9375 |
| bpaco_original              | fingerB     | nan        |   nan      | nan      |
| FPaCo (Heatmap)             | fingerB     | nan        |   nan      | nan      |
| BioMedCLIP                  | fingerC     | nan        |   nan      | nan      |
| CoOp                        | fingerC     | nan        |   nan      | nan      |
| Tip-Adapter (Zero-Shot)     | fingerC     | 0.0513     |     0.0129 |   0.5116 |
| Tip-Adapter (Training-Free) | fingerC     | 0.0513     |     0.0129 |   0.5186 |
| Tip-Adapter (Fine-Tuned)    | fingerC     | 0.0791     |     0.0676 |   0.5335 |
| DPE                         | fingerC     | nan        |   nan      | nan      |
| ce_loss                     | fingerC     | nan        |   nan      | nan      |
| focal_loss                  | fingerC     | nan        |   nan      | nan      |
| paco                        | fingerC     | nan        |   nan      | nan      |
| gpaco                       | fingerC     | **0.5385** |     0.5097 |   0.933  |
| bpaco_original              | fingerC     | nan        |   nan      | nan      |
| FPaCo (Heatmap)             | fingerC     | nan        |   nan      | nan      |
| BioMedCLIP                  | mias        | 0.1429     |     0.0806 |   0.4524 |
| CoOp                        | mias        | 0.1429     |     0.0619 |   0.5119 |
| Tip-Adapter (Zero-Shot)     | mias        | 0.1429     |     0.0806 |   0.4524 |
| Tip-Adapter (Training-Free) | mias        | 0.1429     |     0.1196 |   0.4821 |
| Tip-Adapter (Fine-Tuned)    | mias        | 0.2500     |     0.2028 |   0.4896 |
| DPE                         | mias        | 0.1429     |     0.0357 |   0.4301 |
| ce_loss                     | mias        | 0.1429     |     0.0357 |   0.5982 |
| focal_loss                  | mias        | 0.1429     |     0.0357 |   0.5655 |
| paco                        | mias        | 0.2143     |     0.0996 |   0.619  |
| gpaco                       | mias        | 0.1786     |     0.098  |   0.5759 |
| bpaco_original              | mias        | 0.1429     |     0.0615 |   0.6429 |
| FPaCo (Heatmap)             | mias        | **0.3929** |     0.3413 |   0.625  |
| BioMedCLIP                  | octa        | 0.2041     |     0.1294 |   0.5374 |
| CoOp                        | octa        | 0.1429     |     0.0357 |   0.6166 |
| Tip-Adapter (Zero-Shot)     | octa        | 0.2041     |     0.1294 |   0.5374 |
| Tip-Adapter (Training-Free) | octa        | 0.2041     |     0.1119 |   0.5787 |
| Tip-Adapter (Fine-Tuned)    | octa        | 0.2857     |     0.2251 |   0.6827 |
| DPE                         | octa        | 0.1837     |     0.1005 |   0.5316 |
| ce_loss                     | octa        | 0.3878     |     0.2717 |   0.7833 |
| focal_loss                  | octa        | 0.3878     |     0.3004 |   0.8037 |
| paco                        | octa        | 0.3469     |     0.2304 |   0.826  |
| gpaco                       | octa        | **0.5306** |     0.5446 |   0.8523 |
| bpaco_original              | octa        | 0.3673     |     0.2302 |   0.8251 |
| FPaCo (Heatmap)             | octa        | **0.5306** |     0.5009 |   0.7721 |
| BioMedCLIP                  | oral_cancer | 0.8333     |     0.8286 |   1      |
| CoOp                        | oral_cancer | **1.0000** |     1      |   1      |
| Tip-Adapter (Zero-Shot)     | oral_cancer | 0.8333     |     0.8286 |   1      |
| Tip-Adapter (Training-Free) | oral_cancer | 0.8333     |     0.8286 |   1      |
| Tip-Adapter (Fine-Tuned)    | oral_cancer | **1.0000** |     1      |   1      |
| DPE                         | oral_cancer | 0.8333     |     0.8286 |   1      |
| ce_loss                     | oral_cancer | **1.0000** |     1      |   1      |
| focal_loss                  | oral_cancer | **1.0000** |     1      |   1      |
| paco                        | oral_cancer | **1.0000** |     1      |   1      |
| gpaco                       | oral_cancer | **1.0000** |     1      |   1      |
| bpaco_original              | oral_cancer | **1.0000** |     1      |   1      |
| FPaCo (Heatmap)             | oral_cancer | **1.0000** |     1      |   1      |