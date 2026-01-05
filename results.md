# Experiment Results

![Accuracy Chart](results_chart.png)

## Detailed metrics

| Method                      | Dataset     | Accuracy   |   F1 Score |      AUC |
|:----------------------------|:------------|:-----------|-----------:|---------:|
| BioMedCLIP                  | aptos       | 0.3644     |     0.3594 |   0.7342 |
| CoOp                        | aptos       | 0.5151     |     0.4808 |   0.7403 |
| Tip-Adapter (Zero-Shot)     | aptos       | 0.3644     |     0.3594 |   0.7342 |
| Tip-Adapter (Training-Free) | aptos       | 0.4329     |     0.4168 |   0.764  |
| Tip-Adapter (Fine-Tuned)    | aptos       | 0.4411     |     0.4273 |   0.7684 |
| DPE                         | aptos       | 0.2959     |     0.268  |   0.7051 |
| ce_loss                     | aptos       | 0.7014     |     0.7001 |   0.9229 |
| focal_loss                  | aptos       | 0.7014     |     0.7042 |   0.9283 |
| paco                        | aptos       | 0.6192     |     0.5478 |   0.9247 |
| gpaco                       | aptos       | **0.7452** |     0.743  |   0.9238 |
| bpaco_original              | aptos       | 0.5589     |     0.5259 |   0.8628 |
| FPaCo (NoHeat)              | aptos       | 0.7205     |     0.7196 |   0.909  |
| FPaCo (Heat)                | aptos       | 0.6575     |     0.6549 |   0.883  |
| BioMedCLIP                  | fingerA     | 0.0449     |     0.0106 |   0      |
| CoOp                        | fingerA     | 0.0534     |     0.0314 |   0.5    |
| Tip-Adapter (Zero-Shot)     | fingerA     | 0.0513     |     0.0129 |   0.5116 |
| Tip-Adapter (Training-Free) | fingerA     | 0.0513     |     0.0129 |   0.5186 |
| Tip-Adapter (Fine-Tuned)    | fingerA     | 0.0791     |     0.0676 |   0.5335 |
| DPE                         | fingerA     | 0.0556     |     0.0061 |   0.5231 |
| ce_loss                     | fingerA     | 0.6004     |     0.5676 |   0.9401 |
| focal_loss                  | fingerA     | **0.7297** |     0.5918 |   0.9765 |
| paco                        | fingerA     | 0.6389     |     0.6421 |   0.9523 |
| gpaco                       | fingerA     | 0.6453     |     0.6258 |   0      |
| bpaco_original              | fingerA     | 0.6004     |     0.5721 |   0.9415 |
| FPaCo (NoHeat)              | fingerA     | 0.6581     |     0.6315 |   0.9592 |
| FPaCo (Heat)                | fingerA     | 0.6795     |     0.6696 |   0.9392 |
| BioMedCLIP                  | fingerB     | nan        |   nan      | nan      |
| CoOp                        | fingerB     | nan        |   nan      | nan      |
| Tip-Adapter (Zero-Shot)     | fingerB     | 0.0513     |     0.0129 |   0.5116 |
| Tip-Adapter (Training-Free) | fingerB     | 0.0513     |     0.0129 |   0.5186 |
| Tip-Adapter (Fine-Tuned)    | fingerB     | 0.0791     |     0.0676 |   0.5335 |
| DPE                         | fingerB     | nan        |   nan      | nan      |
| ce_loss                     | fingerB     | 0.5470     |     0.5103 |   0.9062 |
| focal_loss                  | fingerB     | 0.5363     |     0.5006 |   0.9151 |
| paco                        | fingerB     | 0.5214     |     0.5218 |   0.9304 |
| gpaco                       | fingerB     | 0.5812     |     0.554  |   0.9373 |
| bpaco_original              | fingerB     | 0.5000     |     0.4621 |   0.894  |
| FPaCo (NoHeat)              | fingerB     | 0.6175     |     0.5946 |   0.9214 |
| FPaCo (Heat)                | fingerB     | **0.6303** |     0.6279 |   0.9301 |
| BioMedCLIP                  | fingerC     | nan        |   nan      | nan      |
| CoOp                        | fingerC     | nan        |   nan      | nan      |
| Tip-Adapter (Zero-Shot)     | fingerC     | 0.0513     |     0.0129 |   0.5116 |
| Tip-Adapter (Training-Free) | fingerC     | 0.0513     |     0.0129 |   0.5186 |
| Tip-Adapter (Fine-Tuned)    | fingerC     | 0.0791     |     0.0676 |   0.5335 |
| DPE                         | fingerC     | nan        |   nan      | nan      |
| ce_loss                     | fingerC     | 0.4060     |     0.3558 |   0.8922 |
| focal_loss                  | fingerC     | 0.4551     |     0.4247 |   0.9005 |
| paco                        | fingerC     | 0.5235     |     0.53   |   0.9278 |
| gpaco                       | fingerC     | 0.5385     |     0.5078 |   0.9286 |
| bpaco_original              | fingerC     | 0.4380     |     0.4059 |   0.8948 |
| FPaCo (NoHeat)              | fingerC     | 0.5406     |     0.5122 |   0.9197 |
| FPaCo (Heat)                | fingerC     | **0.5598** |     0.5411 |   0.9235 |
| BioMedCLIP                  | mias        | 0.1429     |     0.0806 |   0.4524 |
| CoOp                        | mias        | 0.1429     |     0.0381 |   0.4345 |
| Tip-Adapter (Zero-Shot)     | mias        | 0.1429     |     0.0806 |   0.4524 |
| Tip-Adapter (Training-Free) | mias        | 0.1429     |     0.1196 |   0.4821 |
| Tip-Adapter (Fine-Tuned)    | mias        | 0.2500     |     0.2028 |   0.4896 |
| DPE                         | mias        | 0.1429     |     0.0357 |   0.4301 |
| ce_loss                     | mias        | 0.1429     |     0.0357 |   0.5015 |
| focal_loss                  | mias        | 0.1429     |     0.0357 |   0.4836 |
| paco                        | mias        | 0.2143     |     0.098  |   0.4658 |
| gpaco                       | mias        | 0.1786     |     0.087  |   0.4479 |
| bpaco_original              | mias        | 0.1429     |     0.0357 |   0.4866 |
| FPaCo (NoHeat)              | mias        | 0.2500     |     0.2091 |   0.5193 |
| FPaCo (Heat)                | mias        | **0.2857** |     0.2337 |   0.5372 |
| BioMedCLIP                  | octa        | 0.2041     |     0.1294 |   0.5374 |
| CoOp                        | octa        | 0.2857     |     0.1929 |   0.638  |
| Tip-Adapter (Zero-Shot)     | octa        | 0.2041     |     0.1294 |   0.5374 |
| Tip-Adapter (Training-Free) | octa        | 0.2041     |     0.1119 |   0.5787 |
| Tip-Adapter (Fine-Tuned)    | octa        | 0.2857     |     0.2251 |   0.6827 |
| DPE                         | octa        | 0.1837     |     0.1005 |   0.5316 |
| ce_loss                     | octa        | 0.3878     |     0.2916 |   0.7415 |
| focal_loss                  | octa        | 0.4286     |     0.3467 |   0.7852 |
| paco                        | octa        | 0.3265     |     0.1978 |   0.8013 |
| gpaco                       | octa        | **0.5306** |     0.5446 |   0.8484 |
| bpaco_original              | octa        | 0.3265     |     0.2129 |   0.6842 |
| FPaCo (NoHeat)              | octa        | 0.4898     |     0.4243 |   0.8431 |
| FPaCo (Heat)                | octa        | 0.4490     |     0.4455 |   0.7556 |
| BioMedCLIP                  | oral_cancer | 0.8333     |     0.8286 |   1      |
| CoOp                        | oral_cancer | 0.7500     |     0.7333 |   1      |
| Tip-Adapter (Zero-Shot)     | oral_cancer | 0.8333     |     0.8286 |   1      |
| Tip-Adapter (Training-Free) | oral_cancer | 0.8333     |     0.8286 |   1      |
| Tip-Adapter (Fine-Tuned)    | oral_cancer | **1.0000** |     1      |   1      |
| DPE                         | oral_cancer | 0.8333     |     0.8286 |   1      |
| ce_loss                     | oral_cancer | **1.0000** |     1      |   1      |
| focal_loss                  | oral_cancer | **1.0000** |     1      |   1      |
| paco                        | oral_cancer | **1.0000** |     1      |   1      |
| gpaco                       | oral_cancer | **1.0000** |     1      |   1      |
| bpaco_original              | oral_cancer | **1.0000** |     1      |   1      |
| FPaCo (NoHeat)              | oral_cancer | **1.0000** |     1      |   1      |
| FPaCo (Heat)                | oral_cancer | **1.0000** |     1      |   1      |