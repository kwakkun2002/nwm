# Text vs No-Text Planning Summary (CDiT-S)

Full 100-sample RECON planning eval, CDiT-S, N32/K5/OPT1/rep1. Lower is better.

| Image | Condition | ATE | RPE trans | Pos error | Yaw error | Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| 64 | No-Text | 1.373076 | 0.402662 | 2.204868 | 0.378207 | 2354.2 |
| 64 | Text | 1.376772 | 0.400633 | 2.174499 | 0.352027 | 2369.6 |

## Text minus No-Text

| Image | ATE | RPE trans | Pos error | Yaw error |
|---|---:|---:|---:|---:|
| 64 | +0.003696 (+0.27%) | -0.002029 (-0.50%) | -0.030369 (-1.38%) | -0.026180 (-6.92%) |
