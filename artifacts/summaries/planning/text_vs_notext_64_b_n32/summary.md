# Text vs No-Text Planning Summary (CDiT-B)

Full 100-sample RECON planning eval, CDiT-B, N32/K5/OPT1/rep1. Lower is better.

| Image | Condition | ATE | RPE trans | Pos error | Yaw error | Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| 64 | No-Text | 1.374498 | 0.393809 | 2.145319 | 0.353528 | 3330.3 |
| 64 | Text | 1.353864 | 0.383624 | 2.028811 | 0.305390 | 3249.5 |

## Text minus No-Text

| Image | ATE | RPE trans | Pos error | Yaw error |
|---|---:|---:|---:|---:|
| 64 | -0.020634 (-1.50%) | -0.010186 (-2.59%) | -0.116508 (-5.43%) | -0.048138 (-13.62%) |
