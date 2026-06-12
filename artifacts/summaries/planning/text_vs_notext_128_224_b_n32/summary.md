# Text vs No-Text Planning Summary (CDiT-B)

Full 100-sample RECON planning eval, CDiT-B, N32/K5/OPT1/rep1. Lower is better.

| Image | Condition | ATE | RPE trans | Pos error | Yaw error | Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| 128 | No-Text | 1.258629 | 0.365345 | 1.880550 | 0.288940 | 4313.2 |
| 128 | Text | 1.260505 | 0.358320 | 1.822803 | 0.249904 | 4293.3 |
| 224 | No-Text | 1.161514 | 0.355465 | 1.718338 | 0.315219 | 11418.2 |
| 224 | Text | 1.260712 | 0.354134 | 1.795368 | 0.234796 | 7293.9 |

## Text minus No-Text

| Image | ATE | RPE trans | Pos error | Yaw error |
|---|---:|---:|---:|---:|
| 128 | +0.001877 (+0.15%) | -0.007024 (-1.92%) | -0.057748 (-3.07%) | -0.039035 (-13.51%) |
| 224 | +0.099198 (+8.54%) | -0.001331 (-0.37%) | +0.077030 (+4.48%) | -0.080424 (-25.51%) |
