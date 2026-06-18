# Paper-Style CEM Planning Summary

Comparison for selected CDiT-B planning variants. Paper-style local setting uses N120/K5/rep3/OPT1.

Paper NWM-only reference: ATE 1.13, RPE 0.35. Lower is better.

Full N120 results complete: 3/3.

| Variant | Setting | Samples | Status | ATE | RPE | Pos | Yaw | Time | ATE vs N32 | RPE vs N32 | ATE vs paper | RPE vs paper |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CDiT-B 224 No-Text | N32/K5/rep1/OPT1 | 100 | done | 1.162 | 0.355 | 1.718 | 0.315 | 11418.2 | +0.00% | +0.00% | +2.79% | +1.56% |
| CDiT-B 224 No-Text | N120/K5/rep3/OPT1 | 100 | done | 1.130 | 0.348 | 1.627 | 0.265 | 52342.6 | -2.72% | -2.09% | -0.01% | -0.57% |
| CDiT-B 224 No-Text | N120/K5/rep3/OPT1 smoke | 10 | done | 1.205 | 0.360 | 1.700 | 0.401 | 5270.8 | +3.78% | +1.27% | +6.67% | +2.85% |
| CDiT-B 224 Text | N32/K5/rep1/OPT1 | 100 | done | 1.261 | 0.354 | 1.795 | 0.235 | 7293.9 | +0.00% | +0.00% | +11.57% | +1.18% |
| CDiT-B 224 Text | N120/K5/rep3/OPT1 | 100 | done | 1.293 | 0.350 | 1.722 | 0.154 | 52415.7 | +2.56% | -1.17% | +14.42% | -0.00% |
| CDiT-B 128 Text | N32/K5/rep1/OPT1 | 100 | done | 1.261 | 0.358 | 1.823 | 0.250 | 4293.3 | +0.00% | +0.00% | +11.55% | +2.38% |
| CDiT-B 128 Text | N120/K5/rep3/OPT1 | 100 | done | 1.244 | 0.344 | 1.718 | 0.182 | 34230.2 | -1.28% | -3.99% | +10.12% | -1.71% |
| CDiT-B 128 Text | N120/K5/rep3/OPT1 smoke | 1 | done | 0.471 | 0.121 | 0.736 | 0.157 | 388.5 | -62.66% | -66.36% | -58.35% | -65.56% |
