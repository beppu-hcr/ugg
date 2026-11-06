# Contact tables (+ optimization, 56300 grasps)

## All scales

| scale | group | n | contact points within 5 mm | contact points within 10 mm | fingers touching (5 mm) | palm within 5 mm | palm within 10 mm | anchor-hand distance (mm) | anchor-object distance (mm) |
|---|---|---|---|---|---|---|---|---|---|
| all | success | 38615 | 35.3 | 76.9 | 4.16 | 23.4% | 40.1% | 1.40 | 2.51 |
| all | sim-only failure | 10854 | 32.1 | 66.3 | 3.64 | 13.9% | 25.7% | 1.46 | 3.57 |
| all | pen-only failure | 3954 | 41.3 | 69.4 | 4.04 | 13.1% | 21.3% | 2.58 | 4.60 |
| all | both failure | 2877 | 41.7 | 66.2 | 3.87 | 9.4% | 16.4% | 2.09 | 5.40 |

## Per scale

| scale | group | n | contact points within 5 mm | contact points within 10 mm | fingers touching (5 mm) | palm within 5 mm | palm within 10 mm | anchor-hand distance (mm) | anchor-object distance (mm) |
|---|---|---|---|---|---|---|---|---|---|
| 0.06 | success | 9860 | 29.0 | 68.8 | 3.99 | 22.3% | 37.1% | 1.48 | 2.07 |
| 0.06 | sim-only failure | 1072 | 11.6 | 32.0 | 2.16 | 8.9% | 13.0% | 2.44 | 2.69 |
| 0.08 | success | 9223 | 33.6 | 75.3 | 4.14 | 23.1% | 39.7% | 1.43 | 2.29 |
| 0.08 | sim-only failure | 1262 | 20.6 | 49.5 | 3.05 | 10.4% | 16.6% | 2.06 | 2.94 |
| 0.10 | success | 8174 | 37.2 | 79.8 | 4.22 | 23.8% | 41.3% | 1.43 | 2.53 |
| 0.10 | sim-only failure | 1826 | 28.9 | 62.4 | 3.57 | 11.6% | 23.1% | 1.56 | 3.32 |
| 0.12 | success | 6733 | 39.6 | 82.2 | 4.27 | 25.0% | 42.7% | 1.32 | 2.85 |
| 0.12 | sim-only failure | 2701 | 35.7 | 72.7 | 3.89 | 15.8% | 29.5% | 1.29 | 3.48 |
| 0.15 | success | 4625 | 42.4 | 84.7 | 4.31 | 23.6% | 41.5% | 1.24 | 3.32 |
| 0.15 | sim-only failure | 3993 | 40.4 | 78.4 | 4.09 | 16.2% | 30.7% | 1.08 | 4.18 |

## Fingers touching (5 mm), share of grasps

| group | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| success | 0.5% | 1.7% | 5.6% | 14.8% | 28.3% | 49.1% |
| sim-only failure | 3.1% | 6.2% | 11.1% | 17.8% | 26.7% | 35.2% |
| pen-only failure | 0.0% | 1.0% | 6.6% | 19.9% | 32.1% | 40.4% |
| both failure | 0.1% | 4.0% | 9.9% | 18.4% | 30.2% | 37.5% |

## Each finger and the palm (5 mm), share of grasps

| scale | group | ff | mf | rf | lf | th | palm |
|---|---|---|---|---|---|---|---|
| all | success | 81.4% | 87.7% | 83.1% | 79.6% | 84.3% | 23.4% |
| all | sim-only failure | 75.4% | 76.0% | 74.0% | 71.8% | 67.1% | 13.9% |
| all | pen-only failure | 80.5% | 85.2% | 81.4% | 79.7% | 77.6% | 13.1% |
| all | both failure | 80.5% | 82.1% | 78.7% | 78.1% | 67.8% | 9.4% |

## Each finger and the palm (5 mm) per scale

| scale | group | ff | mf | rf | lf | th | palm |
|---|---|---|---|---|---|---|---|
| 0.06 | success | 79.1% | 84.6% | 77.7% | 74.6% | 82.8% | 22.3% |
| 0.06 | sim-only failure | 52.3% | 44.8% | 42.4% | 39.5% | 37.0% | 8.9% |
| 0.08 | success | 80.9% | 86.8% | 82.4% | 78.6% | 84.9% | 23.1% |
| 0.08 | sim-only failure | 66.0% | 61.7% | 62.4% | 59.1% | 55.2% | 10.4% |
| 0.10 | success | 81.8% | 89.1% | 85.1% | 80.7% | 85.7% | 23.8% |
| 0.10 | sim-only failure | 73.4% | 75.9% | 73.2% | 69.4% | 65.2% | 11.6% |
| 0.12 | success | 82.8% | 89.7% | 86.6% | 83.2% | 84.8% | 25.0% |
| 0.12 | sim-only failure | 80.2% | 81.9% | 78.4% | 75.6% | 73.2% | 15.8% |
| 0.15 | success | 84.3% | 90.8% | 87.5% | 85.1% | 83.3% | 23.6% |
| 0.15 | sim-only failure | 82.1% | 84.9% | 83.6% | 82.9% | 75.6% | 16.2% |
