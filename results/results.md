# obj2hand results on the DexGraspNet test split (RTX 5090)

These are grasps generated from objects (obj2hand) with the released checkpoints, for all 1126 objects of
`data/splits/split_dexgraspobject_test.txt`. They were evaluated with the Isaac Gym simulation test of the
original code, split into two processes (see the README section "Setup on RTX 50-series (Blackwell) GPUs").
Inference ran on 2026-10-28 and the simulation test on 2026-10-29.

The criteria are the original ones: a grasp is **valid** if its maximal penetration is below 0.5 cm
(`thres_pen = 0.005` m) **and** the object stays in contact with the hand in at least one of the six gravity
directions of the simulation. The software versions differ from the authors' setup (see [Caveats](#caveats)).

## Summary

5 scales × 10 grasps per object, 56,300 grasps per condition.

| Condition | Grasps | **Success (valid)** | Penetration test passed | Simulation test passed | Mean max. penetration (cm) | Mean q1 (see caveats) | Paper: success | Paper: penetration (cm) | Paper: q1 |
|---|---|---|---|---|---|---|---|---|---|
| Generated (`hand_param_out`) | 56,300 | **45.5%** | 54.2% | 81.0% | 0.457 | 0.0037 | 43.6% (Table 5) | – | – |
| + test-time optimization (`hand_param_opt`) | 56,300 | **68.6%** | 87.9% | 75.6% | 0.183 | 0.0035 | 64.1% (UGG w/o disc.) | 0.17 | 0.036 |
| + discriminator, top 7 of 10 | 39,410 | **71.4%** | 88.9% | 78.0% | 0.175 | 0.0035 | 72.7% (UGG) | 0.14 | 0.063 |
| + discriminator, top 8 of 10 | 45,040 | 70.8% | 88.6% | 77.5% | 0.177 | 0.0035 | | | |

- **Discriminator selection.** The paper generates 40 grasps per object and scale and keeps the 30 with the
  highest discriminator score (75%). `ugg_obj2hand_test.yaml` generates 10, so the top 7 and the top 8 of each
  10 approximate it. `disc_pred` is computed on the generated grasps, before optimization. The optimization
  treats each grasp independently, so selecting before or after it gives the same set.
- **Penetration** is the mean over all grasps of `E_pen`, the maximal depth of the object point cloud inside
  the hand. It is stored in meters and shown here in cm.
- **q1** is the mean over all grasps. It is about 10 times lower than in the paper because of a bug in the
  original `cal_q1` (see [Caveats](#caveats)). q1 is not part of the success criterion.

The success rates differ from the paper by +1.9 points (generated), +4.5 points (optimized) and −1.3 points
(discriminator, top 7).

## Run time (RTX 5090)

| Step | Time |
|---|---|
| Inference, 1126 objects (`eval_generation.py`, 50 grasps and 100 optimization steps per object) | 4 h 56 min (15.8 s per object) |
| Simulation test, stage 1 (`sim_split_metrics.py`: penetration, q1) | 11.5 min |
| Simulation test, stage 2 (`sim_split_isaac.py`: Isaac Gym, 46 processes of 25 objects) | 41 min |
| Summary and failure analysis (`summarize_sim_split.py`, `analyze_sim_results.py`) | about 1 s each |
| Contact analysis (`analyze_contacts.py`) | about 2 min |

## Failure types

From `scripts/analyze_sim_results.py` ([`analysis/report.md`](analysis/report.md), in Japanese; the tables are in
[`analysis/tables.md`](analysis/tables.md)). Shares of the failed grasps:

| Condition | Failed | Penetration only | Simulation only | Both |
|---|---|---|---|---|
| Generated | 54.5% | **65.1%** | 16.0% | 18.9% |
| + optimization | 31.4% | 22.4% | **61.4%** | 16.3% |
| + optimization, top 7 | 28.6% | 23.0% | **61.1%** | 15.9% |

Same grasp before and after optimization (56,300 grasps):

| Transition | Grasps |
|---|---|
| success → success | 23,343 (41.5%) |
| success → failure | 2,281 (4.1%); 84% of them fail only in the simulation |
| failure → success | 15,272 (27.1%) |
| failure → failure | 15,404 (27.4%) |

- The optimization mostly removes penetration: penetration failures drop from 45.8% to 12.1% of all grasps.
- In exchange, more grasps lose the object in the simulation: 19.0% fail the simulation test before the
  optimization and 24.4% after.

## Discriminator

| Condition | AUC of `disc_pred` for valid, all grasps | Mean AUC within an object and scale |
|---|---|---|
| Generated | 0.667 | 0.592 |
| + optimization | 0.691 | 0.589 |

| `disc_pred` | Grasps | Success, generated | Success, + optimization |
|---|---|---|---|
| < 0.5 | 27,692 | 32.5% | 56.5% |
| 0.5–0.9 | 22,425 | 58.7% | 80.9% |
| 0.9–0.99 | 5,176 | 58.0% | 79.8% |
| ≥ 0.99 | 1,007 | 46.7% | 70.2% |

- The score separates grasps below 0.5 from the rest, but above 0.5 a higher score does not mean a higher
  success rate.
- The top-k selection ranks the grasps within each object and scale, where the AUC is only about 0.59.
  Selecting the top 7 therefore adds 2.8 points (68.6% → 71.4%), against 8.6 points in the paper (64.1% → 72.7%).
- The pooled AUC is higher partly because it also captures the object scale: larger scales have both
  lower scores and lower success rates (scale 0.15: mean score 0.38, 27.8% success for the generated grasps).

## Contacts of grasps that fail only in the simulation

From `scripts/analyze_contacts.py` ([`analysis/contact_analysis.md`](analysis/contact_analysis.md), in Japanese).
Optimized grasps; contact points are the 140 points of `hand_model_mjcf/contact_points.json` within 5 mm of the
object surface. Each cell shows the value for successful grasps, then for grasps that fail only in the simulation:

| Scale | Failed grasps | Contact points | Fingers touching | Thumb touching | Palm within 5 mm | Anchor–hand distance (mm) |
|---|---|---|---|---|---|---|
| 0.06 | 1,072 | 29.0 / **11.6** | 3.99 / **2.16** | 83% / **37%** | 22% / 9% | 1.48 / **2.44** |
| 0.08 | 1,262 | 33.6 / 20.6 | 4.14 / 3.05 | 85% / 55% | 23% / 10% | 1.43 / 2.06 |
| 0.10 | 1,826 | 37.2 / 28.9 | 4.22 / 3.57 | 86% / 65% | 24% / 12% | 1.43 / 1.56 |
| 0.12 | 2,701 | 39.6 / 35.7 | 4.27 / 3.89 | 85% / 73% | 25% / 16% | 1.32 / 1.29 |
| 0.15 | 3,993 | 42.4 / 40.4 | 4.31 / 4.09 | 83% / 76% | 24% / 16% | 1.24 / 1.08 |

- At small scales (0.06, 0.08) these failures have far fewer contacts, and the hand does not reach the
  generated contact anchors.
- At large scales (0.12, 0.15), where 62% of these failures are, the amount of contact is about the same as in
  successful grasps. The remaining differences are thumb and palm contact and anchors farther from the object
  surface (3.3 mm vs. 4.2 mm at scale 0.15). These failures are therefore probably not caused by too little
  contact. Force closure, which would test this directly, could not be checked because q1 is affected by the
  bug below.

## Reproduction check in fresh environments

The README steps were run in a fresh clone with new conda environments, for the first 2 test objects
(`experiments/ugg_obj2hand_test_range0_2.yaml`, 100 grasps per condition), on 2026-11-06:

| | Generated | + optimization | + discriminator, top 7 |
|---|---|---|---|
| Fresh environments | 54 / 100 | 74 / 100 | 57 / 70 |
| Full run above, same 2 objects | 54 / 100 | 75 / 100 | 56 / 70 |

- The generated grasps differ slightly, because inference is not bit-for-bit reproducible on the GPU (up to
  0.021 in `hand_param_out` and 0.06 in `hand_param_opt`). Two inference runs in the original environment
  differ by about as much (0.016 and 0.16).
- Given the same grasps (those of the full run), the fresh environments reproduced the maximal penetration,
  the number of directions kept in the simulation, and the validity of all 200 grasps exactly. Only q1 differs,
  because `cal_q1` uses Python's `random`, which is not seeded. It also differs between two runs in the
  original environment.

## Run conditions

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (32 GB), driver 595 |
| OS | Ubuntu 24.04, kernel 7.0.0 |
| Model env | Python 3.10, PyTorch 2.7.1+cu128, PyTorch3D 0.7.9, TorchSDF 0.1.0 (patched), pytorch-lightning 2.0.4 |
| Isaac env | Python 3.8, Isaac Gym Preview 4, PhysX on the GPU (the run used an existing Python 3.8 env with Isaac Gym; the README creates `ugg-isaac` with NumPy only) |
| Checkpoints | `checkpoints/ugg/ugg_bs8x256_lr0002_l2_cosine/ugg.ckpt`, discriminator `discriminator_on_e1499_bs1024_lr0002_ep80_decoded_oripc_onrefine` |
| Configs | `experiments/ugg_obj2hand_test_full.yaml` (inference), `experiments/ugg_simulation_test_full.yaml` (simulation test) |
| Test objects | 1126 (`split_dexgraspobject_test.txt`), scales 0.06, 0.08, 0.10, 0.12, 0.15 |

## Caveats
- **q1.** When an object has more than 20 contact candidates, `cal_q1` (`utils/metrics.py`) shuffles a torch
  tensor with `random.shuffle`, which duplicates elements. Only 9–10 of the 20 selected contacts were distinct
  in the grasps checked. On 30 objects, replacing only the shuffle with `torch.randperm` raised the mean q1 from
  0.0041 to 0.088. The results above keep the original code.
- **Attention.** The UViT and the discriminator use PyTorch's `scaled_dot_product_attention` instead of
  xformers. On random fp32 inputs, both gave identical outputs.
- **Random numbers in stage 1.** The original simulation test resets the seed every 25 objects; stage 1 runs all
  objects in one process. The object point clouds used for penetration and the contact selection of q1
  therefore differ from what the original would have drawn.
- **Not bit-for-bit reproducible.** Inference with the same seed gives slightly different grasps when re-run
  on the GPU (e.g. up to 0.16 in the optimized parameters of the first 2 objects).
- **Discriminator top-k.** This approximates the paper's top 30 of 40 with the top 7 or 8 of 10.

## Files

| Path | Contents |
|---|---|
| `results/ugg_generation/ugg_bs8x256_lr0002_l2_cosine/sim_summary_full1126.json` | Summary table above, with the result files it was computed from |
| `results/analysis/tables.md`, `failure_breakdown.csv`, `transitions.csv`, `success_to_failure_breakdown.csv`, `disc_auc.csv`, `disc_bins.csv`, `analysis.json` | Failure types and discriminator analysis |
| `results/analysis/report.md` | Report of that analysis (Japanese) |
| `results/analysis/contact_tables.md`, `contact_analysis.md` | Contact analysis (the report is in Japanese) |

Not included because of size:
- the inference output `test_obj2hand_full1126_*.pk` (1.4 GB)
- the simulation results `sim_on_test_obj2hand_full1126/` (`metrics_st_0_*.pk` and 46 `sim_st_*.pk`)
- the per-grasp contact table `analysis/contacts_per_grasp.csv` (8.8 MB)
- the visualizations in `results/vis/` (HTML of about 13 MB each)
