# Tables (1126 objects, 56300 grasps per condition)

## failure_breakdown

| condition | n_grasps | n_fail | fail_rate | pen_only_n | pen_only_of_fail | pen_only_of_all | sim_only_n | sim_only_of_fail | sim_only_of_all | both_n | both_of_fail | both_of_all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| out | 56300 | 30676 | 0.5449 | 19980 | 0.6513 | 0.3549 | 4897 | 0.1596 | 0.0870 | 5799 | 0.1890 | 0.1030 |
| opt | 56300 | 17685 | 0.3141 | 3954 | 0.2236 | 0.0702 | 10854 | 0.6137 | 0.1928 | 2877 | 0.1627 | 0.0511 |
| opt_top7 | 39410 | 11279 | 0.2862 | 2593 | 0.2299 | 0.0658 | 6890 | 0.6109 | 0.1748 | 1796 | 0.1592 | 0.0456 |

## transitions

| subset | transition | n | rate |
|---|---|---|---|
| all | success -> success | 23343 | 0.4146 |
| all | success -> failure | 2281 | 0.0405 |
| all | failure -> success | 15272 | 0.2713 |
| all | failure -> failure | 15404 | 0.2736 |
| top7 | success -> success | 17766 | 0.4508 |
| top7 | success -> failure | 1534 | 0.0389 |
| top7 | failure -> success | 10365 | 0.2630 |
| top7 | failure -> failure | 9745 | 0.2473 |

## success_to_failure_breakdown

| subset | failed_by | n | of_success_to_failure |
|---|---|---|---|
| all | pen_only | 274 | 0.1201 |
| all | sim_only | 1926 | 0.8444 |
| all | both | 81 | 0.0355 |
| top7 | pen_only | 192 | 0.1252 |
| top7 | sim_only | 1287 | 0.8390 |
| top7 | both | 55 | 0.0359 |

## disc_auc

| condition | auc_pooled | auc_within_pair_mean | n_pairs_with_both_classes | n_pairs |
|---|---|---|---|---|
| out | 0.6666 | 0.5924 | 4992 | 5630 |
| opt | 0.6914 | 0.5891 | 4575 | 5630 |

## disc_bins

| disc_pred | n_grasps | share | success_rate_out | success_rate_opt |
|---|---|---|---|---|
| < 0.5 | 27692 | 0.4919 | 0.3245 | 0.5645 |
| 0.5-0.9 | 22425 | 0.3983 | 0.5872 | 0.8093 |
| 0.9-0.99 | 5176 | 0.0919 | 0.5796 | 0.7975 |
| >= 0.99 | 1007 | 0.0179 | 0.4667 | 0.7021 |
