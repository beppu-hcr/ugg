"""
Analyze the results of isaac/run_sim_split.sh (the sim_st_*.pk files) without re-running anything.

1. Failure breakdown: for out / opt / opt_topK, split failures into penetration only (E_pen >= thres_pen
   but contact kept), simulation only (E_pen < thres_pen but contact lost in all 6 directions) and both.
   Transitions of each grasp from out to opt (hand_param_opt is hand_param_out[i] optimized, same index).
2. Discriminator: AUC of disc_pred for predicting valid (pooled over all grasps, and the mean over
   object-scale pairs that have both classes, since the top-K selection is done within a pair), and
   success rates per disc_pred bin. disc_pred is computed on hand_param_out in eval_generation.py.

The criteria are the stored penetration / simulated / valid (unchanged from simulation_test.py).
Writes CSV and Markdown tables and a JSON with all numbers to --save_dir.
"""
import argparse
import csv
import json
import os
import sys
from os.path import join as pjoin

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..', 'isaac'))

import numpy as np
from scipy.stats import rankdata

from summarize_sim_split import load_results, top_k_mask

BINS = [(0.0, 0.5, '< 0.5'), (0.5, 0.9, '0.5-0.9'), (0.9, 0.99, '0.9-0.99'), (0.99, 1.01, '>= 0.99')]


def gather(results, k):
    keys = ['disc_pred', 'object_sc', 'penetration', 'simulated', 'valid',
            'penetration_opt', 'simulated_opt', 'valid_opt']
    cat = {key: [] for key in keys + ['pair', 'top']}
    pair_id = 0
    for code in sorted(results):
        r = results[code]
        sc = r['object_sc'].reshape(-1)
        for key in keys:
            cat[key].append(r[key].reshape(-1))
        _, inv = np.unique(sc, return_inverse=True)
        cat['pair'].append(inv + pair_id)
        pair_id += inv.max() + 1
        cat['top'].append(top_k_mask(r['disc_pred'], r['object_sc'], k))
    cat = {key: np.concatenate(v) for key, v in cat.items()}
    for key in ['penetration', 'valid', 'penetration_opt', 'valid_opt']:
        cat[key] = cat[key].astype(bool)
    cat['simulated'] = cat['simulated'] > 0
    cat['simulated_opt'] = cat['simulated_opt'] > 0
    return cat


def failure_kind(pen, sim):
    return {'pen_only': ~pen & sim, 'sim_only': pen & ~sim, 'both': ~pen & ~sim}


def failure_breakdown(cat, k):
    conds = {'out': ('', np.ones_like(cat['valid'])), 'opt': ('_opt', np.ones_like(cat['valid'])),
             f'opt_top{k}': ('_opt', cat['top'])}
    rows = []
    for name, (s, m) in conds.items():
        pen, sim, valid = cat['penetration' + s][m], cat['simulated' + s][m], cat['valid' + s][m]
        n, n_fail = len(valid), int((~valid).sum())
        row = {'condition': name, 'n_grasps': n, 'n_fail': n_fail, 'fail_rate': n_fail / n}
        for kind, mask in failure_kind(pen, sim).items():
            row[f'{kind}_n'] = int(mask.sum())
            row[f'{kind}_of_fail'] = mask.sum() / n_fail
            row[f'{kind}_of_all'] = mask.sum() / n
        rows.append(row)
    return rows


def transitions(cat, mask, label):
    v0, v1 = cat['valid'][mask], cat['valid_opt'][mask]
    rows = []
    for a, b, name in [(True, True, 'success -> success'), (True, False, 'success -> failure'),
                       (False, True, 'failure -> success'), (False, False, 'failure -> failure')]:
        m = (v0 == a) & (v1 == b)
        rows.append({'subset': label, 'transition': name, 'n': int(m.sum()), 'rate': m.sum() / len(v0)})
    # how the success -> failure grasps fail after optimization
    m = v0 & ~v1
    kinds = failure_kind(cat['penetration_opt'][mask][m], cat['simulated_opt'][mask][m])
    breakdown = [{'subset': label, 'failed_by': kind, 'n': int(x.sum()), 'of_success_to_failure': x.sum() / m.sum()}
                 for kind, x in kinds.items()]
    return rows, breakdown


def auc(score, label):
    # Mann-Whitney U with average ranks for ties
    label = label.astype(bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def disc_auc(cat):
    rows = []
    for name, s in [('out', ''), ('opt', '_opt')]:
        valid = cat['valid' + s]
        per_pair = [auc(cat['disc_pred'][cat['pair'] == p], valid[cat['pair'] == p])
                    for p in np.unique(cat['pair'])]
        per_pair = np.array(per_pair)
        ok = ~np.isnan(per_pair)
        rows.append({'condition': name, 'auc_pooled': auc(cat['disc_pred'], valid),
                     'auc_within_pair_mean': per_pair[ok].mean(),
                     'n_pairs_with_both_classes': int(ok.sum()), 'n_pairs': len(per_pair)})
    return rows


def disc_bins(cat):
    rows = []
    d = cat['disc_pred']
    for lo, hi, name in BINS:
        m = (d >= lo) & (d < hi)
        rows.append({'disc_pred': name, 'n_grasps': int(m.sum()), 'share': m.sum() / len(d),
                     'success_rate_out': cat['valid'][m].mean(), 'success_rate_opt': cat['valid_opt'][m].mean()})
    return rows


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return f'{v:.4f}'
    return str(v)


def write_table(save_dir, name, rows, md):
    with open(pjoin(save_dir, name + '.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: fmt(v) for k, v in r.items()} for r in rows)
    md.append(f'## {name}\n')
    md.append('| ' + ' | '.join(rows[0]) + ' |')
    md.append('|' + '---|' * len(rows[0]))
    for r in rows:
        md.append('| ' + ' | '.join(fmt(v) for v in r.values()) + ' |')
    md.append('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True, help='folder of the sim_st_*.pk files')
    parser.add_argument('--test_file', required=True, help='TEST_FILE used for the run')
    parser.add_argument('--n_objects', type=int, default=1126)
    parser.add_argument('--k', type=int, default=7)
    parser.add_argument('--save_dir', default='results/analysis')
    args = parser.parse_args()

    test_file_name = os.path.splitext(os.path.basename(args.test_file))[0]
    files, results = load_results(args.out_dir, test_file_name)
    assert len(results) == args.n_objects, len(results)
    cat = gather(results, args.k)

    tables = {'failure_breakdown': failure_breakdown(cat, args.k)}
    trans, trans_fail = [], []
    for mask, label in [(np.ones_like(cat['valid']), 'all'), (cat['top'], f'top{args.k}')]:
        t, b = transitions(cat, mask, label)
        trans += t
        trans_fail += b
    tables['transitions'] = trans
    tables['success_to_failure_breakdown'] = trans_fail
    tables['disc_auc'] = disc_auc(cat)
    tables['disc_bins'] = disc_bins(cat)

    os.makedirs(args.save_dir, exist_ok=True)
    md = [f'# Tables ({len(results)} objects, {len(cat["valid"])} grasps per condition)\n']
    for name, rows in tables.items():
        write_table(args.save_dir, name, rows, md)
    with open(pjoin(args.save_dir, 'tables.md'), 'w') as f:
        f.write('\n'.join(md))
    with open(pjoin(args.save_dir, 'analysis.json'), 'w') as f:
        json.dump({'test_file': args.test_file, 'files': files, 'k': args.k,
                   'tables': {n: [{k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()}
                                  for r in rows] for n, rows in tables.items()}}, f, indent=2)
    print('\n'.join(md))
