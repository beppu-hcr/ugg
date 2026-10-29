"""
Summarize the results of isaac/run_sim_split.sh (the sim_st_*.pk files of one run).

Conditions, compared with the paper (Table 1 / Table 5):
  out      generated grasps (hand_param_out), all of them         -> "generation only"
  opt      after test-time optimization (hand_param_opt), all     -> "+TTA" / "UGG (w/o disc)"
  opt_topK hand_param_opt, only the K grasps with the highest disc_pred in each
           object-scale pair (the paper keeps the top 30 of 40 = 75%; here 10 are
           generated per pair, so K = 7 or 8)                     -> "UGG"
The criteria are those stored by sim_split_isaac.py (unchanged from simulation_test.py).
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np


def load_results(out_dir, test_file_name):
    files = sorted(glob.glob(os.path.join(out_dir, f"sim_st_*_on_{test_file_name}_t_*.pk")))
    results = dict()
    for f in files:
        with open(f, 'rb') as fp:
            d = pickle.load(fp)
        dup = set(d) & set(results)
        assert not dup, f"duplicated objects in {f}: {sorted(dup)[:3]}"
        results.update(d)
    return files, results


def top_k_mask(disc_pred, object_sc, k):
    # top k by discriminator score within each object-scale pair
    disc_pred = disc_pred.reshape(-1)
    object_sc = object_sc.reshape(-1)
    mask = np.zeros_like(disc_pred, dtype=bool)
    for sc in np.unique(object_sc):
        idx = np.nonzero(object_sc == sc)[0]
        order = idx[np.argsort(-disc_pred[idx], kind='stable')]
        mask[order[:k]] = True
    return mask


def summarize(results, ks):
    rows = {'out': [], 'opt': []}
    for k in ks:
        rows[f'opt_top{k}'] = []
    for code in sorted(results):
        r = results[code]
        out = dict(valid=r['valid'], penetration=r['penetration'], simulated=r['simulated'] > 0,
                   q1=r['q1'], E_pen=r['E_pen'])
        opt = dict(valid=r['valid_opt'], penetration=r['penetration_opt'], simulated=r['simulated_opt'] > 0,
                   q1=r['q1_opt'], E_pen=r['E_pen_opt'])
        rows['out'].append(out)
        rows['opt'].append(opt)
        for k in ks:
            m = top_k_mask(r['disc_pred'], r['object_sc'], k)
            rows[f'opt_top{k}'].append({key: v[m] for key, v in opt.items()})

    summary = dict()
    for name, per_obj in rows.items():
        cat = {key: np.concatenate([o[key] for o in per_obj]) for key in per_obj[0]}
        pen = cat['penetration']
        summary[name] = {
            'n_grasps': int(cat['valid'].shape[0]),
            'success_rate': float(cat['valid'].mean()),
            'penetration_pass_rate': float(pen.mean()),
            'simulated_pass_rate': float(cat['simulated'].mean()),
            'q1_mean_all': float(cat['q1'].mean()),
            # simulation_test.py prints the q1 average over grasps that pass the penetration test
            'q1_mean_pen_pass': float(cat['q1'][pen].mean()),
            'q1_mean_valid': float(cat['q1'][cat['valid']].mean()),
            'q1_nonzero_rate': float((cat['q1'] > 0).mean()),
            # E_pen is the maximal penetration depth of the object point cloud into the hand, in meters
            'max_pen_cm_mean': float(cat['E_pen'].mean() * 100),
        }
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--test_file', required=True, help='TEST_FILE used for the run')
    parser.add_argument('--n_objects', type=int, default=1126)
    parser.add_argument('--ks', type=int, nargs='+', default=[7, 8])
    parser.add_argument('--save', default=None, help='write the summary as json')
    args = parser.parse_args()

    test_file_name = os.path.splitext(os.path.basename(args.test_file))[0]
    files, results = load_results(args.out_dir, test_file_name)
    with open(args.test_file, 'rb') as f:
        expected = set(pickle.load(f).keys())
    missing = sorted(expected - set(results))
    print(f"{len(files)} files, {len(results)} objects (expected {args.n_objects}), missing {len(missing)}")
    assert len(results) == args.n_objects and not missing, missing[:5]

    summary = summarize(results, args.ks)
    keys = ['n_grasps', 'success_rate', 'penetration_pass_rate', 'simulated_pass_rate',
            'q1_mean_all', 'q1_mean_pen_pass', 'q1_mean_valid', 'q1_nonzero_rate', 'max_pen_cm_mean']
    print('| condition | ' + ' | '.join(keys) + ' |')
    print('|---' * (len(keys) + 1) + '|')
    for name, s in summary.items():
        print(f'| {name} | ' + ' | '.join(
            str(s[k]) if k == 'n_grasps' else f'{s[k]:.4f}' for k in keys) + ' |')
    if args.save:
        with open(args.save, 'w') as f:
            json.dump({'test_file': args.test_file, 'files': files, 'summary': summary}, f, indent=2)
        print(f"saved {args.save}")
