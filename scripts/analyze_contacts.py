"""
Compare the contacts of successful grasps and grasps that fail only in the simulation (after test-time
optimization), from the saved hand_param_opt and the object meshes (run in the `ugg` env from the
repository root). Nothing is re-run.

Per grasp (hand_param_opt, object mesh scaled by object_sc):
  n_cp_5mm / n_cp_10mm   hand contact points (contact_points.json, 140 points on 14 finger links) whose
                         signed distance to the object is >= -5 mm / -10 mm (outside within 5 mm, or inside)
  n_fingers_5mm          fingers (ff, mf, rf, lf, th) with at least one such contact point within 5 mm
  <finger>_5mm           whether that finger touches (within 5 mm)
  palm_5mm / palm_10mm   whether a surface point of robot0:palm is within 5 mm / 10 mm
                         (contact_points.json has no points on the palm, so its surface samples are used)
  anchor_hand_mm         mean |distance| from the 5 generated contact anchors (contact_map) to the hand
                         surface (HandModel.cal_distance, the E_contact term of the test-time optimization)
  anchor_obj_mm          mean |distance| from the anchors to the object surface (for reference)
Writes a per-grasp CSV and Markdown tables to --save_dir.
"""
import argparse
import csv
import os
import pickle
import sys
from os.path import join as pjoin

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))
sys.path.append(pjoin(base_dir, '..', 'isaac'))

import numpy as np
import torch
from tqdm import tqdm

from summarize_sim_split import load_results
from utils.hand_model import HandModel
from utils.object_model import ObjectModel

FINGERS = {'ff': 'robot0:ff', 'mf': 'robot0:mf', 'rf': 'robot0:rf', 'lf': 'robot0:lf', 'th': 'robot0:th'}
PALM = 'robot0:palm'


def link_masks(hand_model):
    links = list(hand_model.mesh)
    cp_link = np.array([links[i] for i in hand_model.global_index_to_link_index])
    finger_of_cp = {f: np.char.startswith(cp_link, prefix) for f, prefix in FINGERS.items()}
    n_sp = [hand_model.mesh[l]['surface_points'].shape[0] for l in links]
    st = sum(n_sp[:links.index(PALM)])
    palm_sp = slice(st, st + n_sp[links.index(PALM)])
    return finger_of_cp, palm_sp


@torch.no_grad()
def contact_features(hand_model, object_model, code, hand_param, object_sc, contact_map, finger_of_cp, palm_sp, device):
    hand_model.set_parameters(torch.tensor(hand_param, dtype=torch.float32, device=device))
    object_model.set_parameters(code + '#-1', object_sc=torch.tensor(object_sc, dtype=torch.float32, device=device))
    cp = hand_model.get_contact_candidates()
    sp = hand_model.get_surface_points()[:, palm_sp]
    anchors = torch.tensor(contact_map, dtype=torch.float32, device=device)
    d_cp = object_model.cal_distance(cp)[0].cpu().numpy()
    d_palm = object_model.cal_distance(sp)[0].cpu().numpy()
    d_anchor_obj = object_model.cal_distance(anchors)[0].abs().cpu().numpy()
    d_anchor_hand = hand_model.cal_distance(anchors).abs().cpu().numpy()

    f = {'n_cp_5mm': (d_cp >= -0.005).sum(1), 'n_cp_10mm': (d_cp >= -0.010).sum(1)}
    touch = {name: (d_cp[:, m] >= -0.005).any(1) for name, m in finger_of_cp.items()}
    f['n_fingers_5mm'] = np.sum([t for t in touch.values()], axis=0)
    for name, t in touch.items():
        f[f'{name}_5mm'] = t
    f['palm_5mm'] = (d_palm >= -0.005).any(1)
    f['palm_10mm'] = (d_palm >= -0.010).any(1)
    f['anchor_hand_mm'] = d_anchor_hand.mean(1) * 1000
    f['anchor_obj_mm'] = d_anchor_obj.mean(1) * 1000
    return f


GROUP_COLS = [('n_cp_5mm', 'contact points within 5 mm', '{:.1f}'),
              ('n_cp_10mm', 'contact points within 10 mm', '{:.1f}'),
              ('n_fingers_5mm', 'fingers touching (5 mm)', '{:.2f}'),
              ('palm_5mm', 'palm within 5 mm', '{:.1%}'),
              ('palm_10mm', 'palm within 10 mm', '{:.1%}'),
              ('anchor_hand_mm', 'anchor-hand distance (mm)', '{:.2f}'),
              ('anchor_obj_mm', 'anchor-object distance (mm)', '{:.2f}')]


def group_table(rows, groups, by_scale):
    md = []
    head = ['scale', 'group', 'n'] + [c[1] for c in GROUP_COLS]
    md.append('| ' + ' | '.join(head) + ' |')
    md.append('|' + '---|' * len(head))
    scales = sorted(set(r['scale'] for r in rows)) if by_scale else ['all']
    for sc in scales:
        for gname, gfun in groups:
            sel = [r for r in rows if gfun(r) and (sc == 'all' or r['scale'] == sc)]
            vals = [fmt.format(np.mean([r[k] for r in sel])) if sel else '-' for k, _, fmt in GROUP_COLS]
            md.append(f"| {sc if sc == 'all' else f'{sc:.2f}'} | {gname} | {len(sel)} | " + ' | '.join(vals) + ' |')
    return md


def dist_table(rows, groups, key, values):
    md = ['| group | ' + ' | '.join(str(v) for v in values) + ' |', '|' + '---|' * (len(values) + 1)]
    for gname, gfun in groups:
        sel = np.array([r[key] for r in rows if gfun(r)])
        md.append(f'| {gname} | ' + ' | '.join(f'{(sel == v).mean():.1%}' for v in values) + ' |')
    return md


def finger_table(rows, groups, by_scale=False):
    md = ['| scale | group | ' + ' | '.join(FINGERS) + ' | palm |', '|' + '---|' * (len(FINGERS) + 3)]
    scales = sorted(set(r['scale'] for r in rows)) if by_scale else ['all']
    for sc in scales:
        for gname, gfun in groups:
            sel = [r for r in rows if gfun(r) and (sc == 'all' or r['scale'] == sc)]
            md.append(f"| {sc if sc == 'all' else f'{sc:.2f}'} | {gname} | " + ' | '.join(
                f"{np.mean([r[f + '_5mm'] for r in sel]):.1%}" for f in list(FINGERS) + ['palm']) + ' |')
    return md


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True, help='folder of the sim_st_*.pk files')
    parser.add_argument('--test_file', required=True, help='TEST_FILE used for the run (has contact_map)')
    parser.add_argument('--n_objects', type=int, default=1126)
    parser.add_argument('--mesh_dir', default='data/meshdata')
    parser.add_argument('--save_dir', default='results/analysis')
    args = parser.parse_args()

    test_file_name = os.path.splitext(os.path.basename(args.test_file))[0]
    _, results = load_results(args.out_dir, test_file_name)
    assert len(results) == args.n_objects, len(results)
    with open(args.test_file, 'rb') as f:
        gen = pickle.load(f)

    device = torch.device('cuda')
    hand_model = HandModel(
        mjcf_path='hand_model_mjcf/shadow_hand_wrist_free.xml',
        mesh_path='hand_model_mjcf/meshes',
        contact_points_path='hand_model_mjcf/contact_points.json',
        penetration_points_path='hand_model_mjcf/penetration_points.json',
        n_surface_points=1024,
        device=device
    )
    object_model = ObjectModel(data_dir=None, mesh_dir=args.mesh_dir, pc_num_points=16, device=device)
    finger_of_cp, palm_sp = link_masks(hand_model)

    rows = []
    for code in tqdm(sorted(results)):
        r = results[code]
        assert np.allclose(r['hand_param_opt'], gen[code]['hand_param_opt'])
        f = contact_features(hand_model, object_model, code, r['hand_param_opt'], r['object_sc'],
                             gen[code]['contact_map'], finger_of_cp, palm_sp, device)
        for i in range(r['valid_opt'].shape[0]):
            row = {'code': code, 'idx': i, 'scale': float(r['object_sc'][i, 0]),
                   'valid': bool(r['valid_opt'][i]), 'penetration': bool(r['penetration_opt'][i]),
                   'simulated': int(r['simulated_opt'][i])}
            row.update({k: v[i].item() for k, v in f.items()})
            rows.append(row)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(pjoin(args.save_dir, 'contacts_per_grasp.csv'), 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = [('success', lambda r: r['valid']),
              ('sim-only failure', lambda r: r['penetration'] and r['simulated'] == 0),
              ('pen-only failure', lambda r: not r['penetration'] and r['simulated'] > 0),
              ('both failure', lambda r: not r['penetration'] and r['simulated'] == 0)]
    md = [f'# Contact tables (+ optimization, {len(rows)} grasps)\n', '## All scales\n']
    md += group_table(rows, groups, False)
    md += ['', '## Per scale\n'] + group_table(rows, groups[:2], True)
    md += ['', '## Fingers touching (5 mm), share of grasps\n'] + dist_table(rows, groups, 'n_fingers_5mm', range(6))
    md += ['', '## Each finger and the palm (5 mm), share of grasps\n'] + finger_table(rows, groups)
    md += ['', '## Each finger and the palm (5 mm) per scale\n'] + finger_table(rows, groups[:2], True)
    with open(pjoin(args.save_dir, 'contact_tables.md'), 'w') as fp:
        fp.write('\n'.join(md) + '\n')
    print('\n'.join(md))
