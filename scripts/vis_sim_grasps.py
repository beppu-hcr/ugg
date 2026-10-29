"""
Visualize grasps evaluated by isaac/run_sim_split.sh (run in the `ugg` env from the repository root).

For each selected object, shows 2 successful and 2 failed grasps among the discriminator top-K of
each object-scale pair, for the generated grasps (hand_param_out) and after test-time optimization
(hand_param_opt). Grasps are chosen separately for the two conditions. Saves an interactive HTML and
a PNG per object.
"""
import argparse
import io
import os
import sys
from os.path import join as pjoin

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))
sys.path.append(pjoin(base_dir, '..', 'isaac'))

import numpy as np
import plotly.graph_objects as go
import torch
import trimesh as tm

from summarize_sim_split import load_results, top_k_mask
from utils.hand_model import HandModel

COLOR_SUCCESS = '#2ca02c'
COLOR_FAILURE = '#d62728'
COLOR_OBJECT = 'lightgray'
PANEL_W, PANEL_H = 450, 450
CONDITIONS = [('out', 'generated', ''), ('opt', '+ optimization', '_opt')]


def condition_arrays(r, suffix):
    return {
        'hand_param': r['hand_param' + ('_opt' if suffix else '_out')],
        'valid': r['valid' + suffix],
        'E_pen': r['E_pen' + suffix],
        'simulated': r['simulated' + suffix],
    }


def pick(r, suffix, k, n_each=2):
    """2 successes and 2 failures with the highest disc_pred among the top-k of each object-scale pair."""
    c = condition_arrays(r, suffix)
    disc = r['disc_pred'].reshape(-1)
    top = top_k_mask(r['disc_pred'], r['object_sc'], k)
    picked = []
    for want in [True, False]:
        idx = np.nonzero(top & (c['valid'] == want))[0]
        idx = idx[np.argsort(-disc[idx], kind='stable')][:n_each]
        if len(idx) < n_each:
            return None
        picked += list(idx)
    return picked


def select_objects(results, categories, k):
    selected = []
    for cat in categories:
        for code in sorted(results):
            if not code.startswith(cat + '-'):
                continue
            if all(pick(results[code], s, k) is not None for _, _, s in CONDITIONS):
                selected.append(code)
                break
        else:
            raise RuntimeError(f"no object of {cat} has 2 successes and 2 failures in both conditions")
    return selected


def object_trace(mesh_dir, code, scale):
    mesh = tm.load(pjoin(mesh_dir, code, 'coacd', 'decomposed.obj'), force='mesh', process=False)
    v = np.asarray(mesh.vertices) * scale
    f = np.asarray(mesh.faces)
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                     color=COLOR_OBJECT, opacity=1.0, flatshading=True, name='object')


def panel_figures(hand_model, mesh_dir, code, r, k, device):
    """One single-scene figure per grasp: several 3D scenes in one figure render meshes incompletely."""
    titles, panels = [], []
    for name, label, suffix in CONDITIONS:
        c = condition_arrays(r, suffix)
        for i in pick(r, suffix, k):
            ok = bool(c['valid'][i])
            titles.append(f"{label} #{i}: {'SUCCESS' if ok else 'FAILURE'}<br>"
                          f"scale {r['object_sc'][i, 0]:.2f}, disc {r['disc_pred'][i, 0]:.2f}, "
                          f"pen {c['E_pen'][i] * 100:.2f} cm, sim {int(c['simulated'][i])}/6")
            panels.append((c['hand_param'][i], ok, r['object_sc'][i, 0]))

    hand_model.set_parameters(torch.tensor(np.stack([p[0] for p in panels]), device=device))
    figs = []
    for j, (_, ok, scale) in enumerate(panels):
        traces = [object_trace(mesh_dir, code, scale)]
        traces += hand_model.get_plotly_data(j, opacity=0.85, color=COLOR_SUCCESS if ok else COLOR_FAILURE)
        fig = go.Figure(traces)
        fig.update_layout(title=dict(text=titles[j], x=0.5, font_size=13),
                          scene=dict(aspectmode='data', camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
                                     xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                          showlegend=False, margin=dict(l=0, r=0, t=60, b=0),
                          paper_bgcolor='white', width=PANEL_W, height=PANEL_H)
        figs.append(fig)
    return figs


def header_text(code, k):
    return [code,
            f"top row: generated, bottom row: + optimization; green = success, red = failure; "
            f"2 successes and 2 failures among the discriminator top {k} of 10 per object-scale pair"]


def save_html(path, figs, header):
    parts = [fig.to_html(full_html=False, include_plotlyjs=(j == 0)) for j, fig in enumerate(figs)]
    cells = ''.join(f'<div class="cell">{p}</div>' for p in parts)
    with open(path, 'w') as f:
        f.write(f"""<!doctype html><html><head><meta charset="utf-8"><title>{header[0]}</title>
<style>body{{font-family:sans-serif;margin:16px;background:#fff}}
.grid{{display:grid;grid-template-columns:repeat(4,{PANEL_W}px);gap:4px}}</style></head>
<body><h3>{header[0]}</h3><p>{header[1]}</p><div class="grid">{cells}</div></body></html>""")


def save_png(path, figs, header):
    from PIL import Image, ImageDraw, ImageFont
    tiles = [Image.open(io.BytesIO(fig.to_image(format='png'))).convert('RGB') for fig in figs]
    top = 70
    sheet = Image.new('RGB', (PANEL_W * 4, top + PANEL_H * 2), 'white')
    for j, tile in enumerate(tiles):
        sheet.paste(tile, ((j % 4) * PANEL_W, top + (j // 4) * PANEL_H))
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    draw.text((15, 10), header[0], fill='black', font=font)
    draw.text((15, 38), header[1], fill='black', font=font)
    sheet.save(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True, help='folder of the sim_st_*.pk files')
    parser.add_argument('--test_file', required=True, help='TEST_FILE used for the run')
    parser.add_argument('--categories', nargs='+', default=['core-mug', 'core-pistol', 'core-cellphone'])
    parser.add_argument('--k', type=int, default=7)
    parser.add_argument('--mesh_dir', default='data/meshdata')
    parser.add_argument('--save_dir', default='results/vis')
    args = parser.parse_args()

    test_file_name = os.path.splitext(os.path.basename(args.test_file))[0]
    _, results = load_results(args.out_dir, test_file_name)
    codes = select_objects(results, args.categories, args.k)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hand_model = HandModel(
        mjcf_path='hand_model_mjcf/shadow_hand_wrist_free.xml',
        mesh_path='hand_model_mjcf/meshes',
        contact_points_path='hand_model_mjcf/contact_points.json',
        penetration_points_path='hand_model_mjcf/penetration_points.json',
        n_surface_points=1024,
        device=device
    )
    os.makedirs(args.save_dir, exist_ok=True)
    for n, code in enumerate(codes):
        figs = panel_figures(hand_model, args.mesh_dir, code, results[code], args.k, device)
        header = header_text(code, args.k)
        stem = pjoin(args.save_dir, f"grasps_{n + 1}_{code.split('-')[1]}_{code.split('-')[-1][:8]}")
        save_html(stem + '.html', figs, header)
        save_png(stem + '.png', figs, header)
        print(stem + '.html', stem + '.png')
