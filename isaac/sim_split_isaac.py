"""
Stage 2 of the split simulation test (run in the Isaac Gym env, Python 3.8).

Reads the output of `sim_split_metrics.py`, runs the six-direction Isaac Gym test of
`simulation_test.py` with the unchanged `IsaacValidator`, and saves the same result_dict as
`obj2hand_sim` (same keys, same criteria: penetration = E_pen < thres_pen, simulated =
contact kept in at least one of the six directions, valid = penetration * simulated).
Only numpy is needed besides Isaac Gym, so torch never has to run on the GPU here.
"""
import os
import sys
from os.path import join as pjoin

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))

print("import isaac first")
from isaac_validator import IsaacValidator

import argparse
import pickle
from datetime import datetime

import numpy as np


NOW_TIME = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
MESH_DIR = 'data/meshdata'


def sim_test(sim: IsaacValidator, grasp_code: str, sim_input: dict):
    # Same as `sim_test` in simulation_test.py, with the pose conversion done in stage 1.
    sim.reset_simulator()
    B = sim_input['rot'].shape[0]
    # For each test, if the batch size is too large, there might be some error.
    assert B <= 500
    sim.set_asset("open_ai_assets", "hand/shadow_hand.xml",
                  os.path.join(MESH_DIR, grasp_code, "coacd"), "coacd.urdf")
    rots = sim_input['rot']
    translations = sim_input['translation']
    hand_poses = sim_input['hand_pose']
    scales = sim_input['scale']
    for i in range(B):
        sim.add_env(rots[i], translations[i], hand_poses[i], scales[i])
    result = sim.run_sim()
    result = np.array(result).reshape(B, 6)
    return result


def obj2hand_sim(args):
    DEVICE_ID = args.gpu

    with open(args.metrics_file, "rb") as f:
        metrics_file = pickle.load(f)
    metrics_all = metrics_file['metrics']
    key_list = list(metrics_all.keys())
    key_list.sort()
    st = args.start
    ed = min(st + args.step, len(key_list))

    ori_file_name = os.path.basename(metrics_file['test_file'])
    ori_file_name = os.path.splitext(ori_file_name)[0]
    save_file_name = f"sim_st_{st}_ed_{ed-1}_on_{ori_file_name}_t_{NOW_TIME}.pk"

    result_dict = dict()

    success, total = 0, 0
    success_opt, total_opt = 0, 0
    sim = IsaacValidator(gpu=DEVICE_ID)
    i = st
    for grasp_code in key_list[st: ed]:
        m = metrics_all[grasp_code]
        print(f"{i}: Object {grasp_code}:")
        hB = m['hB']
        B = m['E_pen'].shape[0]
        E_pen = m['E_pen']
        penetration = m['penetration']
        q1 = m['q1']

        # simulate
        sim_result = sim_test(sim, grasp_code, m['sim_input'])
        sim_result = np.sum(sim_result, axis=-1)
        simulated = (sim_result > 0)
        valid = penetration * simulated

        penetration = np.split(penetration, [hB])
        sim_result = np.split(sim_result, [hB])
        simulated = np.split(simulated, [hB])
        valid = np.split(valid, [hB])
        q1 = np.split(q1, [hB])
        E_pen = np.split(E_pen, [hB])

        result = {
            'object_sc': m['object_sc'],
            'hand_param_out': m['hand_param_out'],
            'hand_param_opt': m['hand_param_opt'],

            'E_pen': E_pen[0],
            'penetration': penetration[0],
            'simulated': sim_result[0],
            'valid': valid[0],
            'q1': q1[0],

            'E_pen_opt': E_pen[1],
            'penetration_opt': penetration[1],
            'simulated_opt': sim_result[1],
            'valid_opt': valid[1],
            'q1_opt': q1[1],

            'disc_pred': m['disc_pred'],
        }
        result_dict[grasp_code] = result

        # print
        print("Before:")
        print(f"  penetration suc rate: {penetration[0].mean()}")
        print(f"  simulated suc rate: {simulated[0].mean()}")
        print(f"  valid rate: {valid[0].mean()}")
        print(f"  avg q1: {q1[0][penetration[0]].mean()}")
        print(f"  avg pen: {E_pen[0].mean()}")
        print("After Optimization:")
        print(f"  penetration suc rate: {penetration[1].mean()}")
        print(f"  simulated suc rate: {simulated[1].mean()}")
        print(f"  valid rate: {valid[1].mean()}")
        print(f"  avg q1: {q1[1][penetration[1]].mean()}")
        print(f"  avg pen: {E_pen[1].mean()}")

        success += valid[0].sum()
        total += hB

        success_opt += valid[1].sum()
        total_opt += (B - hB)

        i += 1

    print(f"Total for current batch, avg success {success / total}")
    print(f"Total for current batch, avg opt success {success_opt / total_opt}")
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = pjoin(args.out_dir, save_file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved to {save_path}")
    print("Done test.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Eval in Simulation, stage 2 (Isaac Gym)')
    parser.add_argument('--metrics_file', required=True, type=str,
                        help='output of isaac/sim_split_metrics.py')
    parser.add_argument('--out_dir', required=True, type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--step', default=25, type=int)
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU for PhysX (simulation_test.py hard codes DEVICE_ID = 6)')
    args = parser.parse_args()
    obj2hand_sim(args)
