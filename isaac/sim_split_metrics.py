"""
Stage 1 of the split simulation test (run in the `ugg` env, Python 3.10, GPU).

`simulation_test.py` computes E_pen / q1 with HandModel (TorchSDF, pytorch3d) and runs
Isaac Gym in the same process, which is not possible on sm_120 GPUs: Isaac Gym needs
Python <= 3.8, where no PyTorch build supports sm_120. This script does the non-Isaac part
of `obj2hand_sim` unchanged and saves, for every object, the metrics and the exact inputs of
`sim_test` (rotation as quaternion, translation, joint angles, scale). Stage 2
(`sim_split_isaac.py`) runs the simulation and combines both into the original result_dict.
"""
import os
import sys
from os.path import join as pjoin

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))

import pickle
from datetime import datetime

import numpy as np
import torch
import transforms3d
from loguru import logger

from utils.hand_model import HandModel
from utils.metrics import cal_pen, cal_q1
from utils.object_model import ObjectModel
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from utils.utils import try_to_device, try_to_torch, try_to_cpu, try_to_numpy


NOW_TIME = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def sim_inputs(hand_param: torch.Tensor, object_sc: torch.Tensor):
    # Same conversion as `sim_test` in simulation_test.py, without the simulator.
    B = hand_param.shape[0]
    hand_param = hand_param.cpu()
    rotations = try_to_numpy(robust_compute_rotation_matrix_from_ortho6d(hand_param[:, 3:9]))
    translations = try_to_numpy(hand_param[:, :3])
    hand_poses = try_to_numpy(hand_param[:, 9:])
    scales = try_to_numpy(object_sc)
    rots = np.stack([transforms3d.quaternions.mat2quat(rotations[i]) for i in range(B)])
    return {
        'rot': rots,
        'translation': translations,
        'hand_pose': hand_poses,
        'scale': scales,
    }


def obj2hand_metrics(cfg):
    DEVICE_ID = 0

    test_cfg = cfg.MODEL.test
    output_path = cfg.OUTPUT_PATH
    st = cfg.start
    step = cfg.step

    device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    hand_model = HandModel(
        mjcf_path='hand_model_mjcf/shadow_hand_wrist_free.xml',
        mesh_path='hand_model_mjcf/meshes',
        contact_points_path='hand_model_mjcf/contact_points.json',
        penetration_points_path='hand_model_mjcf/penetration_points.json',
        n_surface_points=2048,
        device=device
    )
    print(cfg.DATASET.lower())
    object_model = ObjectModel(data_dir=cfg.DATA.DATA_DIR,
                               mesh_dir=cfg.DATA.MESH_DIR,
                               pc_num_points=2048,
                               device=device,
                               dataset_name=cfg.DATASET.lower())

    # Unlike simulation_test.py, the test file must be given explicitly.
    assert len(cfg.TEST_FILE), "Please provide TEST_FILE"
    test_file = cfg.TEST_FILE

    with open(test_file, "rb") as f:
        obj2hand_dict = pickle.load(f)
    key_list = list(obj2hand_dict.keys())
    key_list.sort()
    ed = min(st+step, len(key_list))

    ori_file_name = os.path.basename(test_file)
    ori_file_name = os.path.splitext(ori_file_name)[0]
    save_file_name = f"metrics_st_{st}_ed_{ed-1}_on_{ori_file_name}_t_{NOW_TIME}.pk"

    metrics_dict = dict()
    i = st
    for grasp_code in key_list[st: ed]:
        hand_dict = obj2hand_dict[grasp_code]
        print(f"{i}: Object {grasp_code}:")
        code = grasp_code

        object_sc_np = hand_dict['object_sc']
        object_sc = try_to_torch(object_sc_np)
        object_model.set_parameters(code + "#-1", object_sc=object_sc)

        hand_param_out = hand_dict['hand_param_out']
        hand_param_opt = hand_dict['hand_param_opt']
        hand_param_out = try_to_device(try_to_torch(hand_param_out), device)
        hand_param_opt = try_to_device(try_to_torch(hand_param_opt), device)
        hB = hand_param_out.shape[0]

        # To save time, we test them together.
        hand_for_test = torch.cat([hand_param_out, hand_param_opt], dim=0)

        object_model.extend()
        hand_model.set_parameters(hand_for_test)
        pc_for_test = object_model.object_pc
        sc_for_test = try_to_cpu(object_model.object_sc)
        B = hand_for_test.shape[0]

        # calc e_pen
        E_pen = cal_pen(hand_model, pc_for_test)
        E_pen = E_pen.detach().cpu().numpy()
        penetration = (E_pen < test_cfg.thres_pen)

        # calc q1
        q1, contact_points, contact_normals = cal_q1(test_cfg, B, object_model, hand_model, device, with_contact=True)
        q1 = q1.detach().cpu().numpy()

        metrics_dict[grasp_code] = {
            'object_sc': object_sc_np,
            'hand_param_out': hand_dict['hand_param_out'],
            'hand_param_opt': hand_dict['hand_param_opt'],
            'disc_pred': hand_dict['disc_pred'],
            'hB': hB,
            # over the concatenation [hand_param_out, hand_param_opt]
            'E_pen': E_pen,
            'penetration': penetration,
            'q1': q1,
            'sim_input': sim_inputs(hand_for_test, sc_for_test),
        }
        print(f"  penetration suc rate: {penetration[:hB].mean()} (opt {penetration[hB:].mean()})")
        i += 1

    if cfg.folder is not None:
        os.makedirs(pjoin(output_path, "test_obj2hand", cfg.folder), exist_ok=True)
        save_path = pjoin(output_path, "test_obj2hand", cfg.folder, save_file_name)
    else:
        save_path = pjoin(output_path, "test_obj2hand", save_file_name)
    with open(save_path, 'wb') as f:
        pickle.dump({'test_file': test_file, 'thres_pen': test_cfg.thres_pen, 'metrics': metrics_dict},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved metrics to {}", save_path)
    print("Done metrics.")


if __name__ == '__main__':
    from utils.config import cfg
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    args = parse_args('Eval in Simulation, stage 1 (E_pen, q1)')

    torch.manual_seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    file_suffix = f"_st_{args.start}_ed_{args.start + args.step - 1}"
    if cfg.LOG_FILE_NAME is not None and len(cfg.LOG_FILE_NAME) > 0:
        file_suffix += "_{}".format(cfg.LOG_FILE_NAME)
    file_suffix += f"_{NOW_TIME}"
    full_log_name = f"metrics_log{file_suffix}"

    cfg.start = args.start
    cfg.step = args.step
    cfg.folder = args.folder

    if args.folder is not None:
        os.makedirs(os.path.join(cfg.OUTPUT_PATH, "sim_log", args.folder), exist_ok=True)
        log_path = os.path.join(cfg.OUTPUT_PATH, "sim_log", args.folder, f"{full_log_name}.log")
    else:
        os.makedirs(os.path.join(cfg.OUTPUT_PATH, "sim_log"), exist_ok=True)
        log_path = os.path.join(cfg.OUTPUT_PATH, "sim_log", f"{full_log_name}.log")

    with DupStdoutFileManager(log_path) as _:
        obj2hand_metrics(cfg)
