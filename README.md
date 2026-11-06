# UGG: Unified Generative Grasping

This repository contains PyTorch implementation of the paper "[UGG: Unified Generative Grasping](https://arxiv.org/abs/2311.16917)", ECCV 2024.

TL;DR: We present a unified modeling of hand-object interaction leverating diffusion model with our proposed *contact anchors* and *physics discriminator*.

![generation](docs/generation.png)
*Figure: Left: grasp generation; Middle: object generation; Right: joint generation*

> **RTX 50-series (Blackwell) GPUs:** the original environment does not run on sm_120. See
> [Setup on RTX 50-series (Blackwell) GPUs](#setup-on-rtx-50-series-blackwell-gpus) for the steps used on an
> RTX 5090, and [`results/results.md`](results/results.md) for the obj2hand results obtained with them.

## Installation

### Training and Generation

This repository is developed on a machine with Ubuntu 20.04.4 LTS and CUDA 11.4.
To set up the required environment, follow these steps:

1. Create a new Anaconda environment named `ugg`:
    ```shell
    conda env create -f environment/environment.yaml
    conda activate ugg
    ```
2. Install third party resources:
    ```shell
    cd thirdparty/pytorch_kinematics
    pip install -e . && cd ..
    git clone https://github.com/wrc042/TorchSDF.git
    cd TorchSDF && git checkout 0.1.0 && pip install -e . && cd ..
    cd ../LION && python build_pkg.py
    ```

### Evaluation with IsaacGym

We recommend using Docker for evaluation with [IsaacGym](https://developer.nvidia.com/isaac-gym). Please download the [IsaacGym](https://developer.nvidia.com/isaac-gym) from the official source. Then, copy files under `environment/docker` to your download `isaacgym/docker` and build the docker.
```shell
cp environment/docker/* isaacgym/docker/
cd isaacgym/docker
sh build_isaac.sh
sh run_isaac.sh
```

## Setup on RTX 50-series (Blackwell) GPUs
> **Note:** These steps were verified by following them from start to finish in a fresh clone with newly
> created conda environments, through the 2-object check of step 4 (inference and simulation test).
> DexGraspNet was downloaded with the commands below. The checkpoint zip and the Isaac Gym archive were
> taken from an existing local copy rather than downloaded. The full 1126-object run, the analysis
> scripts on the full results and the visualization (with the optional kaleido) were run in the
> environment the steps were derived from, not in the fresh one.

The environment above (PyTorch 1.12 + CUDA 11.3) does not run on sm_120 GPUs such as the RTX 5090.
The steps below were used on an RTX 5090 (driver 595, Ubuntu 24.04) for **obj2hand inference with the
released checkpoints** and the Isaac Gym simulation test. Training, hand2obj and joint generation were not
tested. The results are in [`results/results.md`](results/results.md).

Two conda environments are needed, because Isaac Gym Preview 4 only supports Python ≤ 3.8, and no PyTorch
build for Python 3.8 supports sm_120:

| Env | Python | PyTorch | Used for |
|---|---|---|---|
| `ugg` | 3.10 | 2.7.1+cu128 | inference (`eval_generation.py`), stage 1 of the simulation test (penetration, q1), analysis, visualization |
| `ugg-isaac` | 3.8 | none | stage 2 of the simulation test (Isaac Gym; PhysX runs on the GPU) |

The original `isaac/simulation_test.py` computes penetration with the hand model (TorchSDF, CUDA only) in the
same process as Isaac Gym, so it cannot run on these GPUs. It is split into two scripts that run in the two
environments: `isaac/sim_split_metrics.py` (stage 1, the non-Isaac part of `simulation_test.py`, unchanged)
and `isaac/sim_split_isaac.py` (stage 2, the unchanged `IsaacValidator`). The criteria are those of the
original code. Stage 2 needs only NumPy besides Isaac Gym, so the Isaac env has no PyTorch.

Requirements: an NVIDIA driver for CUDA 12.8 (≥ 570), conda, and a system C/C++ compiler (tested with gcc 13.3),
`git`, `wget` and `unzip`. Run all commands from the repository root:
```
git clone https://github.com/beppu-hcr/ugg.git
cd ugg
```

### 1. `ugg` environment (inference and metrics)
CUDA 12.8 (with `nvcc`) is installed into the conda env, so no system CUDA toolkit is needed:
```
conda create -y -n ugg -c nvidia/label/cuda-12.8.1 -c conda-forge \
    python=3.10 cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-cccl
conda activate ugg
```
The CUDA extensions need some variables (below). Install the scripts that set them on `conda activate`, and
re-activate:
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d
cp environment/rtx5090/activate_ugg.sh   $CONDA_PREFIX/etc/conda/activate.d/zz_ugg_cuda.sh
cp environment/rtx5090/deactivate_ugg.sh $CONDA_PREFIX/etc/conda/deactivate.d/zz_ugg_cuda.sh
conda deactivate && conda activate ugg
```
They set:
- `CUDA_HOME`, `CPATH`, `LIBRARY_PATH`, `LD_LIBRARY_PATH` and `TORCH_CUDA_ARCH_LIST=12.0`, to build the CUDA
  extensions with the env's CUDA 12.8 for sm_120.
- `TORCH_EXTENSIONS_DIR=$CONDA_PREFIX/share/torch_extensions`: LION's pvcnn is compiled on first use with the
  extension name `_pvcnn_backend`. Other projects that ship LION use the same name, and the default cache is
  shared by all envs.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`: since PyTorch 2.6, `torch.load` defaults to `weights_only=True`, and the
  released checkpoints fail with `Unsupported global: GLOBAL easydict.EasyDict`. Lightning 2.0.4's
  `load_from_checkpoint` cannot pass `weights_only=False`, so the old default is restored for the whole env.
  Use it only with trusted checkpoints. PyTorch prints a warning about this variable on every load.

Install PyTorch and the Python packages (versions are those of the tested env):
```
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install "numpy==1.26.4" "setuptools==69.5.1" wheel "ninja==1.13.2" \
    "pytorch-lightning==2.0.4" "torchmetrics==0.11.4" "lightning-utilities==0.15.3" "torch_geometric==2.6.1" \
    "easydict==1.13" "einops==0.8.2" "loguru==0.7.3" "scipy==1.15.3" "trimesh==4.6.4" "transforms3d==0.4.2" \
    "pyyaml==6.0.3" "tqdm==4.70.1" "plotly==7.1.0" "open3d==0.20.0" "pillow==12.3.0" "iopath==0.1.10" \
    "transformations==2025.1.1" "absl-py==2.5.0" "lxml==6.1.3"
pip install "pyg_lib==0.5.0+pt27cu128" "torch_cluster==1.6.3+pt27cu128" -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```
Pinned versions that matter:
- `pytorch-lightning==2.0.4` and `torchmetrics==0.11.4`: the versions of the original `environment/environment.yml`,
  which the code was written for.
- `torch_geometric==2.6.1`: `model/ugg/UViT/pn2_layer.py` uses `fps`; newer releases require `pyg-lib>=0.6`
  for it, which has no PyTorch 2.7 wheel. 2.6.1 falls back to `torch_cluster`.
- `numpy==1.26.4`, `setuptools==69.5.1` (used to build the CUDA extensions below), `trimesh==4.6.4`:
  the tested versions. NumPy 2 and newer setuptools were not tried.
- xformers is not needed: with PyTorch 2.x the attention layers use `scaled_dot_product_attention`
  (see [Differences](#differences-from-the-original-readme-and-code-known-issues)).

Build PyTorch3D from source (about 10 minutes with 16 jobs):
```
FORCE_CUDA=1 MAX_JOBS=16 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
```
Install the bundled `pytorch_kinematics` (a modified 0.3.0 from DexGraspNet; the PyPI releases do not work
with `utils/hand_model.py`):
```
pip install --no-deps -e thirdparty/pytorch_kinematics
```
TorchSDF 0.1.0 does not build with PyTorch 2.x as is. `environment/rtx5090/torchsdf-0.1.0-torch2.patch`
defines `IGNORE_TORCH_VER` in `setup.py` (it is read but never defined, so any PyTorch newer than 1.12.1
fails with `NameError`), and replaces `CHECK_EQ`, which PyTorch 2.x no longer provides, with `TORCH_CHECK`
in `torchsdf/csrc/utils.h`:
```
git clone https://github.com/wrc042/TorchSDF.git thirdparty/TorchSDF
cd thirdparty/TorchSDF && git checkout 0.1.0 && git apply ../../environment/rtx5090/torchsdf-0.1.0-torch2.patch
FORCE_CUDA=1 IGNORE_TORCH_VER=1 MAX_JOBS=16 pip install --no-build-isolation -e .
cd ../..
```
Compile LION's pvcnn (about 2 minutes; it goes to `$TORCH_EXTENSIONS_DIR`):
```
(cd LION && python build_pkg.py)
```

**Optional, for the PNG export of the visualization** (`scripts/vis_sim_grasps.py`): kaleido and the Chrome
it uses. `plotly_get_chrome` puts Chrome in `~/.local/share/choreographer/`, outside the env:
```
pip install "kaleido==1.5.0"
plotly_get_chrome -y
```

### 2. `ugg-isaac` environment (Isaac Gym)
```
conda create -y -n ugg-isaac python=3.8
conda activate ugg-isaac
pip install "numpy==1.24.4"
```
Download [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym) **by hand**: the download requires
registering with NVIDIA. Put `IsaacGym_Preview_4_Package.tar.gz` in the repository root, unpack it to
`IsaacGym_Preview_4_Package/isaacgym/` (git-ignored), and make it importable with a `.pth` file. Installing
it with `pip install -e` would pull in PyTorch, which this env does not need:
```
mkdir IsaacGym_Preview_4_Package && tar xf IsaacGym_Preview_4_Package.tar.gz -C IsaacGym_Preview_4_Package
echo "$PWD/IsaacGym_Preview_4_Package/isaacgym/python" > $(python -c "import site; print(site.getsitepackages()[0])")/isaacgym_local.pth
```
Isaac Gym needs the env's `libpython3.8` on `LD_LIBRARY_PATH` (`isaac/run_sim_split.sh` sets it). Check the import:
```
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python -c "from isaacgym import gymapi; print('Isaac Gym OK')"
conda deactivate
```

### 3. Data and checkpoints
**Checkpoints, splits and discriminator data.** Download the
[Google Drive folder](https://drive.google.com/drive/folders/1hcE9dDOv2Kx4UJTJhwbkhjexSZF98iJk?usp=sharing)
(in the browser, "Download" on the folder gives a zip of about 1.1 GB, `ugg-<date>-001.zip`). Put the zip in
the repository root and move its contents into place:
```
unzip -q ugg-*.zip -d ugg_release
mkdir -p data
mv ugg_release/ugg/checkpoints .
mv ugg_release/ugg/splits ugg_release/ugg/discriminator_data data/
rm -r ugg_release
```
**DexGraspNet** (grasps and object meshes, 373 MB and 346 MB; 1.1 GB and 1.9 GB unpacked). From the
[DexGraspNet mirror](https://mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/):
```
wget -P data https://mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/dexgraspnet.tar.gz
wget -P data https://mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/meshdata.tar.gz
tar xzf data/dexgraspnet.tar.gz -C data && tar xzf data/meshdata.tar.gz -C data
rm data/dexgraspnet.tar.gz data/meshdata.tar.gz
```
To keep the dataset elsewhere, unpack it there and symlink `data/dexgraspnet` and `data/meshdata` to it.
The final layout:
```
checkpoints
├── hand_vae/hand_vae_bs1024_lr004_kl00001_22_32_64_mse_step/hand_vae.ckpt
├── lion/aeb159h_hvae_lion_B32/{cfg.yml,checkpoints/epoch_5999_iters_1667999.pt}
├── ugg/ugg_bs8x256_lr0002_l2_cosine/ugg.ckpt
└── ugg_discriminator/discriminator_on_e1499_bs1024_lr0002_ep80_decoded_oripc_onrefine/discriminator.ckpt
data
├── dexgraspnet/<object code>.npy                (5355 files)
├── meshdata/<object code>/coacd/{coacd.urdf,decomposed.obj,...}   (5751 objects)
├── splits/split_{dexgraspnet,dexgraspobject}_{train,test}.txt
└── discriminator_data/e1499/
IsaacGym_Preview_4_Package/isaacgym/
```
The first inference run writes `data/splits/saved_dexgraspobject_test.pk`.

### 4. Running obj2hand inference and the simulation test
Times are for an RTX 5090. Results go to `results/ugg_generation/ugg_bs8x256_lr0002_l2_cosine/`:
```
R=results/ugg_generation/ugg_bs8x256_lr0002_l2_cosine/test_obj2hand
```
`isaac/run_sim_split.sh <TEST_FILE>` runs both stages of the simulation test on a `.pk` written by
`eval_generation.py`. It activates the envs named by `UGG_ENV` and `ISAAC_ENV` (default `ugg` and
`ugg-isaac`). The output goes to `$R/sim_on_<name of the .pk without the time stamp>/`.

**Check with 2 objects** (inference about 40 seconds, simulation test about 1 minute).
`experiments/ugg_obj2hand_test_range0_2.yaml` is `ugg_obj2hand_test.yaml` with `DATA.RANGE: [0, 2]`:
```
conda activate ugg
python model/ugg/eval_generation.py --cfg experiments/ugg_obj2hand_test_range0_2.yaml
TEST_FILE=$(ls -t $R/test_obj2hand_range0_2_*.pk | head -1)
bash isaac/run_sim_split.sh $TEST_FILE
python isaac/summarize_sim_split.py --out_dir $R/sim_on_test_obj2hand_range0_2 --test_file $TEST_FILE --n_objects 2
```
Each object gets 50 grasps (5 scales × 10). The last command prints the success rates of the generated
grasps (`out`), after test-time optimization (`opt`) and of the discriminator's top 7 and 8 of each 10
(`opt_top7`, `opt_top8`).

**All 1126 test objects.** Inference takes about 5 hours. Run it in `tmux` (or similar) so that it survives
the terminal; it saves the `.pk` every 20 objects:
```
mkdir -p logs
python model/ugg/eval_generation.py --cfg experiments/ugg_obj2hand_test_full.yaml 2>&1 | tee logs/obj2hand_full1126.log
```
The simulation test takes about 52 minutes (stage 1: 12 minutes; stage 2: 41 minutes in 46 processes of 25
objects). Then summarize:
```
TEST_FILE=$(ls -t $R/test_obj2hand_full1126_*.pk | head -1)
bash isaac/run_sim_split.sh $TEST_FILE 2>&1 | tee logs/sim_full1126.log
python isaac/summarize_sim_split.py --out_dir $R/sim_on_test_obj2hand_full1126 --test_file $TEST_FILE \
    --save results/ugg_generation/ugg_bs8x256_lr0002_l2_cosine/sim_summary_full1126.json
```
**Optional analyses** of the saved results (nothing is re-run; `ugg` env). They write to `results/analysis/`
and `results/vis/`:
```
# failure types, transitions out -> opt, discriminator AUC and score bins (seconds)
python scripts/analyze_sim_results.py --out_dir $R/sim_on_test_obj2hand_full1126 --test_file $TEST_FILE
# contacts of successful grasps and of grasps that fail only in the simulation (about 2 minutes)
python scripts/analyze_contacts.py --out_dir $R/sim_on_test_obj2hand_full1126 --test_file $TEST_FILE
# 3D views (HTML) and PNGs of 2 successful and 2 failed grasps for a mug, a pistol and a phone (needs kaleido)
python scripts/vis_sim_grasps.py --out_dir $R/sim_on_test_obj2hand_full1126 --test_file $TEST_FILE
```
The analysis scripts expect 1126 objects by default (`--n_objects`).

### Differences from the original README and code, known issues
- **`environment/environment.yaml`** in the original README is `environment/environment.yml` in the repository.
- **`sh run_sim.sh`**: the script is `isaac/run_sim.sh`, and it calls `isaac/simulate_test.py`, while the file
  is `isaac/simulation_test.py`. Neither is used here; `isaac/run_sim_split.sh` replaces it.
- **GPU index.** `simulation_test.py` hard-codes `DEVICE_ID = 6`. The split scripts use GPU 0 (stage 2 takes `--gpu`).
- **`TEST_FILE`.** When it is empty, `simulation_test.py` evaluates the newest `.pk` in `test_obj2hand/`.
  `run_sim_split.sh` always passes it explicitly (in `sim_on_*/test_file.yaml`).
- **Attention.** With PyTorch 2.x, `model/ugg/UViT/UViT.py` and the discriminator use
  `scaled_dot_product_attention` (`attention mode is flash`) and print "Provided checkpoint only supports
  xformers." The attention has no weights, the checkpoints load normally, and on random fp32 inputs the
  output matched xformers' `memory_efficient_attention` exactly (max. difference 0.0).
- **`weights_only`**: see `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` in step 1.
- **q1 is about 10 times lower than in the paper**, because of the original code. When an object has more than
  20 contact candidates, `cal_q1` (`utils/metrics.py`) shuffles a torch tensor with `random.shuffle`. On a
  tensor this duplicates elements (indexing returns views), so only 9–10 of the 20 selected contacts were
  distinct in the grasps checked. On 30 objects (1500 optimized grasps), replacing only the shuffle with
  `torch.randperm` raised the mean q1 from 0.0041 to 0.088 and the share of grasps with q1 > 0 from 18% to
  64%. The reported results keep the original code. q1 is not part of the success criterion.
- **The test split has 1126 objects**, not 1125: `split_dexgraspobject_test.txt` has no newline after the
  last line, so `wc -l` prints 1125. (`isaac/run_sim.sh` has `total=1125` with `-le`, which covers all 1126.)
- **Random sampling in stage 1.** The original runs one process per 25 objects and resets the seed to 42
  each time; stage 1 runs all objects in one process. The object point clouds used for penetration (sampled
  from the mesh) and the contact selection of q1 therefore use different random numbers than the original.
  The criteria are the same.
- **Not bit-for-bit reproducible.** Re-running inference with the same seed gives slightly different grasps
  on the GPU (e.g. up to 0.016 in `hand_param_out` and 0.16 in `hand_param_opt` for the first 2 objects).
- **Harmless warnings**: `torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly`,
  `None of the inputs have requires_grad=True` (gradient checkpointing at inference), and the PyTorch 2.x
  `FutureWarning`s about `torch.cuda.amp`.
- Notes in Japanese on how the steps were found are in [`SETUP_NOTES.md`](SETUP_NOTES.md).

## Datasets
We provide support for the [DexGraspNet Dataset](https://pku-epic.github.io/DexGraspNet/). To download the dataset, please refer to the dataset website. You may find the dataset split [here](https://github.com/PKU-EPIC/DexGraspNet/issues/4) or download the files with our model checkpoints below. The data to train the discriminator is also available below.

After processing the data, ensure that you have a folder named `data` with the following structure:
```
data
├── dexgraspnet
│   ├── core-bottle-1a7ba1f4c892e2da30711cdbdbc73924.npy
│   ├── ...
├── meshdata
│   ├── core-bottle-1a7ba1f4c892e2da30711cdbdbc73924/cocad
│   │   ├── coacd_convex_piece_0.obj
│   │   ├── ...
│   │   ├── coacd.urdf
│   │   ├── decomposed.obj
│   │   ├── ...
├── splits
│   ├── split_dexgraspnet_test.txt
│   ├── split_dexgraspnet_train.txt
│   ├── split_dexgraspobject_test.txt
│   ├── split_dexgraspobject_train.txt
├── discriminator_data
│   ├── ...
```

## Run the Experiment

To train hand VAE, run
```shell
python model/ugg/hand_vae.py --cfg experiments/ugg_hand_vae_train.yaml
```

To train UGG, run
```shell
python model/ugg/train_generation.py --cfg experiments/ugg_generation_train.yaml
```

To train physics discriminator, run
```shell
python model/ugg/discriminator/latent_discriminator.py --cfg experiments/ugg_discriminator_train.yaml
```

For inference, run
```shell
python model/ugg/eval_generation.py --cfg experiments/ugg_{TASK}_test.yaml
```
You may replace `{TASK}` by `obj2hand`, `hand2obj`, or `joint`. For simplicity, we only support single GPU inference here.

To evaluate the generation result, run
```shell
sh run_sim.sh
```

You may find all results under `results/PROJECT/MODEL_NAME`.

## Model Weights
To use the model, download the weight file [**here**](https://drive.google.com/drive/folders/1hcE9dDOv2Kx4UJTJhwbkhjexSZF98iJk?usp=sharing) and put the `checkpoints` folder under this main foler. Add following configuration to the configuration file:
```yaml
WEIGHT_FILE: checkpoints/ugg/ugg_bs8x256_lr0002_l2_cosine/ugg.ckpt
```

## FAQ
We list some potential problems encoutered by us and users in [faq](docs/faq.md). We recommend to read through it first and we welcome users to enrich it by opening issues.

## Licensing
- Files in the `LION/`, `hand_model_mjcf/`, and `open_ai_assets/` directories are licensed under their own license. The use of IsaacGym should follow the license from the official source. Check them out before using.
- The main project is licensed udner the MIT License.

## Acknowledgement

We would like to express our gratitude to the authors of the following repositories, from which we referenced code:

* [LION](https://github.com/nv-tlabs/LION)
* [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet/tree/main)
* [UViT](https://github.com/baofff/U-ViT/tree/main)

## Citation
If you find this repository useful in your research, please cite
```
@inproceedings{lu2024ugg,
    author = {Lu, Jiaxin and Kang, Hao and Li, Haoxiang and Liu, Bo and Yang, Yiding and Huang, Qixing and Hua, Gang},
    title = {UGG: Unified Generative Grasping},
    year = {2024},
    publisher = {Springer-Verlag},
    address = {Berlin, Heidelberg},
    booktitle = {Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part LXVII},
    pages = {414–433},
    numpages = {20},
    keywords = {Dexterous Grasping, Contact Representation, Generative Model},
    location = {Milan, Italy}
}
```
