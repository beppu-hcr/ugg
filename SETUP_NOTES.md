# RTX 5090（sm_120）での UGG セットアップメモ

README の環境（Python 3.8 / PyTorch 1.12.1 / CUDA 11.3）は sm_120 で動かないため、
`~/grasp2grasp` の `g2g` 環境（Python 3.10 / PyTorch 2.7.1+cu128）を複製して使う。
ここまでで確認できたのは **obj2hand の推論（2物体）まで**。学習と Isaac Gym のシミュレーション評価は未実施。

確認した環境：RTX 5090、ドライバ 595.91.07、Ubuntu 24.04（2026-10-28）。

## 1. conda 環境 `ugg`

`g2g` を複製し、UGG 固有のものを追加する（`g2g` と `grasp2grasp` には手を加えない）。
```
conda create -y -n ugg --clone g2g
conda activate ugg
pip install "pytorch-lightning==2.0.4" "torchmetrics==0.11.4" kornia==0.7.1 yacs transforms3d \
    urdf-parser-py plotly transformations tabulate omegaconf
```
- `g2g` の pytorch-lightning 1.9.5 は UGG（2.0.4 用）では使えないので上げる。
  `lightning-bolts 0.7.0` が `<2.0.0` を要求するという警告が出るが、UGG は lightning-bolts を使わないので無視してよい。
- pytorch_kinematics は同梱の改変版（DexGraspNet 由来の 0.3.0）が必要。`g2g` から入っている pip 版 0.7.5 を外して入れ替える：
  ```
  pip uninstall -y pytorch-kinematics
  cd thirdparty/pytorch_kinematics && pip install --no-deps -e . && cd ../..
  ```

### activate 時の環境変数
`environment/rtx5090/activate_ugg.sh` と `deactivate_ugg.sh` を環境の activate.d / deactivate.d に置く：
```
cp environment/rtx5090/activate_ugg.sh   $CONDA_PREFIX/etc/conda/activate.d/zz_ugg_cuda.sh
cp environment/rtx5090/deactivate_ugg.sh $CONDA_PREFIX/etc/conda/deactivate.d/zz_ugg_cuda.sh
conda deactivate && conda activate ugg
```
設定する内容：
- `CUDA_HOME`、`CPATH`、`LIBRARY_PATH`、`LD_LIBRARY_PATH`、`TORCH_CUDA_ARCH_LIST=12.0`：
  CUDA 拡張を、環境内の CUDA 12.8 で sm_120 向けにビルドするため（g2g と同じ）。
- `TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions_ugg`：LION の pvcnn は JIT ビルドで、
  拡張名 `_pvcnn_backend` が g2g と同じ。既定のキャッシュ（`~/.cache/torch_extensions/py310_cu128`）を
  g2g と共有しないように分ける。
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`：PyTorch 2.6 から `torch.load` の既定が `weights_only=True` になり、
  配布チェックポイント（ハイパーパラメータに `easydict.EasyDict` を含む）が
  `UnpicklingError: Weights only load failed ... Unsupported global: GLOBAL easydict.EasyDict` で読めない。
  リポジトリ内の `torch.load` だけでなく Lightning 2.0.4 の `load_from_checkpoint` も `weights_only` を渡せないため、
  コードを直すのではなく環境変数で以前の既定に戻している。信頼できるチェックポイント（公式配布）にだけ使うこと。
  実行時に「Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected ...」という警告が出るのは想定どおり。

### TorchSDF（0.1.0）
README どおり `thirdparty/TorchSDF` に clone し、PyTorch 2.x 向けに2か所パッチを当ててビルドする
（パッチは `environment/rtx5090/torchsdf-0.1.0-torch2.patch`）。
```
cd thirdparty
git clone https://github.com/wrc042/TorchSDF.git
cd TorchSDF && git checkout 0.1.0
git apply ../../environment/rtx5090/torchsdf-0.1.0-torch2.patch
FORCE_CUDA=1 IGNORE_TORCH_VER=1 MAX_JOBS=16 pip install --no-build-isolation -e .
cd ../..
```
- `setup.py`：対応 PyTorch が `<=1.12.1` で、範囲外のとき `IGNORE_TORCH_VER` を見るが、その変数が未定義で
  `NameError: name 'IGNORE_TORCH_VER' is not defined` になる。環境変数から定義する1行を追加。
- `torchsdf/csrc/utils.h`：`CHECK_EQ(...) << ...` は PyTorch 2.x の c10 にないため
  `error: identifier "CHECK_EQ" is undefined` でビルドが失敗する。`TORCH_CHECK` に置き換え。
- `thirdparty/TorchSDF/` は外部リポジトリなので `.gitignore` に追加した。

### LION（pvcnn）
```
cd LION && python build_pkg.py && cd ..
```
変更なしでビルドできる（`~/.cache/torch_extensions_ugg/_pvcnn_backend/` に出力）。
`torch.cuda.amp.custom_fwd/custom_bwd` の FutureWarning が出るが動作には影響しない。

## 2. データとチェックポイント

- UGG の配布物（Google Drive の zip、1.14 GB）を展開して置く：
  - `checkpoints/`（ugg、hand_vae、ugg_discriminator、lion）
  - `data/splits/`、`data/discriminator_data/`
- DexGraspNet（https://mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/）の
  `dexgraspnet.tar.gz`（373 MB）と `meshdata.tar.gz`（346 MB）を `~/datasets/DexGraspNet` に展開
  （展開後 1.1 GB / 1.9 GB、5355 / 5751 物体）し、シンボリックリンクで置く：
  ```
  ln -s ~/datasets/DexGraspNet/dexgraspnet data/dexgraspnet
  ln -s ~/datasets/DexGraspNet/meshdata    data/meshdata
  ```
  `split_dexgraspobject_test.txt` の1126物体はすべて `meshdata` にある（最終行に改行がないので `wc -l` だと1125と出る）。
- 初回実行時に `data/splits/saved_dexgraspobject_test.pk` が自動生成される。

## 3. obj2hand 推論

2物体だけで試す設定 `experiments/ugg_obj2hand_test_range0_2.yaml`
（`ugg_obj2hand_test.yaml` に `DATA.RANGE: [0, 2]` と `LOG_FILE_NAME: range0_2` を追加したもの）：
```
conda activate ugg
python model/ugg/eval_generation.py --cfg experiments/ugg_obj2hand_test_range0_2.yaml
```
出力：`results/ugg_generation/ugg_bs8x256_lr0002_l2_cosine/test_obj2hand/test_obj2hand_range0_2_<時刻>.pk`

結果（2026-10-28）：
- 2物体で全体 36 秒（起動・読み込み約4秒、1物体あたり約16秒、うち後処理の最適化100ステップが約4.5秒）。
  全1126物体なら約5時間の見込み。
- .pk は物体コードをキーとする dict。1物体あたり 5スケール（0.06, 0.08, 0.1, 0.12, 0.15）× 10 = 50 把持：
  `hand_param_out` / `hand_param_opt` (50, 31)（並進3 + 回転6D 6 + 関節22）、`object_pc` (50, 2048, 3)、
  `object_sc` (50, 1)、`contact_map` (50, 5, 3)、`disc_pred` (50, 1)、
  `seed_dict`（`hand_param_latent_gen` (50, 64)、`hand_R_gen` (50, 6)、`hand_t_gen` (50, 3)、`contact_map_gen` (50, 15)）。
- ファイルサイズは2物体で 2.5 MB（大半は `object_pc`）。1126物体では約1.4 GB の見込み。

### attention について
PyTorch 2.x では `model/ugg/UViT/UViT.py` が `scaled_dot_product_attention`（`attention mode is flash`）を選び、
「Provided checkpoint only supports xformers.」という警告が出る。attention 部分に学習パラメータはなく、
チェックポイントは問題なく読める。fp32 のランダム入力で xformers の `memory_efficient_attention` と比べると
出力の最大差は 0.0 だった。

### そのほかの警告（無害）
- `torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly`、
  `None of the inputs have requires_grad=True`：推論時の gradient checkpointing によるもの。
- `torch.cuda.amp.autocast(args...) is deprecated`（`utils/hand_model.py`）など、PyTorch 2.x の FutureWarning。

### 全1126物体
設定 `experiments/ugg_obj2hand_test_full.yaml`（`ugg_obj2hand_test.yaml` に `LOG_FILE_NAME: full1126` を追加。
`DATA.RANGE` は既定の `[0, -1]` で全物体）。Claude Code の終了で止まらないよう tmux セッション `ugg-full` で実行し、
標準出力を `logs/obj2hand_full1126.log` にも保存する：
```
tmux new-session -d -s ugg-full -c ~/ugg
tmux send-keys -t ugg-full 'conda activate ugg && set -o pipefail && python model/ugg/eval_generation.py --cfg experiments/ugg_obj2hand_test_full.yaml 2>&1 | tee logs/obj2hand_full1126.log' Enter
```
出力：`results/ugg_generation/ugg_bs8x256_lr0002_l2_cosine/test_obj2hand/test_obj2hand_full1126_<時刻>.pk`
（20物体ごとに途中結果で上書き保存される）。スクリプト自身のログは同じディレクトリ階層の
`eval_log/eval_log_full1126_<時刻>.log` にも出る。

## 未対応（シミュレーション評価）
`isaac/simulation_test.py` は Isaac Gym（Python 3.8）と同じプロセスで、CUDA 専用の TorchSDF を使う HandModel を
GPU で動かすため、このままでは RTX 5090 では動かない。ほかにも `isaac/run_sim.sh` が存在しない
`isaac/simulate_test.py` を呼んでいる、`DEVICE_ID = 6` が直書き、といった問題がある。

## 次にやること
- シミュレーション評価（README の手順4）。
  - `simulation_test.py` は `TEST_FILE` が空だと `test_obj2hand/` の中で更新時刻が一番新しい .pk を自動で読む。
    2物体の試験結果などを誤って読まないよう、`TEST_FILE` で全物体の結果
    （`test_obj2hand_full1126_<時刻>.pk`）を明示する。
  - 評価の判定基準は元のコードから変えない。元のコードでは
    `valid = (E_pen < thres_pen) かつ (手と物体をまとめて回転させた6つの向き（重力に対する6方向）のうち1つ以上で、
    sim_step 後も手と物体の接触が残っている)`（`simulation_test.py` の `sim_result.sum(-1) > 0`、
    `isaac_validator.py` の `run_sim`）。q1 は `cal_q1` で計算して記録するだけで、成功の判定には使わない。変えるのはプログラムを2つに分けることだけ：
    E_pen と q1 は `ugg` 環境（Python 3.10、GPU）で計算し、Isaac Gym のシミュレーションだけを
    `grasp2grasp` 環境（Python 3.8）で動かして、結果を元と同じ形式でまとめる。
  - あわせて `run_sim.sh` のスクリプト名（`simulate_test.py` → `simulation_test.py`）と `DEVICE_ID = 6`（→ 0）を直す。
