# UGG (RTX 5090): CUDA extensions build against the CUDA 12.8 in this env
export _UGG_OLD_CPATH="${CPATH:-}" _UGG_OLD_LIBRARY_PATH="${LIBRARY_PATH:-}" _UGG_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_CUDA_ARCH_LIST=12.0
# keep JIT-built extensions (LION pvcnn) separate from the g2g env's cache
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions_ugg
# PyTorch >= 2.6 defaults torch.load(weights_only=True); the released checkpoints store EasyDict
# hyperparameters and Lightning 2.0.4 cannot pass weights_only=False, so restore the old default
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
