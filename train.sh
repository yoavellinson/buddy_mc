export TORCH_USE_RTLD_GLOBAL=YES
export CUDA_LAUNCH_BLOCKING=0

torchrun --standalone --nproc_per_node=gpu train.py --config-name=conf_VCTK_binaural.yaml
