#!/bin/zsh
# Build the minimal FlashVSR mirror run_flashvsr.py executes: upstream files copied UNMODIFIED, pinned, plus weights.
#   FlashVSR  OpenImagingLab/FlashVSR @ cf910c61a60733e610e9c6e8b607f80c3a6c202b (Apache-2.0)
#   weights   HF JunhaoZhuang/FlashVSR-v1.1 (Apache-2.0): diffusion_pytorch_model_streaming_dmd.safetensors (DiT,
#             5,676,070,392 B fp32), LQ_proj_in.ckpt, TCDecoder.ckpt; the fixed prompt tensor posi_prompt.pth is in
#             the repo. Wan2.1_VAE.pth is NOT needed by the tiny pipeline.
# usage: setup_mirror.sh <mirrorDir>      then  FLASHVSR_MIRROR=<mirrorDir> python run_flashvsr.py …
set -euo pipefail
M=${1:?mirror dir}
mkdir -p $M/w
[[ -d $M/src ]] || git clone -q https://github.com/OpenImagingLab/FlashVSR.git $M/src
git -C $M/src checkout -q cf910c61a60733e610e9c6e8b607f80c3a6c202b
D=$M/src/diffsynth
mkdir -p $M/diffsynth/{models,schedulers,pipelines}
for f in models/utils.py models/wan_video_dit.py models/wan_video_vae.py schedulers/flow_match.py \
         pipelines/base.py pipelines/flashvsr_tiny.py; do cp $D/$f $M/diffsynth/$f; done
: > $M/diffsynth/__init__.py; : > $M/diffsynth/schedulers/__init__.py; : > $M/diffsynth/pipelines/__init__.py
echo "ModelManager = None  # imported by the pipeline only for from_model_manager(), never called here" \
  > $M/diffsynth/models/__init__.py
cp $M/src/examples/WanVSR/utils/utils.py $M/fv_utils.py
cp $M/src/examples/WanVSR/utils/TCDecoder.py $M/fv_tcdecoder.py
cp $M/src/examples/WanVSR/prompt_tensor/posi_prompt.pth $M/w/
for f in diffusion_pytorch_model_streaming_dmd.safetensors LQ_proj_in.ckpt TCDecoder.ckpt; do
  [[ -f $M/w/$f ]] || curl -fL -o $M/w/$f "https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/resolve/main/$f"
done
shasum -a 256 $M/w/*
# python deps: torch (MPS), torchvision, einops, safetensors, pillow, numpy, tqdm
