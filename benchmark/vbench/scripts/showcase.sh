#!/usr/bin/env bash
# Optional single-video examples, not benchmark generation or evaluation.
set -euo pipefail

usage() {
  echo "Usage: bash scripts/showcase.sh {t2v|i2v} LOCAL_MODEL OUTPUT.mp4 OMNI_SOURCE"
}
fail() { echo "Error: $*" >&2; exit 1; }
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
[[ $# == 4 ]] || { usage >&2; exit 2; }
task=$1
[[ $task == t2v || $task == i2v ]] || fail "Task must be t2v or i2v."
[[ -d $2 ]] || fail "LOCAL_MODEL must be a downloaded checkpoint directory."
model=$(realpath -- "$2")
output=$(realpath -m -- "$3")
[[ $output == *.mp4 ]] || fail "Output must end in .mp4."
[[ ! -e $output && ! -L $output ]] || fail "Refusing to overwrite $output"
[[ -d $4 ]] || fail "OMNI_SOURCE must be a local checkout."
source_dir=$(realpath -- "$4")
pin=7465c0f52bcb46a55f62c1b8613846cda30d0083
[[ $(git -C "$source_dir" rev-parse --show-toplevel) == "$source_dir" ]] ||
  fail "OMNI_SOURCE must be the checkout root."
[[ $(git -C "$source_dir" rev-parse HEAD) == "$pin" ]] ||
  fail "Expected vLLM-Omni commit $pin"
[[ -z $(git -C "$source_dir" ls-files --unmerged) ]] || fail "Source has merge conflicts."
[[ -z $(git -C "$source_dir" status --porcelain --untracked-files=no) ]] ||
  fail "Source must have no tracked modifications (including staged changes)."
if [[ $task == t2v ]]; then
  example="$source_dir/examples/offline_inference/text_to_video/text_to_video.py"
else
  example="$source_dir/examples/offline_inference/image_to_video/image_to_video.py"
fi
[[ -f $example ]] || fail "Official example is missing: $example"
command -v python >/dev/null || fail "Activate the configured Python environment first."
export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
export DIFFUSION_ATTENTION_BACKEND=TRTLLM_ATTN
export PYTHONUNBUFFERED=1
parent=$(dirname -- "$output")
mkdir -p -- "$parent"
negative='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
args=(--model "$model" --negative-prompt "$negative"
      --num-frames 81 --fps 16 --num-inference-steps 40 --enforce-eager
      --output "$output")
if [[ $task == t2v ]]; then
  # The pinned official parser has no --linear-backend flag. Do not invent one.
  args+=(--model-class-name WanPipeline
         --prompt 'Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.'
         --height 720 --width 1280 --guidance-scale 4.0 --guidance-scale-high 3.0
         --boundary-ratio 0.875 --flow-shift 5.0 --seed 42)
else
  image="$parent/input.jpg"
  [[ ! -L $image ]] || fail "Refusing a symlink at $image"
  if [[ ! -e $image ]]; then
    command -v curl >/dev/null || fail "curl is required for the official input image."
    tmp=$(mktemp "$parent/input.jpg.XXXXXX.tmp")
    trap 'rm -f -- "$tmp"' EXIT
    curl --fail --location --retry 3 \
      https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG \
      --output "$tmp"
    [[ -s $tmp ]] || fail "Downloaded input image is empty."
    mv -n -- "$tmp" "$image"
    [[ ! -e $tmp ]] || fail "Input appeared during download; refusing to overwrite it."
    trap - EXIT
  fi
  [[ -f $image && -s $image ]] || fail "Input must be a nonempty regular file: $image"
  # Omit width/height: official max_area=480*832 and multiple-of-16 rounding.
  args+=(--model-class-name WanImageToVideoPipeline --image "$image"
         --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
         --guidance-scale 3.5 --guidance-scale-high 3.5
         --boundary-ratio 0.9 --flow-shift 3.0 --seed 0
         --init-timeout 7200 --stage-init-timeout 7200)
fi
exec python "$example" "${args[@]}"