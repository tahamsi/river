#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT=${RIVER_EXTERNAL_ROOT:-external}
PYTHON_BIN=${RIVER_PYTHON_BIN:-python3.11}
mkdir -p "$TARGET_ROOT" "$TARGET_ROOT/envs"

install_checkout() {
  name=$1
  url=$2
  revision=$3
  install_kind=$4
  checkout="$TARGET_ROOT/$name"
  env_dir="$TARGET_ROOT/envs/$name"
  if [ ! -d "$checkout/.git" ]; then
    git clone "$url" "$checkout"
  fi
  git -C "$checkout" fetch --tags origin
  git -C "$checkout" checkout --detach "$revision"
  "$PYTHON_BIN" -m venv "$env_dir"
  "$env_dir/bin/python" -m pip install --upgrade pip
  if [ "$install_kind" = requirements ]; then
    # These pinned 2023-era requirements.txt files include flash-attn/vllm,
    # whose setup.py imports torch at build time. pip's isolated build env
    # does not have torch installed, so the build fails with
    # "ModuleNotFoundError: No module named 'torch'" unless torch is
    # installed into the venv first and the rest is installed with
    # --no-build-isolation so the build step can see it. requirements.txt
    # may pin torch explicitly (CRAG) or only transitively via vllm
    # (self-rag) -- in the latter case use the pin from the same era.
    if grep -qE '^torch==' "$checkout/requirements.txt"; then
      torch_pin=$(grep -E '^torch==' "$checkout/requirements.txt")
    else
      torch_pin="torch==2.1.2"
    fi
    "$env_dir/bin/pip" install "$torch_pin"
    # flash-attn's setup.py compiles CUDA kernels with nvcc, which is absent
    # when only the driver, not the CUDA toolkit, is installed. flash-attn publishes
    # prebuilt wheels on GitHub releases for exact torch/cuda/python/abi
    # combos; installing the matching one first avoids the from-source
    # build (and the CUDA_HOME requirement) entirely.
    flash_attn_pin=$(grep -E '^flash-attn==' "$checkout/requirements.txt" || true)
    if [ -n "$flash_attn_pin" ]; then
      flash_attn_version=${flash_attn_pin#flash-attn==}
      flash_attn_wheel_url=$(python3 - "$flash_attn_version" <<'PY'
import sys, urllib.request, json
version = sys.argv[1]
url = f"https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v{version}"
with urllib.request.urlopen(url) as resp:
    assets = json.load(resp)["assets"]
candidates = [a["browser_download_url"] for a in assets if "cp311" in a["name"] and "torch2.1" in a["name"] and "abiFALSE" in a["name"]]
candidates.sort(key=lambda u: "cu121" not in u)  # prefer exact cu121 match, else closest cu12x
print(candidates[0] if candidates else "")
PY
)
      if [ -n "$flash_attn_wheel_url" ]; then
        "$env_dir/bin/pip" install "$flash_attn_wheel_url"
      fi
    fi
    "$env_dir/bin/pip" install --no-build-isolation -r "$checkout/requirements.txt"
  elif [ "$install_kind" = editable_subpackage ]; then
    # GraphRAG is a monorepo at this pin: the repository root is a private
    # meta-package ("Private :: Do Not Upload", version 0.0.0) that installs no
    # modules, so `pip install -e .` succeeds while leaving `import graphrag`
    # failing. The importable distribution lives under packages/graphrag.
    "$env_dir/bin/pip" install -e "$checkout/packages/graphrag"
  else
    "$env_dir/bin/pip" install -e "$checkout"
  fi
}

install_checkout self-rag https://github.com/AkariAsai/self-rag.git 1fcdc420e48f50a7d7ab1ece5494221b93252e99 requirements
install_checkout CRAG https://github.com/HuskyInSalt/CRAG.git de7c2961ae624a1483a138c5798e1f6d0c4fb0e0 requirements
install_checkout graphrag https://github.com/microsoft/graphrag.git 14a00ad88fc33cf2b52f4f113f25807556f8e25e editable_subpackage
install_checkout letta https://github.com/letta-ai/letta.git b76da9092518cbaa2d09042e52fdcbde69243e18 editable
install_checkout graphiti https://github.com/getzep/graphiti.git 0b4bcf1284ee5fba56b77ed9961568a541e0d418 editable

echo "Installed pinned native sources. Copy infrastructure/native.env.example to a private .env and configure local endpoints."
