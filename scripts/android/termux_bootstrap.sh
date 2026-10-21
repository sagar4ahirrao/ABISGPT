#!/data/data/com.termux/files/usr/bin/bash
# ========================================================================= #
# ABISGPT on-device bootstrap — Android 11+ via Termux + proot-distro       #
#                                                                           #
# Boots a complete biometric API (encoder + PAD + matcher + console) on the #
# phone itself: no laptop, no backend server. The pipeline runs the         #
# BruteForce vector index (exact cosine, tiny in-memory galleries fit the   #
# phone) with the trained ONNX toy-CNN encoder (487 KB, onnxruntime CPU).   #
#                                                                           #
# Run INSIDE Termux:                                                        #
#     bash termux_bootstrap.sh              # first boot (~20-40 min)       #
#     bash termux_bootstrap.sh --no-train   # skip dataset train (ready pgm)#
#                                                                           #
# Tested on device expectations:                                            #
#   * Android 11+, Termux from F-Droid (GitHub builds lack proot webview)   #
#   * ~1.5 GB free storage, 4 GB+ RAM recommended (torch import peak)       #
#   * arm64 (aarch64) CPU — every modern phone                              #
# NOT verified on-device (delivered as best-effort script; see ANDROID.md)  #
# ========================================================================= #
set -euo pipefail

ABI=$(uname -m)
REPO_URL="${ABISGPT_REPO:-https://github.com/OWNER/ABISGPT.git}"
WORK="${ABISGPT_DIR:-$HOME/ABISGPT}"
DISTRO=ubuntu

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$ABI" = "aarch64" ] || die "needs arm64/aarch64 (got $ABI)"

# ----------------------------------------------------------------------- #
# 1. Termux layer: proot-distro + a real Ubuntu userland                   #
# ----------------------------------------------------------------------- #
say "1/5 Termux base packages"
apt update -y && apt install -y proot-distro git

say "2/5 proot Ubuntu userland (rootless, no device changes)"
if ! proot-distro list 2>/dev/null | grep -q "Status: INSTALLED" ; then
    proot-distro install "$DISTRO"
fi

# Everything below runs INSIDE the proot Ubuntu session.
say "3/5 Ubuntu deps: git, python, build tools"
proot-distro login "$DISTRO" -- bash -s -- <<'UBUNTU_COMMON'
    set -euo pipefail
    apt update -y
    apt install -y git python3 python3-venv python3-pip file
    ln -sf "$(command -v python3)" /usr/local/bin/python || true
UBUNTU_COMMON

# Get / refresh the repository inside Ubuntu $HOME (persists in proot fs).
say "4/5 source tree"
proot-distro login "$DISTRO" -- bash -s -- <<UBUNTU_REPO
    set -euo pipefail
    cd ~
    if [ -d ABISGPT/.git ]; then
        cd ABISGPT && git pull --ff-only || true
    else
        git clone "$REPO_URL"
        cd ABISGPT
    fi
UBUNTU_REPO

# ----------------------------------------------------------------------- #
# 5. venv + runtime deps + artifact build (train + export ONNX)            #
# ----------------------------------------------------------------------- #
TRAIN_JOBS="yes"
[ "${1:-}" = "--no-train" ] && TRAIN_JOBS="no"

#           NOTE: we build the trained encoder ON DEVICE once (needs the
#           CPU torch wheel, ~200 MB). After it exists it is cached, so
#           subsequent boots with --no-train start in seconds.
proot-distro login "$DISTRO" -- bash -s -- <<UBUNTU_VENV
    set -euo pipefail
    cd ~/ABISGPT
    python3 -m venv .venv
    . .venv/bin/activate
    pip install --upgrade pip wheel

    # Torch-free runtime first (fast, reproducible: docker/requirements-light.txt)
    pip install -r docker/requirements-light.txt

    if [ "$TRAIN_JOBS" = "yes" ] && [ ! -f artifacts/encoder/face_encoder.onnx ]; then
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        python scripts/prepare_dataset.py --dataset olivetti \
            --out artifacts/dataset/olivetti.npz
        python scripts/train_olivetti.py --epochs 12 --device cpu \
            --out-dir artifacts/olivetti
        python scripts/export_encoder_onnx.py \
            --weights artifacts/olivetti/olivetti_encoder.pt \
            --out artifacts/encoder/face_encoder.onnx
    fi
UBUNTU_VENV

say "5/5 launch script -> ~/abisgpt-serve.sh"
cat > "$HOME/abisgpt-serve.sh" <<'LAUNCH'
#!/data/data/com.termux/files/usr/bin/bash
# Start the ABISGPT API + console on the phone (http://localhost:8000).
exec proot-distro login ubuntu -- bash -lc '
    set -euo pipefail
    cd ~/ABISGPT
    . .venv/bin/activate
    export ABISGPT_INDEX_BACKEND=brute      # exact in-memory vector search
    export ABISGPT_VLM=heuristic            # VLM PAD tier (console proposal)
    export ABISGPT_ENCODER=onnx             # trained toy-CNN ONNX encoder
    : "${ABISGPT_INDEX_URI:=}"              # brute needs no URI
    exec uvicorn abisgpt.api.service:app --host 127.0.0.1 --port 8000
'
LAUNCH
chmod +x "$HOME/abisgpt-serve.sh"

say "Done. Start the service any time with:  ~/abisgpt-serve.sh"
say "Then open http://localhost:8000 in the phone browser (or run curl smoke)."
