#!/usr/bin/env bash
# =============================================================================
# MILO -- Setup Mistral 7B with llama.cpp (CUDA GPU offload)
# =============================================================================
#
# This script:
#   1. Installs build dependencies (cmake, git, gcc-12, etc.)
#   2. Checks that CUDA toolkit >= 12.8 is available (required for Blackwell)
#   3. Clones and builds llama.cpp with CUDA support for RTX 5070 Ti
#   4. Downloads Mistral 7B Instruct v0.3 GGUF (Q4_K_M, ~4.37 Go)
#   5. Tests inference with a simple prompt
#   6. Shows how to launch llama-server (OpenAI-compatible API on port 8080)
#
# Target hardware:
#   - NVIDIA RTX 5070 Ti (Blackwell, sm_120, 16 Go VRAM)
#   - WSL2 Ubuntu 22.04, CUDA 13.1, driver 591.86
#
# Model:
#   - bartowski/Mistral-7B-Instruct-v0.3-GGUF (Q4_K_M)
#   - File: Mistral-7B-Instruct-v0.3-Q4_K_M.gguf (~4.37 Go)
#   - VRAM usage: ~4.5 Go with full GPU offload (-ngl 99)
#
# Usage:
#   chmod +x /home/florent/milo/scripts/05_setup_mistral.sh
#   /home/florent/milo/scripts/05_setup_mistral.sh
#
# =============================================================================

set -euo pipefail

# -- Configuration ------------------------------------------------------------

MILO_DIR="/home/florent/milo"
LLAMA_CPP_DIR="${MILO_DIR}/llama.cpp"
MODELS_DIR="${MILO_DIR}/models/mistral-7b"

# Model from bartowski -- Mistral 7B Instruct v0.3, Q4_K_M quantization
# This is the latest instruct-tuned Mistral 7B in GGUF format.
# bartowski provides high-quality quantizations with imatrix calibration.
HF_REPO="bartowski/Mistral-7B-Instruct-v0.3-GGUF"
GGUF_FILE="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"

# Server config
SERVER_HOST="0.0.0.0"
SERVER_PORT="8080"
GPU_LAYERS="99"          # Offload all layers to GPU
CTX_SIZE="8192"          # Mistral 7B supports up to 32k, 8k is practical

# Build config for RTX 5070 Ti (Blackwell)
# sm_120 = Blackwell desktop (RTX 5070 Ti, RTX 5080, RTX 5090)
# We also include sm_89 (Ada) and sm_86 (Ampere) as fallbacks
CUDA_ARCHITECTURES="86;89;120"

# -- Colors -------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# =============================================================================
# STEP 0: Pre-flight checks
# =============================================================================

echo ""
echo "============================================================"
echo "  MILO -- Setup Mistral 7B + llama.cpp (CUDA)"
echo "============================================================"
echo ""

# Check we are on Linux / WSL2
if [[ ! -f /proc/version ]]; then
    err "This script must be run in Linux / WSL2."
fi
info "Running on: $(uname -r)"

# Check NVIDIA GPU is visible
if ! command -v nvidia-smi &> /dev/null; then
    err "nvidia-smi not found. NVIDIA drivers must be installed on Windows host."
fi

info "GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# =============================================================================
# STEP 1: Install build dependencies
# =============================================================================

info "Installing build dependencies..."

sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    cmake \
    git \
    gcc-12 \
    g++-12 \
    wget \
    curl \
    pkg-config \
    > /dev/null 2>&1

ok "Build dependencies installed."

# =============================================================================
# STEP 2: Check / Install CUDA Toolkit
# =============================================================================

info "Checking CUDA toolkit..."

# Try to find nvcc
NVCC_PATH=""
if command -v nvcc &> /dev/null; then
    NVCC_PATH=$(which nvcc)
elif [ -f /usr/local/cuda/bin/nvcc ]; then
    NVCC_PATH="/usr/local/cuda/bin/nvcc"
elif [ -f /usr/lib/cuda/bin/nvcc ]; then
    NVCC_PATH="/usr/lib/cuda/bin/nvcc"
fi

if [ -n "$NVCC_PATH" ]; then
    CUDA_VERSION=$($NVCC_PATH --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    ok "CUDA toolkit found: nvcc $CUDA_VERSION at $NVCC_PATH"
else
    warn "CUDA toolkit (nvcc) not found. Installing CUDA toolkit..."
    echo ""
    echo "  For WSL2 + Blackwell (RTX 5070 Ti), you need CUDA >= 12.8."
    echo "  Installing cuda-toolkit from NVIDIA WSL2 repository..."
    echo ""

    # Install CUDA toolkit for WSL2 (does NOT install drivers -- WSL2 uses Windows driver)
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm -f cuda-keyring_1.1-1_all.deb
    sudo apt-get update -qq
    sudo apt-get install -y -qq cuda-toolkit > /dev/null 2>&1

    # Find the installed nvcc
    if [ -f /usr/local/cuda/bin/nvcc ]; then
        NVCC_PATH="/usr/local/cuda/bin/nvcc"
    else
        # Find any nvcc
        NVCC_PATH=$(find /usr/local/cuda-* -name nvcc -type f 2>/dev/null | head -1 || true)
    fi

    if [ -z "$NVCC_PATH" ]; then
        err "Failed to install CUDA toolkit. Install manually: https://developer.nvidia.com/cuda-downloads"
    fi

    CUDA_VERSION=$($NVCC_PATH --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    ok "CUDA toolkit installed: nvcc $CUDA_VERSION"
fi

# Ensure CUDA bin is in PATH
CUDA_BIN_DIR=$(dirname "$NVCC_PATH")
export PATH="${CUDA_BIN_DIR}:${PATH}"

# Ensure CUDA lib is in LD_LIBRARY_PATH
CUDA_ROOT_DIR=$(dirname "$CUDA_BIN_DIR")
if [ -d "${CUDA_ROOT_DIR}/lib64" ]; then
    export LD_LIBRARY_PATH="${CUDA_ROOT_DIR}/lib64:${LD_LIBRARY_PATH:-}"
fi

echo ""

# =============================================================================
# STEP 3: Clone and build llama.cpp with CUDA
# =============================================================================

info "Setting up llama.cpp..."

if [ -d "${LLAMA_CPP_DIR}" ]; then
    warn "llama.cpp directory already exists at ${LLAMA_CPP_DIR}"
    read -p "  Delete and re-clone? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${LLAMA_CPP_DIR}"
    else
        info "Keeping existing llama.cpp. Skipping to build step..."
    fi
fi

if [ ! -d "${LLAMA_CPP_DIR}" ]; then
    info "Cloning llama.cpp from ggml-org/llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${LLAMA_CPP_DIR}"
    ok "llama.cpp cloned."
fi

info "Building llama.cpp with CUDA support..."
info "  CUDA architectures: ${CUDA_ARCHITECTURES}"
info "  Compilers: gcc-12 / g++-12"
echo ""

# Clean previous build if it exists
rm -rf "${LLAMA_CPP_DIR}/build"

# Configure with CMake
# Key flags:
#   -DGGML_CUDA=ON              : Enable CUDA/GPU acceleration
#   -DGGML_NATIVE=OFF           : Don't optimize for build machine CPU only
#   -CMAKE_CUDA_ARCHITECTURES   : Target sm_120 (Blackwell) + fallbacks
#   -DCMAKE_C_COMPILER=gcc-12   : Use gcc-12 (compatible with CUDA toolkit)
cmake -B "${LLAMA_CPP_DIR}/build" \
    -S "${LLAMA_CPP_DIR}" \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DCMAKE_C_COMPILER=gcc-12 \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DCMAKE_BUILD_TYPE=Release

# Build (use all available cores)
cmake --build "${LLAMA_CPP_DIR}/build" --config Release -j "$(nproc)"

# Verify binaries exist
LLAMA_CLI="${LLAMA_CPP_DIR}/build/bin/llama-cli"
LLAMA_SERVER="${LLAMA_CPP_DIR}/build/bin/llama-server"

if [ ! -f "$LLAMA_CLI" ]; then
    err "Build failed: llama-cli not found at ${LLAMA_CLI}"
fi
if [ ! -f "$LLAMA_SERVER" ]; then
    err "Build failed: llama-server not found at ${LLAMA_SERVER}"
fi

ok "llama.cpp built successfully!"
echo "  llama-cli:    ${LLAMA_CLI}"
echo "  llama-server: ${LLAMA_SERVER}"
echo ""

# =============================================================================
# STEP 4: Download Mistral 7B GGUF model
# =============================================================================

info "Setting up model directory..."
mkdir -p "${MODELS_DIR}"

MODEL_PATH="${MODELS_DIR}/${GGUF_FILE}"

if [ -f "${MODEL_PATH}" ]; then
    ok "Model already downloaded: ${MODEL_PATH}"
    MODEL_SIZE=$(du -h "${MODEL_PATH}" | cut -f1)
    info "  Size: ${MODEL_SIZE}"
else
    info "Downloading Mistral 7B Instruct v0.3 (Q4_K_M, ~4.37 Go)..."
    info "  Repo: ${HF_REPO}"
    info "  File: ${GGUF_FILE}"
    echo ""

    # Method 1: Try huggingface-cli (preferred, supports resume)
    if command -v huggingface-cli &> /dev/null; then
        info "Using huggingface-cli for download..."
        huggingface-cli download "${HF_REPO}" "${GGUF_FILE}" \
            --local-dir "${MODELS_DIR}" \
            --local-dir-use-symlinks False
    else
        # Method 2: Fall back to wget
        info "huggingface-cli not found. Using wget..."
        info "  (Install 'pip install huggingface-hub' for better download support)"
        DOWNLOAD_URL="https://huggingface.co/${HF_REPO}/resolve/main/${GGUF_FILE}"
        wget -c -O "${MODEL_PATH}" "${DOWNLOAD_URL}"
    fi

    if [ ! -f "${MODEL_PATH}" ]; then
        err "Download failed. Check your internet connection and try again."
    fi

    MODEL_SIZE=$(du -h "${MODEL_PATH}" | cut -f1)
    ok "Model downloaded: ${MODEL_PATH} (${MODEL_SIZE})"
fi

echo ""

# =============================================================================
# STEP 5: Test inference with llama-cli
# =============================================================================

info "Testing inference with llama-cli..."
echo ""

TEST_PROMPT="[INST] Translate the following English text to Malagasy: 'Hello, how are you today?' [/INST]"

echo "--- Test prompt ---"
echo "${TEST_PROMPT}"
echo "--- Output ---"

"${LLAMA_CLI}" \
    -m "${MODEL_PATH}" \
    -ngl "${GPU_LAYERS}" \
    -c 512 \
    -n 128 \
    --temp 0.7 \
    --repeat-penalty 1.1 \
    -p "${TEST_PROMPT}" \
    --no-display-prompt \
    2>/dev/null || {
        warn "Inference test failed. This may be a CUDA architecture issue."
        warn "Try building with -DCMAKE_CUDA_ARCHITECTURES=native"
        warn "Or fall back to CPU: remove -ngl flag"
    }

echo ""
echo "--- End of test ---"
echo ""

# =============================================================================
# STEP 6: Summary and server launch instructions
# =============================================================================

echo ""
echo "============================================================"
echo "  SETUP COMPLETE"
echo "============================================================"
echo ""
echo "  Model:        ${MODEL_PATH}"
echo "  llama-cli:    ${LLAMA_CLI}"
echo "  llama-server: ${LLAMA_SERVER}"
echo "  GPU layers:   ${GPU_LAYERS} (full offload)"
echo "  VRAM usage:   ~4.5 Go (model) + ~0.5 Go (context)"
echo ""
echo "============================================================"
echo "  HOW TO START THE API SERVER"
echo "============================================================"
echo ""
echo "  # Start llama-server (OpenAI-compatible API on port ${SERVER_PORT}):"
echo ""
echo "  ${LLAMA_SERVER} \\"
echo "      -m ${MODEL_PATH} \\"
echo "      --host ${SERVER_HOST} \\"
echo "      --port ${SERVER_PORT} \\"
echo "      -ngl ${GPU_LAYERS} \\"
echo "      -c ${CTX_SIZE} \\"
echo "      --threads $(nproc)"
echo ""
echo "  # Then query it:"
echo ""
echo '  curl http://localhost:8080/v1/chat/completions \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{'
echo '      "model": "mistral-7b",'
echo '      "messages": [{"role": "user", "content": "Bonjour, parle-moi du Madagascar."}],'
echo '      "temperature": 0.7'
echo '    }'"'"''
echo ""
echo "  # Or use the Python test script:"
echo "  python3 /home/florent/milo/scripts/05b_test_mistral.py"
echo ""
echo "============================================================"
echo ""

# =============================================================================
# STEP 7 (optional): Auto-launch server in background
# =============================================================================

read -p "Start llama-server now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    info "Starting llama-server on port ${SERVER_PORT}..."
    echo "  (Press Ctrl+C to stop)"
    echo ""

    "${LLAMA_SERVER}" \
        -m "${MODEL_PATH}" \
        --host "${SERVER_HOST}" \
        --port "${SERVER_PORT}" \
        -ngl "${GPU_LAYERS}" \
        -c "${CTX_SIZE}" \
        --threads "$(nproc)"
fi
