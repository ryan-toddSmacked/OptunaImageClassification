#!/usr/bin/env bash
# setup.sh - Prepares the Python environment for ChipTrainer on Linux

# --- Configuration ---
REQUIRED_PYTHON_VERSION="3.10"
VENV_DIR=".venv"
# Supported CUDA versions and their corresponding PyTorch wheel identifiers
declare -A SUPPORTED_CUDA_VERSIONS=(
    ["12.6"]="cu126"
    ["12.8"]="cu128"
    ["13.0"]="cu130"
)

# --- Color Codes ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Helper Functions ---
function find_python {
    # Try to find the best Python version available
    local required_version=$REQUIRED_PYTHON_VERSION
    local best_python=""
    local best_version=""
    
    # Check for specific python3.x versions (python3.14, python3.13, python3.12, etc.)
    for minor in {14..6}; do
        local py_cmd="python3.$minor"
        if command -v "$py_cmd" &> /dev/null; then
            local ver=$("$py_cmd" --version 2>&1 | awk '{print $2}')
            if printf '%s\n' "$required_version" "$ver" | sort -V -C; then
                if [[ -z "$best_version" ]] || printf '%s\n' "$best_version" "$ver" | sort -V | head -n1 | grep -q "$best_version"; then
                    best_python="$py_cmd"
                    best_version="$ver"
                fi
            fi
        fi
    done
    
    # If no specific version found, try generic python3
    if [[ -z "$best_python" ]] && command -v python3 &> /dev/null; then
        local ver=$(python3 --version 2>&1 | awk '{print $2}')
        if printf '%s\n' "$required_version" "$ver" | sort -V -C; then
            best_python="python3"
            best_version="$ver"
        fi
    fi
    
    if [[ -n "$best_python" ]]; then
        echo "$best_python"
        return 0
    else
        return 1
    fi
}

function test_python_version {
    PYTHON_CMD=$(find_python)
    if [[ -n "$PYTHON_CMD" ]]; then
        version_string=$("$PYTHON_CMD" --version 2>&1 | awk '{print $2}')
        echo -e "${GREEN}Python version $version_string found via $PYTHON_CMD. (OK)${NC}"
        return 0
    else
        echo -e "${RED}Python $REQUIRED_PYTHON_VERSION or newer not found. Please install Python $REQUIRED_PYTHON_VERSION or newer.${NC}"
        return 1
    fi
}

function get_nvidia_gpu {
    echo "Checking for NVIDIA GPU..."
    if command -v lspci &> /dev/null; then
        if lspci | grep -i nvidia &> /dev/null; then
            gpu_name=$(lspci | grep -i nvidia | grep -i vga | head -n 1 | cut -d':' -f3)
            echo -e "${GREEN}NVIDIA GPU found:$gpu_name${NC}"
            return 0
        else
            echo -e "${YELLOW}No NVIDIA GPU found.${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}lspci command not found. Cannot check for GPU.${NC}"
        return 1
    fi
}

function get_cuda_version {
    echo "Checking for CUDA version..." >&2
    if command -v nvcc &> /dev/null; then
        nvcc_output=$(nvcc --version 2>&1)
        if [[ $? -eq 0 ]]; then
            version=$(echo "$nvcc_output" | grep -oP 'release \K[0-9]+\.[0-9]+')
            if [[ -n "$version" ]]; then
                echo -e "${GREEN}CUDA Toolkit version $version found via nvcc.${NC}" >&2
                echo "$version"
                return 0
            fi
        fi
    fi
    echo -e "${YELLOW}nvcc not found or failed. Make sure the NVIDIA CUDA Toolkit is installed and 'nvcc' is in your PATH.${NC}" >&2
    return 1
}

# --- Main Script ---
echo "--- ChipTrainer Environment Setup ---"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Verify Python
echo "Step 1: Checking Python version..."
if ! test_python_version; then
    echo -e "${RED}Setup cannot continue. Please install the required Python version.${NC}"
    exit 1
fi

# 2. Create Virtual Environment
if [[ -d "$VENV_DIR" ]]; then
    echo -e "${YELLOW}Step 2: Virtual environment '$VENV_DIR' already exists. Skipping creation.${NC}"
else
    echo "Step 2: Creating Python virtual environment in '$VENV_DIR'..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}Failed to create virtual environment.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Virtual environment created successfully.${NC}"
fi

# 3. Determine PyTorch version to install (CPU vs GPU)
PYTORCH_INSTALL_ARGS="torch torchvision"
PYTORCH_INDEX_URL=""

if get_nvidia_gpu; then
    cuda_version=$(get_cuda_version)
    if [[ -n "$cuda_version" ]] && [[ -n "${SUPPORTED_CUDA_VERSIONS[$cuda_version]}" ]]; then
        cuda_wheel="${SUPPORTED_CUDA_VERSIONS[$cuda_version]}"
        PYTORCH_INDEX_URL="https://download.pytorch.org/whl/$cuda_wheel"
        echo -e "${CYAN}Setup will install PyTorch with CUDA $cuda_version support.${NC}"
    else
        echo -e "${YELLOW}Supported CUDA version not found. Falling back to CPU-only PyTorch.${NC}"
        PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    fi
else
    echo -e "${CYAN}Proceeding with CPU-only PyTorch installation.${NC}"
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
fi

# 4. Install Dependencies
echo "Step 4: Installing dependencies..."

# Define path to Python executable in the virtual environment
# Try both python and python3 in venv
if [[ -f "$SCRIPT_DIR/$VENV_DIR/bin/python" ]]; then
    PYTHON_EXE="$SCRIPT_DIR/$VENV_DIR/bin/python"
elif [[ -f "$SCRIPT_DIR/$VENV_DIR/bin/python3" ]]; then
    PYTHON_EXE="$SCRIPT_DIR/$VENV_DIR/bin/python3"
else
    echo -e "${RED}Python executable not found in virtual environment.${NC}"
    echo "Please ensure the virtual environment was created correctly."
    exit 1
fi

echo "Upgrading pip..."
"$PYTHON_EXE" -m pip install --upgrade pip

echo "Installing PyTorch..."
if [[ -n "$PYTORCH_INDEX_URL" ]]; then
    "$PYTHON_EXE" -m pip install $PYTORCH_INSTALL_ARGS --index-url "$PYTORCH_INDEX_URL"
else
    "$PYTHON_EXE" -m pip install $PYTORCH_INSTALL_ARGS
fi

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Failed to install PyTorch.${NC}"
    exit 1
fi


echo -e "${GREEN}All dependencies installed successfully.${NC}"

echo "--- Setup Complete ---"
echo "To activate the environment in your terminal, run:"
echo "source .venv/bin/activate"