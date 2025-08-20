#!/bin/bash
# Script to install problematic packages for Python 3.12 + CUDA 12.8
# Run this after installing requirements.txt

set -e  # Exit on any error

echo "Installing problematic packages for SceneSplat..."
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version 2>/dev/null || echo 'CUDA not found in PATH')"

# Function to try multiple installation methods
try_install() {
    local package_name="$1"
    shift
    local methods=("$@")
    
    echo "Installing $package_name..."
    
    for method in "${methods[@]}"; do
        echo "  Trying: $method"
        if eval "$method"; then
            echo "  ✅ $package_name installed successfully with: $method"
            return 0
        else
            echo "  ❌ Failed: $method"
        fi
    done
    
    echo "  ⚠️ All methods failed for $package_name, skipping..."
    return 1
}

# 1. Install numba (Python 3.12 compatibility issues)
echo "=== Installing numba ==="
try_install "numba" \
    "pip install numba==0.61.2" \
    "pip install numba>=0.59.0" \
    "pip install numba" \
    "echo 'Skipping numba - not critical for inference'"

# 2. Install sharedarray (Python 3.12 compatibility issues)
echo "=== Installing sharedarray ==="
try_install "sharedarray" \
    "pip install sharedarray==3.2.4" \
    "pip install sharedarray" \
    "pip install --upgrade setuptools wheel && pip install sharedarray --no-cache-dir" \
    "pip install git+https://github.com/cleemesser/numpy-sharedmem.git" \
    "echo 'Skipping sharedarray - may affect data loading performance'"

# 3. Install spconv (CUDA compatibility)
echo "=== Installing spconv ==="
try_install "spconv" \
    "pip install spconv-cu118" \
    "pip install spconv-cu124" \
    "pip install spconv-cu117" \
    "pip install spconv-cu116" \
    "echo 'Skipping spconv - will try source compilation later'"

# 4. Install Flash Attention (optional, compilation required)
echo "=== Installing Flash Attention (optional) ==="
try_install "flash-attn" \
    "pip install flash-attn --no-build-isolation" \
    "pip install flash-attn" \
    "echo 'Skipping flash-attn - it is optional for inference'"

# 5. Install git-based packages
echo "=== Installing git-based packages ==="

# OCNN-PyTorch
try_install "ocnn-pytorch" \
    "pip install git+https://github.com/octree-nn/ocnn-pytorch.git" \
    "echo 'Skipping ocnn-pytorch - may not be critical'"

# CLIP
try_install "CLIP" \
    "pip install git+https://github.com/openai/CLIP.git" \
    "pip install clip-by-openai" \
    "echo 'Skipping CLIP - required for text features'"

# Flash Attention from git (if pip version failed)pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp312-cp312-linux_x86_64.whl
    ('sharedarray', 'sharedarray'),
    ('spconv', 'spconv'),
    ('flash_attn', 'flash_attn'),
    ('clip', 'clip'),
]

print('Package Installation Status:')
for name, import_name in packages:
    try:
        __import__(import_name)
        print(f'  ✅ {name}: OK')
    except ImportError:
        print(f'  ❌ {name}: Not installed or not working')
    except Exception as e:
        print(f'  ⚠️ {name}: Error - {e}')
"

echo ""
echo "=== Next Steps ==="
echo "1. If spconv failed, you may need to compile from source:"
echo "   git clone https://github.com/traveller59/spconv.git"
echo "   cd spconv && python setup.py bdist_wheel && pip install dist/*.whl"
echo ""
echo "2. If CLIP failed, text features may not work"
echo ""
echo "3. If sharedarray failed, data loading may be slower"
echo ""
echo "4. Continue with compiling custom CUDA extensions (pointops, pointgroup_ops)"
echo ""
echo "Installation script completed!"