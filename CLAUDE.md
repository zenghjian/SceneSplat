# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


See @docs/ROADMAP.md for current status and next steps.
Task-based development workflow with numbered tasks in `/tasks` directory.
For each point of tasks, you should add test and stop waiting for approval.
ultrathink before you start.


## Setup Commands

**Environment Setup:**
```bash
conda env create -f env.yaml
conda activate scene_splat
```

**CUDA Extensions Build:**
```bash
# Build custom CUDA operations for point processing
pip install -e ./libs/pointops
pip install -e ./libs/pointgroup_ops
```

## Training Commands

**Vision-Language Pretraining:**
```bash
# Joint training on multiple datasets
python tools/train.py \
  --config-file configs/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py \
  --options save_path=exp_runs/lang_pretrainer/<experiment_name> \
  batch_size=4 batch_size_val=4 batch_size_test=4 num_worker=8 gpu_nums=1 \
  --num-gpus 1

# Resume from checkpoint
python tools/train.py \
  --config-file <config_file> \
  --options weight=model_best.pth resume=True \
  --num-gpus 1
```

**Self-Supervised Pretraining:**
```bash
python tools/ssl_pretrain.py \
  --config-file configs/concat_dataset/ssl-pretrain-concat-scan-ppv2-matt-3rscan-arkit-hyper-mcmc-base.py \
  --options save_path=exp_runs/ssl_pretrainer/<experiment_name> \
  --num-gpus 1
```

**Testing:**
```bash
# Test only mode
python tools/train.py \
  --config-file <config_file> \
  --options weight=model_best.pth test_only=True \
  --num-gpus 1

# Standalone testing
python tools/test.py \
  --config-file <config_file> \
  --options weight=model_best.pth \
  --num-gpus 1
```

**Multi-node Training:**
```bash
# For SLURM clusters with NCCL
srun python tools/train.py \
  --config-file <config_file> \
  --multi_node \
  --num-gpus <gpus_per_node>
```

## Code Architecture

**Core Framework**: Built on top of Pointcept, specializing in 3D Gaussian Splatting (3DGS) scene understanding.

**Main Components:**
- `pointcept/models/` - Model architectures including Point Transformer V3, sparse UNet, and SSL variants
- `pointcept/datasets/` - Datasets for ScanNet, ScanNet++, Matterport3D, HoliCity with 3DGS support
- `pointcept/engines/` - Training/testing engines with distributed training support
- `configs/` - Configuration files organized by dataset and training type
- `libs/` - Custom CUDA operations (pointops, pointgroup_ops) for efficient point processing

**Key Models:**
- Point Transformer V3 (`point_transformer_v3/`) - Primary backbone for semantic segmentation
- SimDINO SSL (`point_transformer_v3_ssl/`) - Self-supervised pretraining variant
- Context-aware classifiers - Open-vocabulary classification with vision-language features

**Data Format**: 3DGS scenes stored as separate `.npy` files:
- `coord.npy` - 3D coordinates
- `color.npy` - RGB values  
- `opacity.npy`, `quat.npy`, `scale.npy` - Gaussian parameters
- `lang_feat.npy` - Language features for vision-language training
- `segment.npy` - Semantic labels (optional)

**Training Types:**
1. **Vision-Language Pretraining**: Uses language features with contrastive learning
2. **Self-Supervised Pretraining**: SimDINO-based approach on Gaussian parameters only
3. **Semantic Segmentation**: Supervised training for scene understanding

**Configuration System**: Python-based configs with inheritance from `_base_/` directory. Configs specify model architecture, datasets, training parameters, and evaluation settings.

**Multi-Dataset Support**: Concurrent training on ScanNet, ScanNet++, Matterport3D with unified data loaders and evaluation protocols.

## Data Preprocessing

**Convert 3DGS to training format:**
```bash
python scripts/preprocess_gs.py --input_path <gs_scene.ply> --output_path <output_dir>
```

**Encode custom labels:**
```bash
python scripts/encode_labels.py --labels <class_names.txt> --output <text_embeddings.pt>
```

## Scene Visualization

**Visualize sample scene:**
```bash
# Point cloud visualization with Open3D (interactive 3D viewer)
python tools/visualize_scene.py sample/scene0708_00_0 --mode pointcloud --sample-ratio 0.1

# Matplotlib visualization (static plot, good for headless systems)
python tools/visualize_scene.py sample/scene0708_00_0 --backend matplotlib --sample-ratio 0.05

# Full resolution point cloud (may be slow)
python tools/visualize_scene.py sample/scene0708_00_0 --mode pointcloud

# Gaussian ellipsoids visualization (experimental)
python tools/visualize_scene.py sample/scene0708_00_0 --mode gaussians --sample-ratio 0.01
```

**Visualize inference features with PCA:**
```bash
# Load saved inference results and visualize with PCA colors
python tools/visualize_features_pca.py --results-dir results --scene-name scene0708_00_0

# Specify exact file paths
python tools/visualize_features_pca.py --features results/scene_*_features.npy --coords results/scene_*_coords.npy

# Different PCA components (1=grayscale, 2=RG, 3=RGB)
python tools/visualize_features_pca.py --results-dir results --scene-name scene0708_00_0 --n-components 3

# Save PCA colors for later use
python tools/visualize_features_pca.py --results-dir results --scene-name scene0708_00_0 --save-colors pca_colors.npy

# Use matplotlib backend with full resolution
python tools/visualize_features_pca.py --results-dir results --scene-name scene0708_00_0 --backend matplotlib --sample-ratio 1.0
```

**Text-to-Geometry Extraction:**
```bash
# Extract geometry based on text queries
python tools/text_to_geometry.py sample/scene0708_00_0 --queries "chair" "table" "sofa" --visualize --save-individual

# Extract with custom confidence threshold
python tools/text_to_geometry.py sample/scene0708_00_0 --queries "door" "window" --threshold 0.15

# Process large scenes with memory limits
python tools/text_to_geometry.py /path/to/large/scene --queries "bed" "desk" --max-points 30000 --save-individual

# Test text query workflow (simulation)
python tools/simple_text_test.py
```

**Text-to-Geometry Extraction:**
```bash
# Extract geometry based on text queries
python tools/text_to_geometry.py sample/scene0708_00_0 --queries "chair" "table" "sofa" --visualize --save-individual

# Extract with custom confidence threshold
python tools/text_to_geometry.py sample/scene0708_00_0 --queries "door" "window" --threshold 0.15

# Process large scenes with memory limits
python tools/text_to_geometry.py /path/to/large/scene --queries "bed" "desk" --max-points 30000 --save-individual

# Test text query workflow (simulation)
python tools/simple_text_test.py
```

## Configuration Guidelines

Before training, update dataset paths in configs:
```python
scannet_data_root = "/path/to/your/scannet_3dgs_mcmc_preprocessed"
scannetpp_data_root = "/path/to/your/scannetpp_v2_mcmc_3dgs"
matterport3d_data_root = "/path/to/your/matterport3d_region_mcmc_3dgs"
```

## GPU Requirements

- Vision-language pretraining: Minimum 48GB GPU memory
- Self-supervised pretraining: ~24GB GPU memory
- Inference: ~16GB GPU memory
- Multi-node training recommended for large-scale pretraining (3.5 days on 16x H100 for full training)