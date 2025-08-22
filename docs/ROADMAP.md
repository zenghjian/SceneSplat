# ROADMAP

## ✅ Task 1: Environment Setup (Completed)
- [x] Create conda environment from env.yaml
- [x] Activate scene_splat environment  
- [x] Install CUDA extensions (pointops, pointgroup_ops)
- [x] Verify environment setup

## ✅ Task 2: Model Testing & Inference (Completed)
- [x] Set up inference environment
- [x] Configure model paths and data paths  
- [x] Run test inference with pretrained model
- [x] Analyze inference results
- [x] Validate output quality
- [x] Save inference output features

## ✅ Task 3: Scene Visualization (Completed)
- [x] Explore sample scene structure and data format
- [x] Implement basic 3D Gaussian scene visualization
- [x] Test visualization with sample scene data
- [x] Add interactive viewing capabilities
- [x] Document visualization usage

## ✅ Task 4: Feature Visualization with PCA (Completed)
- [x] Design PCA-based feature visualization pipeline
- [x] Create unified script combining inference and visualization
- [x] Implement PCA dimensionality reduction (768D → 3D RGB)
- [x] Visualize high-dimensional features as colors on 3D points
- [x] Add support for loading saved inference results
- [x] Test with different PCA components and color mappings

## ✅ Task 5: Text-based Scene Query (Completed)
- [x] Design text-to-3D query system architecture
- [x] Implement text encoding pipeline (SigLIP-based for 768-dim features)
- [x] Develop feature similarity computation for 3D point segmentation
- [x] Create segmentation mask generation from similarity scores
- [x] Implement scene visualization with highlighted query results
- [x] Implement isolated object geometry extraction and visualization
- [x] Test and validate with sample queries ("chair" tested successfully)
- [x] Add support for multiple query methods (threshold, top-k, adaptive)