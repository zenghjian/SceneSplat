#!/usr/bin/env python3
"""
SceneSplat Feature Visualization with PCA

Visualizes high-dimensional features (768D) from SceneSplat inference
by reducing them to RGB colors using PCA dimensionality reduction.
"""

import numpy as np
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath('.'))

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_inference_results(features_path=None, coords_path=None, results_dir=None, scene_name=None):
    """
    Load inference results from saved files.
    
    Args:
        features_path: Direct path to features.npy
        coords_path: Direct path to coords.npy
        results_dir: Directory containing results
        scene_name: Scene name to search for
    
    Returns:
        dict with 'features' and 'coords' arrays
    """
    data = {}
    
    if features_path and coords_path:
        # Direct file paths provided
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        if not os.path.exists(coords_path):
            raise FileNotFoundError(f"Coordinates file not found: {coords_path}")
        
        data['features'] = np.load(features_path)
        data['coords'] = np.load(coords_path)
        print(f"Loaded features: {data['features'].shape}")
        print(f"Loaded coordinates: {data['coords'].shape}")
        
    elif results_dir and scene_name:
        # Search in results directory
        results_path = Path(results_dir)
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        # Find the latest files for this scene
        feature_files = list(results_path.glob(f"{scene_name}_*_features.npy"))
        coord_files = list(results_path.glob(f"{scene_name}_*_coords.npy"))
        
        if not feature_files:
            raise FileNotFoundError(f"No feature files found for scene: {scene_name}")
        if not coord_files:
            raise FileNotFoundError(f"No coordinate files found for scene: {scene_name}")
        
        # Use the most recent files
        features_path = max(feature_files, key=os.path.getmtime)
        coords_path = max(coord_files, key=os.path.getmtime)
        
        data['features'] = np.load(features_path)
        data['coords'] = np.load(coords_path)
        print(f"Loaded features from: {features_path.name}")
        print(f"Loaded coordinates from: {coords_path.name}")
        print(f"Features shape: {data['features'].shape}")
        print(f"Coordinates shape: {data['coords'].shape}")
    
    else:
        raise ValueError("Provide either (features_path, coords_path) or (results_dir, scene_name)")
    
    return data


def apply_pca_to_features(features, n_components=3, random_state=42):
    """
    Apply PCA to reduce high-dimensional features to RGB colors.
    
    Args:
        features: (N, D) array of high-dimensional features
        n_components: Number of PCA components (1-3)
        random_state: Random seed for reproducibility
    
    Returns:
        dict with 'colors', 'pca_features', 'pca_model', and statistics
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for PCA. Install with: pip install scikit-learn")
    
    print(f"\nApplying PCA: {features.shape[1]}D -> {n_components}D")
    
    # Fit PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_features = pca.fit_transform(features)
    
    # Print statistics
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Normalize to [0, 1] range
    pca_min = pca_features.min(axis=0)
    pca_max = pca_features.max(axis=0)
    pca_range = pca_max - pca_min
    
    # Avoid division by zero
    pca_range[pca_range == 0] = 1.0
    
    pca_normalized = (pca_features - pca_min) / pca_range
    
    # Create RGB colors
    if n_components == 1:
        # Grayscale
        colors = np.repeat(pca_normalized, 3, axis=1)
    elif n_components == 2:
        # RG with blue=0.5
        colors = np.column_stack([pca_normalized, np.full(len(pca_normalized), 0.5)])
    else:  # n_components == 3
        colors = pca_normalized
    
    # Ensure colors are in valid range
    colors = np.clip(colors, 0, 1)
    
    return {
        'colors': colors,
        'pca_features': pca_features,
        'pca_model': pca,
        'explained_variance': pca.explained_variance_ratio_,
        'n_components': n_components
    }


def visualize_with_open3d(coords, colors, point_size=1.0, show_axes=True):
    """
    Visualize point cloud with PCA colors using Open3D.
    
    Args:
        coords: (N, 3) array of 3D coordinates
        colors: (N, 3) array of RGB colors in [0, 1]
        point_size: Size of points in visualization
        show_axes: Show coordinate axes
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D not available. Install with: pip install open3d")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"\nVisualizing {len(coords)} points with PCA colors")
    
    # Setup visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SceneSplat PCA Feature Visualization", width=1280, height=720)
    vis.add_geometry(pcd)
    
    if show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(axes)
    
    # Set render options
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    print("\nControls:")
    print("  Mouse: Rotate, zoom, pan")
    print("  '+'/'-': Increase/decrease point size")
    print("  'H': Show help")
    print("  'Q' or ESC: Quit")
    
    vis.run()
    vis.destroy_window()


def visualize_with_matplotlib(coords, colors, sample_ratio=0.1, figsize=(15, 5)):
    """
    Visualize point cloud with PCA colors using Matplotlib.
    
    Args:
        coords: (N, 3) array of 3D coordinates
        colors: (N, 3) array of RGB colors in [0, 1]
        sample_ratio: Ratio of points to display
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib not available. Install with: pip install matplotlib")
    
    # Sample points for performance
    n_points = len(coords)
    n_sample = int(n_points * sample_ratio)
    if n_sample < n_points:
        indices = np.random.choice(n_points, n_sample, replace=False)
        coords = coords[indices]
        colors = colors[indices]
        print(f"\nSampled {n_sample}/{n_points} points for visualization")
    
    fig = plt.figure(figsize=figsize)
    
    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                c=colors, s=1, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'PCA-Colored Point Cloud ({n_sample} points)')
    
    # Top view (XY)
    ax2 = fig.add_subplot(132)
    ax2.scatter(coords[:, 0], coords[:, 1], 
                c=colors, s=1, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top View (XY)')
    ax2.axis('equal')
    
    # Side view (XZ)
    ax3 = fig.add_subplot(133)
    ax3.scatter(coords[:, 0], coords[:, 2], 
                c=colors, s=1, alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Side View (XZ)')
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.show()


def run_inference_and_visualize(scene_path, max_points=50000):
    """
    Run inference on a scene and visualize with PCA.
    
    Args:
        scene_path: Path to scene directory
        max_points: Maximum points for inference
    
    Returns:
        dict with features, coords, and PCA results
    """
    # Import inference function
    from test_single_scene_lang import test_single_scene, CONFIG
    
    # Configure for the specified number of points
    CONFIG['max_points'] = max_points
    CONFIG['enable_subsampling'] = True
    
    print(f"Running inference on scene: {scene_path}")
    print(f"Max points: {max_points}")
    
    # Temporarily modify the scene path in the test function
    import test_single_scene_lang
    original_path = "/home/huajianzeng/project/SceneSplat/sample/scene0708_00_0"
    test_single_scene_lang.test_single_scene.__code__.co_consts = tuple(
        scene_path if c == original_path else c 
        for c in test_single_scene_lang.test_single_scene.__code__.co_consts
    )
    
    # Run inference
    success, output = test_single_scene()
    
    if not success:
        raise RuntimeError("Inference failed")
    
    # Extract features and coordinates
    if 'point_feat' in output:
        point_feat = output['point_feat']
        features = point_feat.feat.detach().cpu().numpy()
        coords = point_feat.coord.detach().cpu().numpy()
    else:
        raise ValueError("No point features in output")
    
    return {
        'features': features,
        'coords': coords,
        'output': output
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize SceneSplat features using PCA')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--features', help='Path to features.npy file')
    input_group.add_argument('--scene', help='Path to scene directory for direct inference')
    input_group.add_argument('--results-dir', help='Results directory to search')
    
    # Additional inputs
    parser.add_argument('--coords', help='Path to coords.npy file (required with --features)')
    parser.add_argument('--scene-name', help='Scene name to search for (required with --results-dir)')
    
    # PCA options
    parser.add_argument('--n-components', type=int, default=3, choices=[1, 2, 3],
                        help='Number of PCA components (default: 3)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for PCA (default: 42)')
    
    # Visualization options
    parser.add_argument('--backend', choices=['open3d', 'matplotlib'], default='open3d',
                        help='Visualization backend (default: open3d)')
    parser.add_argument('--sample-ratio', type=float, default=0.1,
                        help='Sample ratio for matplotlib (default: 0.1)')
    parser.add_argument('--point-size', type=float, default=2.0,
                        help='Point size for visualization (default: 2.0)')
    parser.add_argument('--no-axes', action='store_true',
                        help='Hide coordinate axes')
    
    # Inference options (when using --scene)
    parser.add_argument('--max-points', type=int, default=50000,
                        help='Maximum points for inference (default: 50000)')
    
    # Output options
    parser.add_argument('--save-colors', help='Save PCA colors to file')
    
    args = parser.parse_args()
    
    # Load or generate data
    if args.scene:
        # Run inference and get features
        print("Running inference and visualization pipeline...")
        data = run_inference_and_visualize(args.scene, args.max_points)
    elif args.features:
        # Load from files
        if not args.coords:
            parser.error("--coords required when using --features")
        data = load_inference_results(args.features, args.coords)
    else:
        # Search in results directory
        if not args.scene_name:
            parser.error("--scene-name required when using --results-dir")
        data = load_inference_results(results_dir=args.results_dir, scene_name=args.scene_name)
    
    # Apply PCA
    pca_result = apply_pca_to_features(
        data['features'], 
        n_components=args.n_components,
        random_state=args.random_state
    )
    
    # Save colors if requested
    if args.save_colors:
        np.save(args.save_colors, pca_result['colors'])
        print(f"Saved PCA colors to: {args.save_colors}")
    
    # Visualize
    if args.backend == 'open3d':
        visualize_with_open3d(
            data['coords'], 
            pca_result['colors'],
            point_size=args.point_size,
            show_axes=not args.no_axes
        )
    else:
        visualize_with_matplotlib(
            data['coords'],
            pca_result['colors'],
            sample_ratio=args.sample_ratio
        )


if __name__ == '__main__':
    main()