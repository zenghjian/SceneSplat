#!/usr/bin/env python3
"""
SceneSplat Scene Visualization Tool

Visualizes 3D Gaussian Splatting scenes from .npy data files.
Supports basic point cloud rendering and interactive 3D viewing.
"""

import numpy as np
import argparse
import os
from pathlib import Path

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


def load_scene_data(scene_path):
    """Load all scene data from directory containing .npy files."""
    scene_path = Path(scene_path)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_path}")
    
    data = {}
    required_files = ['coord.npy', 'color.npy']
    optional_files = ['opacity.npy', 'quat.npy', 'scale.npy', 'normal.npy', 
                     'pc_coord.npy', 'lang_feat.npy', 'valid_feat_mask.npy']
    
    # Load required files
    for file in required_files:
        filepath = scene_path / file
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")
        data[Path(file).stem] = np.load(filepath)
    
    # Load optional files
    for file in optional_files:
        filepath = scene_path / file
        if filepath.exists():
            data[Path(file).stem] = np.load(filepath)
        else:
            print(f"Optional file not found: {file}")
    
    return data


def create_point_cloud_open3d(data, use_normals=True, sample_ratio=1.0):
    """Create Open3D point cloud from scene data."""
    coords = data['coord']
    colors = data['color']
    
    # Sample points if requested
    if sample_ratio < 1.0:
        n_points = int(len(coords) * sample_ratio)
        indices = np.random.choice(len(coords), n_points, replace=False)
        coords = coords[indices]
        colors = colors[indices]
        if 'normal' in data and use_normals:
            normals = data['normal'][indices]
    else:
        if 'normal' in data and use_normals:
            normals = data['normal']
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors
    
    if 'normal' in data and use_normals:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return pcd


def create_gaussian_ellipsoids_open3d(data, max_gaussians=1000):
    """Create Open3D mesh geometries representing Gaussian ellipsoids."""
    if not all(key in data for key in ['coord', 'scale', 'quat', 'opacity']):
        print("Warning: Missing data for Gaussian visualization")
        return []
    
    coords = data['coord']
    scales = data['scale'] 
    quats = data['quat']
    opacities = data['opacity']
    colors = data['color'] / 255.0 if 'color' in data else None
    
    # Sample a subset for performance
    n_gaussians = min(max_gaussians, len(coords))
    indices = np.random.choice(len(coords), n_gaussians, replace=False)
    
    ellipsoids = []
    for i, idx in enumerate(indices):
        # Create unit sphere
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=8)
        
        # Scale by Gaussian scales
        scale_matrix = np.diag(scales[idx])
        
        # Create rotation matrix from quaternion
        q = quats[idx]  # [w, x, y, z] or [x, y, z, w] - check format
        # Assuming [x, y, z, w] format, convert to rotation matrix
        x, y, z, w = q
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])
        
        # Apply transformation
        transformation = np.eye(4)
        transformation[:3, :3] = R @ scale_matrix
        transformation[:3, 3] = coords[idx]
        
        sphere.transform(transformation)
        
        # Set color based on opacity and RGB
        if colors is not None:
            color = colors[idx] * opacities[idx]  # Modulate by opacity
        else:
            color = [opacities[idx]] * 3
        sphere.paint_uniform_color(color)
        
        ellipsoids.append(sphere)
    
    return ellipsoids


def visualize_with_open3d(data, mode='pointcloud', sample_ratio=1.0, show_normals=True):
    """Visualize scene using Open3D."""
    if not HAS_OPEN3D:
        raise ImportError("Open3D not available. Install with: pip install open3d")
    
    geometries = []
    
    if mode in ['pointcloud', 'both']:
        pcd = create_point_cloud_open3d(data, use_normals=show_normals, sample_ratio=sample_ratio)
        geometries.append(pcd)
        print(f"Point cloud: {len(pcd.points)} points")
    
    if mode in ['gaussians', 'both']:
        ellipsoids = create_gaussian_ellipsoids_open3d(data, max_gaussians=10000)
        geometries.extend(ellipsoids)
        print(f"Gaussians: {len(ellipsoids)} ellipsoids")
    
    # Set up visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SceneSplat Visualization", width=1280, height=720)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    print("Controls:")
    print("- Mouse: Rotate, zoom, pan")
    print("- 'H': Show help")
    print("- 'Q' or ESC: Quit")
    
    vis.run()
    vis.destroy_window()


def visualize_with_matplotlib(data, sample_ratio=0.1):
    """Simple matplotlib visualization for systems without Open3D."""
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib not available. Install with: pip install matplotlib")
    
    coords = data['coord']
    colors = data['color'] / 255.0
    
    # Sample for performance
    n_points = int(len(coords) * sample_ratio)
    indices = np.random.choice(len(coords), n_points, replace=False)
    coords = coords[indices]
    colors = colors[indices]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
               c=colors, s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title(f'SceneSplat Point Cloud ({n_points} points)')
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize SceneSplat 3D Gaussian scenes')
    parser.add_argument('scene_path', help='Path to scene directory containing .npy files')
    parser.add_argument('--mode', choices=['pointcloud', 'gaussians', 'both'], 
                       default='pointcloud', help='Visualization mode')
    parser.add_argument('--backend', choices=['open3d', 'matplotlib'], 
                       default='open3d', help='Visualization backend')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                       help='Ratio of points to sample (0-1)')
    parser.add_argument('--no-normals', action='store_true',
                       help='Disable normal visualization')
    
    args = parser.parse_args()
    
    # Load scene data
    print(f"Loading scene from: {args.scene_path}")
    data = load_scene_data(args.scene_path)
    
    # Print data summary
    print(f"\nScene data summary:")
    print(f"- Points: {len(data['coord']):,}")
    print(f"- Point cloud coords: {len(data.get('pc_coord', [])):,}")
    if 'lang_feat' in data:
        print(f"- Language features: {data['lang_feat'].shape}")
    
    # Visualize
    if args.backend == 'open3d':
        visualize_with_open3d(data, mode=args.mode, 
                             sample_ratio=args.sample_ratio,
                             show_normals=not args.no_normals)
    else:
        visualize_with_matplotlib(data, sample_ratio=args.sample_ratio)


if __name__ == '__main__':
    main()