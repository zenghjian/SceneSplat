#!/usr/bin/env python3
"""
Single scene language model inference test
"""
import sys
import os
import torch
import numpy as np
sys.path.insert(0, os.path.abspath('.'))

from pointcept.utils.config import Config
from pointcept.models import build_model
from datetime import datetime
import pickle

# Configuration for point sampling
CONFIG = {
    'max_points': 100000,  # Maximum points before subsampling
    'subsample_method': 'random',  # 'random' or 'fps' (furthest point sampling)
    'random_seed': 42,  # For reproducible random sampling
    'enable_subsampling': True,  # Enable/disable subsampling
}

def test_single_scene():
    print("SceneSplat Single Scene Language Model Inference Test")
    print("=" * 50)
    
    # Scene path
    scene_path = "/home/huajianzeng/project/SceneSplat/sample/scene0708_00_0"
    
    print("1. Loading scene data...")
    try:
        coord = np.load(os.path.join(scene_path, 'coord.npy'))
        color = np.load(os.path.join(scene_path, 'color.npy'))
        opacity = np.load(os.path.join(scene_path, 'opacity.npy'))
        quat = np.load(os.path.join(scene_path, 'quat.npy'))
        scale = np.load(os.path.join(scene_path, 'scale.npy'))
        lang_feat = np.load(os.path.join(scene_path, 'lang_feat.npy'))
        
        print(f"   Coordinate shape: {coord.shape}")
        print(f"   Color shape: {color.shape}")
        print(f"   Opacity shape: {opacity.shape}")
        print(f"   Quaternion shape: {quat.shape}")
        print(f"   Scale shape: {scale.shape}")
        print(f"   Language feature shape: {lang_feat.shape}")
        
        # Combined features (color:3 + quat:4 + scale:3 + opacity:1 = 11 dims)
        feat = np.concatenate([color, quat, scale, opacity[:, None]], axis=1)
        print(f"   Combined feature shape: {feat.shape}")
        
        # Subsampling based on configuration
        total_points = len(coord)
        
        if CONFIG['enable_subsampling'] and total_points > CONFIG['max_points']:
            print(f"   Scene has {total_points} points, exceeds max_points ({CONFIG['max_points']})")
            print(f"   Applying {CONFIG['subsample_method']} subsampling...")
            
            n_points = CONFIG['max_points']
            
            if CONFIG['subsample_method'] == 'random':
                # Random subsampling
                np.random.seed(CONFIG['random_seed'])
                sample_indices = np.random.choice(total_points, size=n_points, replace=False)
                sample_indices = np.sort(sample_indices)  # Sort to maintain some spatial locality
            elif CONFIG['subsample_method'] == 'fps':
                # Furthest point sampling (simplified version)
                # For now, use random as FPS implementation would be more complex
                print("   Note: FPS not implemented, using random instead")
                np.random.seed(CONFIG['random_seed'])
                sample_indices = np.random.choice(total_points, size=n_points, replace=False)
                sample_indices = np.sort(sample_indices)
            else:
                raise ValueError(f"Unknown subsample_method: {CONFIG['subsample_method']}")
            
            # Apply subsampling
            coord = coord[sample_indices]
            color = color[sample_indices]
            opacity = opacity[sample_indices]
            quat = quat[sample_indices]
            scale = scale[sample_indices]
            lang_feat = lang_feat[sample_indices]
            
            print(f"   >> Subsampled to {n_points} points ({n_points/total_points*100:.1f}% of original)")
        else:
            n_points = total_points
            if CONFIG['enable_subsampling']:
                print(f"   Scene has {total_points} points, within max_points ({CONFIG['max_points']})")
            print(f"   >> Using all {n_points} points")
        
        # Recreate combined features after potential subsampling
        feat = np.concatenate([color, quat, scale, opacity[:, None]], axis=1)
        
    except Exception as e:
        print(f"   Data loading failed: {e}")
        return False
    
    print("\n2. Loading model config...")
    try:
        cfg = Config.fromfile('/home/huajianzeng/project/SceneSplat/configs/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py')
        print("   Config loaded successfully")
    except Exception as e:
        print(f"   Config loading failed: {e}")
        return False
    
    print("\n3. Building model...")
    try:
        model = build_model(cfg.model)
        model = model.cuda()
        print(f"   Model built successfully: {type(model)}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   Model building failed: {e}")
        return False
    
    print("\n4. Loading pretrained weights...")
    try:
        checkpoint = torch.load('/home/huajianzeng/project/SceneSplat/models/pretrained/scenesplat_main/model_best.pth', weights_only=False)
        
        # Clean weight names
        from collections import OrderedDict
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if key.startswith("module."):
                key = key[7:]  # Remove module. prefix
            weight[key] = value
        
        model.load_state_dict(weight, strict=True)
        print(f"   Weights loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    except Exception as e:
        print(f"   Weight loading failed: {e}")
        return False
    
    print("\n5. Preparing input data...")
    try:
        # Convert to tensor (no batch dimension needed)
        coord = torch.from_numpy(coord).float().cuda()  # [N, 3]
        feat = torch.from_numpy(feat).float().cuda()    # [N, 11]
        lang_feat = torch.from_numpy(lang_feat).float().cuda()  # [N, D]
        
        print(f"   Input coordinates: {coord.shape}")
        print(f"   Input features: {feat.shape}")
        print(f"   Language features: {lang_feat.shape}")
        
    except Exception as e:
        print(f"   Data preparation failed: {e}")
        return False
    
    print("\n6. Running forward inference...")
    try:
        model.eval()
        with torch.no_grad():
            # Build input dictionary
            input_dict = {
                'coord': coord,
                'feat': feat,
                'lang_feat': lang_feat,
                'offset': torch.tensor([0, coord.shape[0]], device='cuda'),  # [0, N]
                'grid_size': 0.02,  # Add grid size
            }
            
            print("   Starting inference...")
            output = model(input_dict)
            print(f"   Inference completed successfully!")
            
            # Save inference results (simplified)
            from save_inference_features_simple import save_inference_output_simple
            scene_name = os.path.basename(scene_path)
            saved_files = save_inference_output_simple(output, input_dict, scene_name)
            
            return True, output
            
    except Exception as e:
        print(f"   Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test single scene language model inference')
    parser.add_argument('--max-points', type=int, default=CONFIG['max_points'],
                        help=f"Maximum points before subsampling (default: {CONFIG['max_points']})")
    parser.add_argument('--no-subsample', action='store_true',
                        help='Disable subsampling, use all points')
    parser.add_argument('--method', type=str, default=CONFIG['subsample_method'],
                        choices=['random', 'fps'],
                        help=f"Subsampling method (default: {CONFIG['subsample_method']})")
    parser.add_argument('--seed', type=int, default=CONFIG['random_seed'],
                        help=f"Random seed for sampling (default: {CONFIG['random_seed']})")
    
    args = parser.parse_args()
    
    # Update config from command line arguments
    CONFIG['max_points'] = args.max_points
    CONFIG['enable_subsampling'] = not args.no_subsample
    CONFIG['subsample_method'] = args.method
    CONFIG['random_seed'] = args.seed
    
    print(f"Configuration:")
    print(f"  - Max points: {CONFIG['max_points']}")
    print(f"  - Subsampling: {'Enabled' if CONFIG['enable_subsampling'] else 'Disabled'}")
    print(f"  - Method: {CONFIG['subsample_method']}")
    print(f"  - Random seed: {CONFIG['random_seed']}")
    print()
    
    result = test_single_scene()
    if isinstance(result, tuple):
        success, output = result
        if success:
            print("\nSingle scene language model inference test completed successfully!")
        else:
            print("\nInference test failed")
    else:
        success = result
        if success:
            print("\nSingle scene language model inference test completed successfully!")
        else:
            print("\nInference test failed")
    
    sys.exit(0 if success else 1)