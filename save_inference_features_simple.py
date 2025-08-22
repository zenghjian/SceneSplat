#!/usr/bin/env python3
"""
SceneSplat inference results simplified save script
Save only core necessary files to reduce file count
"""
import os
import numpy as np
import torch
import pickle
from datetime import datetime

def save_inference_output_simple(output, input_dict, scene_name, results_dir="/home/huajianzeng/project/SceneSplat/results"):
    """
    Simplified inference output save - save only core files
    
    Args:
        output: Model output
        input_dict: Input data dictionary
        scene_name: Scene name
        results_dir: Save directory
    
    Returns:
        dict: Dictionary of saved file paths
    """
    print(f"\nSaving inference output...")
    
    # Create save directory
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = {}
    
    try:
        # 1. Save complete output structure (pickle format, maintain original structure)
        output_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_full_output.pkl")
        output_for_save = {}
        
        if isinstance(output, dict):
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    output_for_save[k] = v.detach().cpu()
                elif hasattr(v, '__dict__'):
                    # Handle custom classes like Point objects
                    obj_dict = {}
                    for attr_name, attr_value in v.__dict__.items():
                        if isinstance(attr_value, torch.Tensor):
                            obj_dict[attr_name] = attr_value.detach().cpu()
                        else:
                            obj_dict[attr_name] = attr_value
                    output_for_save[k] = obj_dict
                else:
                    output_for_save[k] = v
        else:
            if isinstance(output, torch.Tensor):
                output_for_save = output.detach().cpu()
            else:
                output_for_save = output
        
        with open(output_file, 'wb') as f:
            pickle.dump(output_for_save, f)
        
        saved_files['full_output'] = output_file
        print(f"   Full output: {os.path.basename(output_file)}")
        
        # 2. Save main output features (most important)
        if isinstance(output, dict) and 'point_feat' in output:
            point_feat = output['point_feat']
            
            if hasattr(point_feat, 'feat') and point_feat.feat is not None:
                # Main features - most important output
                feat_array = point_feat.feat.detach().cpu().numpy()
                feat_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_features.npy")
                np.save(feat_file, feat_array)
                saved_files['features'] = feat_file
                print(f"   Main features: {os.path.basename(feat_file)} {feat_array.shape}")
            
            if hasattr(point_feat, 'coord') and point_feat.coord is not None:
                # Output coordinates - 3D positions corresponding to features
                coord_array = point_feat.coord.detach().cpu().numpy()
                coord_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_coords.npy")
                np.save(coord_file, coord_array)
                saved_files['coords'] = coord_file
                print(f"   Output coordinates: {os.path.basename(coord_file)} {coord_array.shape}")
        
        elif isinstance(output, torch.Tensor):
            # If output is a single tensor
            feat_array = output.detach().cpu().numpy()
            feat_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_features.npy")
            np.save(feat_file, feat_array)
            saved_files['features'] = feat_file
            print(f"   Output features: {os.path.basename(feat_file)} {feat_array.shape}")
        
        # 3. Save input data for comparison (optional but recommended)
        if input_dict:
            # Original language features - for comparison analysis
            if 'lang_feat' in input_dict:
                lang_array = input_dict['lang_feat'].detach().cpu().numpy()
                lang_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_input_lang.npy")
                np.save(lang_file, lang_array)
                saved_files['input_lang'] = lang_file
                print(f"   Input language features: {os.path.basename(lang_file)} {lang_array.shape}")
            
            # Original geometric features - for understanding input
            if 'feat' in input_dict:
                geom_array = input_dict['feat'].detach().cpu().numpy()
                geom_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_input_geom.npy")
                np.save(geom_file, geom_array)
                saved_files['input_geom'] = geom_file
                print(f"   Input geometric features: {os.path.basename(geom_file)} {geom_array.shape}")
        
        # 4. Generate quick load script
        load_script = os.path.join(results_dir, f"load_{scene_name}_{timestamp}.py")
        with open(load_script, 'w') as f:
            f.write(f'''#!/usr/bin/env python3
"""
Quick load SceneSplat inference results - Simplified version
Generated: {datetime.now()}
Scene: {scene_name}
"""
import numpy as np
import pickle
import os

def load_results():
    """Load core inference results"""
    results = {{}}
    
    print("Loading inference results...")
    
    # Load complete output
    full_output_file = "{output_file}"
    if os.path.exists(full_output_file):
        with open(full_output_file, 'rb') as f:
            results['full_output'] = pickle.load(f)
        print("Complete output loaded")
    
''')
            
            # Add loading code
            for key, file_path in saved_files.items():
                if file_path.endswith('.npy'):
                    f.write(f'''    # {key}
    {key}_file = "{file_path}"
    if os.path.exists({key}_file):
        results['{key}'] = np.load({key}_file)
        print(f"{key}: {{results['{key}'].shape}}")
    else:
        print(f"File not found: {{{key}_file}}")
    
''')
            
            f.write('''    return results

def analyze_features():
    """Analyze main features"""
    data = load_results()
    
    if 'features' not in data:
        print("Main features not found")
        return
    
    features = data['features']
    print(f"\\nFeature analysis:")
    print(f"   Shape: {features.shape}")
    print(f"   Type: {features.dtype}")
    print(f"   Range: [{features.min():.6f}, {features.max():.6f}]")
    print(f"   Mean: {features.mean():.6f}")
    print(f"   Std: {features.std():.6f}")
    
    # Compare with input language features
    if 'input_lang' in data:
        input_lang = data['input_lang']
        print(f"\\nComparison with input language features:")
        print(f"   Input shape: {input_lang.shape}")
        print(f"   Input range: [{input_lang.min():.6f}, {input_lang.max():.6f}]")
        
        # Calculate similarity
        if features.shape == input_lang.shape:
            # Calculate cosine similarity
            features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
            input_norm = input_lang / np.linalg.norm(input_lang, axis=1, keepdims=True)
            cosine_sim = np.mean(np.sum(features_norm * input_norm, axis=1))
            print(f"   Cosine similarity: {cosine_sim:.6f}")
            
            if cosine_sim > 0.8:
                print("   High similarity - mainly preserves input language features")
            elif cosine_sim > 0.3:
                print("   Medium similarity - fusion of language and geometric features")
            else:
                print("   Low similarity - significant feature transformation")
    
    return data

if __name__ == "__main__":
    print("SceneSplat Inference Results Loader - Simplified Version")
    print("=" * 50)
    
    # Load and analyze
    data = analyze_features()
    
    print(f"\\nComplete! Available data: {list(data.keys()) if data else 'None'}")
    
    print("\\nUsage tips:")
    print("   - Main output features in data['features']")
    print("   - Corresponding coordinates in data['coords'] (if available)")
    print("   - Input language features in data['input_lang']")
''')
        
        saved_files['load_script'] = load_script
        print(f"   Quick load script: {os.path.basename(load_script)}")
        
        # 5. Generate summary
        summary_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"SceneSplat Inference Results - Simplified Version\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Scene: {scene_name}\\n\\n")
            
            f.write("Saved files:\\n")
            for key, file_path in saved_files.items():
                f.write(f"- {key}: {os.path.basename(file_path)}\\n")
            
            f.write(f"\\nCore file descriptions:\\n")
            if 'full_output' in saved_files:
                f.write(f"- full_output.pkl: Complete inference output, maintains original structure\\n")
            if 'features' in saved_files:
                f.write(f"- features.npy: Main output features for downstream tasks\\n")
            if 'coords' in saved_files:
                f.write(f"- coords.npy: 3D coordinates corresponding to features\\n")
            if 'input_lang' in saved_files:
                f.write(f"- input_lang.npy: Input language features for comparison\\n")
            if 'input_geom' in saved_files:
                f.write(f"- input_geom.npy: Input geometric features\\n")
            
            f.write(f"\\nUsage:\\n")
            f.write(f"python {os.path.basename(load_script)}\\n")
        
        saved_files['summary'] = summary_file
        print(f"   Summary file: {os.path.basename(summary_file)}")
        
        print(f"\nFiles saved in: {results_dir}")
        npy_count = len([f for f in saved_files.values() if f.endswith('.npy')])
        pkl_count = len([f for f in saved_files.values() if f.endswith('.pkl')])
        print(f"Core files: {npy_count} npy files + {pkl_count} pkl files")
        print(f"Main output: {os.path.basename(saved_files.get('features', 'N/A'))}")
        print(f"Full output: {os.path.basename(saved_files.get('full_output', 'N/A'))}")
        
        return saved_files
        
    except Exception as e:
        print(f"   Save failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def load_simple_results(results_dir, scene_name=None, timestamp=None):
    """
    Load simplified saved results
    
    Args:
        results_dir: Results directory
        scene_name: Scene name (optional)
        timestamp: Timestamp (optional)
    
    Returns:
        dict: Core feature data
    """
    if not os.path.exists(results_dir):
        print(f"Results directory does not exist: {results_dir}")
        return {}
    
    files = os.listdir(results_dir)
    
    # Find matching files
    if scene_name and timestamp:
        prefix = f"{scene_name}_{timestamp}_"
    elif scene_name:
        # Find the latest results for this scene
        scene_files = [f for f in files if f.startswith(f"{scene_name}_") and f.endswith('.npy')]
        if not scene_files:
            print(f"No results found for scene {scene_name}")
            return {}
        # Get the latest timestamp
        timestamps = list(set([f.split('_')[2] for f in scene_files if len(f.split('_')) >= 3]))
        timestamp = max(timestamps) if timestamps else None
        prefix = f"{scene_name}_{timestamp}_" if timestamp else f"{scene_name}_"
    else:
        print("Please provide scene name")
        return {}
    
    results = {}
    
    # Load core files
    core_files = {
        'features': f"{prefix}features.npy",
        'coords': f"{prefix}coords.npy", 
        'input_lang': f"{prefix}input_lang.npy",
        'input_geom': f"{prefix}input_geom.npy"
    }
    
    for key, filename in core_files.items():
        file_path = os.path.join(results_dir, filename)
        if os.path.exists(file_path):
            try:
                results[key] = np.load(file_path)
                print(f"{key}: {results[key].shape}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
    
    return results

if __name__ == "__main__":
    print("SceneSplat Simplified Feature Save Tool")
    print("Function: Save only core necessary files to reduce file count")
    print("Usage: save_inference_output_simple(output, input_dict, scene_name)")