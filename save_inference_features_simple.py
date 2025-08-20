#!/usr/bin/env python3
"""
SceneSplat推理结果简化保存脚本
只保存核心必要文件，减少文件数量
"""
import os
import numpy as np
import torch
import pickle
from datetime import datetime

def save_inference_output_simple(output, input_dict, scene_name, results_dir="/home/huajianzeng/project/SceneSplat/results"):
    """
    简化版推理输出保存 - 只保存核心文件
    
    Args:
        output: 模型输出
        input_dict: 输入数据字典
        scene_name: 场景名称
        results_dir: 保存目录
    
    Returns:
        dict: 保存的文件路径字典
    """
    print(f"\n💾 简化保存推理输出...")
    
    # 创建保存目录
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = {}
    
    try:
        # 1. 保存完整输出结构 (pickle格式，保持原始结构)
        output_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_full_output.pkl")
        output_for_save = {}
        
        if isinstance(output, dict):
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    output_for_save[k] = v.detach().cpu()
                elif hasattr(v, '__dict__'):
                    # 处理Point对象等自定义类
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
        print(f"   ✅ 完整输出: {os.path.basename(output_file)}")
        
        # 2. 保存主要输出特征 (最重要)
        if isinstance(output, dict) and 'point_feat' in output:
            point_feat = output['point_feat']
            
            if hasattr(point_feat, 'feat') and point_feat.feat is not None:
                # 主要特征 - 这是最重要的输出
                feat_array = point_feat.feat.detach().cpu().numpy()
                feat_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_features.npy")
                np.save(feat_file, feat_array)
                saved_files['features'] = feat_file
                print(f"   ✅ 主要特征: {os.path.basename(feat_file)} {feat_array.shape}")
            
            if hasattr(point_feat, 'coord') and point_feat.coord is not None:
                # 输出坐标 - 与特征对应的3D位置
                coord_array = point_feat.coord.detach().cpu().numpy()
                coord_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_coords.npy")
                np.save(coord_file, coord_array)
                saved_files['coords'] = coord_file
                print(f"   ✅ 输出坐标: {os.path.basename(coord_file)} {coord_array.shape}")
        
        elif isinstance(output, torch.Tensor):
            # 如果输出是单个tensor
            feat_array = output.detach().cpu().numpy()
            feat_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_features.npy")
            np.save(feat_file, feat_array)
            saved_files['features'] = feat_file
            print(f"   ✅ 输出特征: {os.path.basename(feat_file)} {feat_array.shape}")
        
        # 2. 保存输入数据用于对比 (可选但推荐)
        if input_dict:
            # 原始语言特征 - 用于对比分析
            if 'lang_feat' in input_dict:
                lang_array = input_dict['lang_feat'].detach().cpu().numpy()
                lang_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_input_lang.npy")
                np.save(lang_file, lang_array)
                saved_files['input_lang'] = lang_file
                print(f"   ✅ 输入语言特征: {os.path.basename(lang_file)} {lang_array.shape}")
            
            # 原始几何特征 - 用于理解输入
            if 'feat' in input_dict:
                geom_array = input_dict['feat'].detach().cpu().numpy()
                geom_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_input_geom.npy")
                np.save(geom_file, geom_array)
                saved_files['input_geom'] = geom_file
                print(f"   ✅ 输入几何特征: {os.path.basename(geom_file)} {geom_array.shape}")
        
        # 3. 生成快速加载脚本
        load_script = os.path.join(results_dir, f"load_{scene_name}_{timestamp}.py")
        with open(load_script, 'w') as f:
            f.write(f'''#!/usr/bin/env python3
"""
快速加载SceneSplat推理结果 - 简化版
生成时间: {datetime.now()}
场景: {scene_name}
"""
import numpy as np
import pickle
import os

def load_results():
    """加载核心推理结果"""
    results = {{}}
    
    print("🔄 加载推理结果...")
    
    # 加载完整输出
    full_output_file = "{output_file}"
    if os.path.exists(full_output_file):
        with open(full_output_file, 'rb') as f:
            results['full_output'] = pickle.load(f)
        print("✅ 完整输出已加载")
    
''')
            
            # 添加加载代码
            for key, file_path in saved_files.items():
                if file_path.endswith('.npy'):
                    f.write(f'''    # {key}
    {key}_file = "{file_path}"
    if os.path.exists({key}_file):
        results['{key}'] = np.load({key}_file)
        print(f"✅ {key}: {{results['{key}'].shape}}")
    else:
        print(f"❌ 文件不存在: {{{key}_file}}")
    
''')
            
            f.write('''    return results

def analyze_features():
    """分析主要特征"""
    data = load_results()
    
    if 'features' not in data:
        print("❌ 没有找到主要特征")
        return
    
    features = data['features']
    print(f"\\n📊 特征分析:")
    print(f"   形状: {features.shape}")
    print(f"   类型: {features.dtype}")
    print(f"   范围: [{features.min():.6f}, {features.max():.6f}]")
    print(f"   均值: {features.mean():.6f}")
    print(f"   标准差: {features.std():.6f}")
    
    # 与输入语言特征对比
    if 'input_lang' in data:
        input_lang = data['input_lang']
        print(f"\\n🔍 与输入语言特征对比:")
        print(f"   输入形状: {input_lang.shape}")
        print(f"   输入范围: [{input_lang.min():.6f}, {input_lang.max():.6f}]")
        
        # 计算相似度
        if features.shape == input_lang.shape:
            # 计算余弦相似度
            features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
            input_norm = input_lang / np.linalg.norm(input_lang, axis=1, keepdims=True)
            cosine_sim = np.mean(np.sum(features_norm * input_norm, axis=1))
            print(f"   余弦相似度: {cosine_sim:.6f}")
            
            if cosine_sim > 0.8:
                print("   🔍 高相似度 - 主要保持了输入语言特征")
            elif cosine_sim > 0.3:
                print("   🔍 中等相似度 - 语言与几何特征融合")
            else:
                print("   🔍 低相似度 - 大幅特征变换")
    
    return data

if __name__ == "__main__":
    print("SceneSplat推理结果加载器 - 简化版")
    print("=" * 50)
    
    # 加载并分析
    data = analyze_features()
    
    print(f"\\n✅ 完成! 可用数据: {list(data.keys()) if data else '无'}")
    
    print("\\n💡 使用提示:")
    print("   - 主要输出特征在 data['features'] 中")
    print("   - 对应坐标在 data['coords'] 中 (如果有)")
    print("   - 输入语言特征在 data['input_lang'] 中")
''')
        
        saved_files['load_script'] = load_script
        print(f"   ✅ 快速加载脚本: {os.path.basename(load_script)}")
        
        # 4. 生成简要摘要
        summary_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"SceneSplat推理结果 - 简化版\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"场景: {scene_name}\\n\\n")
            
            f.write("保存文件:\\n")
            for key, file_path in saved_files.items():
                f.write(f"- {key}: {os.path.basename(file_path)}\\n")
            
            f.write(f"\\n核心文件说明:\\n")
            if 'full_output' in saved_files:
                f.write(f"- full_output.pkl: 完整推理输出，保持原始结构\\n")
            if 'features' in saved_files:
                f.write(f"- features.npy: 主要输出特征，用于下游任务\\n")
            if 'coords' in saved_files:
                f.write(f"- coords.npy: 特征对应的3D坐标\\n")
            if 'input_lang' in saved_files:
                f.write(f"- input_lang.npy: 输入语言特征，用于对比\\n")
            if 'input_geom' in saved_files:
                f.write(f"- input_geom.npy: 输入几何特征\\n")
            
            f.write(f"\\n使用方法:\\n")
            f.write(f"python {os.path.basename(load_script)}\\n")
        
        saved_files['summary'] = summary_file
        print(f"   ✅ 摘要文件: {os.path.basename(summary_file)}")
        
        print(f"\\n📁 文件保存在: {results_dir}")
        npy_count = len([f for f in saved_files.values() if f.endswith('.npy')])
        pkl_count = len([f for f in saved_files.values() if f.endswith('.pkl')])
        print(f"🎯 核心文件: {npy_count} 个npy文件 + {pkl_count} 个pkl文件")
        print(f"📋 主要输出: {os.path.basename(saved_files.get('features', 'N/A'))}")
        print(f"📦 完整输出: {os.path.basename(saved_files.get('full_output', 'N/A'))}")
        
        return saved_files
        
    except Exception as e:
        print(f"   ❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        return {}

def load_simple_results(results_dir, scene_name=None, timestamp=None):
    """
    加载简化保存的结果
    
    Args:
        results_dir: 结果目录
        scene_name: 场景名称（可选）
        timestamp: 时间戳（可选）
    
    Returns:
        dict: 核心特征数据
    """
    if not os.path.exists(results_dir):
        print(f"❌ 结果目录不存在: {results_dir}")
        return {}
    
    files = os.listdir(results_dir)
    
    # 查找匹配文件
    if scene_name and timestamp:
        prefix = f"{scene_name}_{timestamp}_"
    elif scene_name:
        # 找到该场景最新的结果
        scene_files = [f for f in files if f.startswith(f"{scene_name}_") and f.endswith('.npy')]
        if not scene_files:
            print(f"❌ 没有找到场景 {scene_name} 的结果")
            return {}
        # 取最新的时间戳
        timestamps = list(set([f.split('_')[2] for f in scene_files if len(f.split('_')) >= 3]))
        timestamp = max(timestamps) if timestamps else None
        prefix = f"{scene_name}_{timestamp}_" if timestamp else f"{scene_name}_"
    else:
        print("❌ 请提供场景名称")
        return {}
    
    results = {}
    
    # 加载核心文件
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
                print(f"✅ {key}: {results[key].shape}")
            except Exception as e:
                print(f"❌ 加载失败 {filename}: {e}")
    
    return results

if __name__ == "__main__":
    print("SceneSplat简化特征保存工具")
    print("功能: 只保存核心必要文件，减少文件数量")
    print("使用: save_inference_output_simple(output, input_dict, scene_name)")