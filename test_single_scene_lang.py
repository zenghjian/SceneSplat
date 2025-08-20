#!/usr/bin/env python3
"""
单场景语言模型推理测试
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

def test_single_scene():
    print("SceneSplat 单场景语言模型推理测试")
    print("=" * 50)
    
    # 场景路径
    scene_path = "/home/huajianzeng/project/SceneSplat/sample/scene0708_00_0"
    
    print("1. 加载场景数据...")
    try:
        coord = np.load(os.path.join(scene_path, 'coord.npy'))
        color = np.load(os.path.join(scene_path, 'color.npy'))
        opacity = np.load(os.path.join(scene_path, 'opacity.npy'))
        quat = np.load(os.path.join(scene_path, 'quat.npy'))
        scale = np.load(os.path.join(scene_path, 'scale.npy'))
        lang_feat = np.load(os.path.join(scene_path, 'lang_feat.npy'))
        
        print(f"   坐标形状: {coord.shape}")
        print(f"   颜色形状: {color.shape}")
        print(f"   透明度形状: {opacity.shape}")
        print(f"   四元数形状: {quat.shape}")
        print(f"   缩放形状: {scale.shape}")
        print(f"   语言特征形状: {lang_feat.shape}")
        
        # 组合特征 (color:3 + quat:4 + scale:3 + opacity:1 = 11维)
        feat = np.concatenate([color, quat, scale, opacity[:, None]], axis=1)
        print(f"   组合特征形状: {feat.shape}")
        
        # 由于GPU内存限制，只使用前10000个点进行测试
        n_points = min(10000, len(coord))
        coord = coord[:n_points]
        color = color[:n_points]
        opacity = opacity[:n_points]
        quat = quat[:n_points]
        scale = scale[:n_points]
        lang_feat = lang_feat[:n_points]
        feat = np.concatenate([color, quat, scale, opacity[:, None]], axis=1)
        
        print(f"   >> 使用子集进行测试: {n_points} 个点")
        
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return False
    
    print("\n2. 加载模型配置...")
    try:
        cfg = Config.fromfile('/home/huajianzeng/project/SceneSplat/configs/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py')
        print("   ✅ 配置加载成功")
    except Exception as e:
        print(f"   ❌ 配置加载失败: {e}")
        return False
    
    print("\n3. 构建模型...")
    try:
        model = build_model(cfg.model)
        model = model.cuda()
        print(f"   ✅ 模型构建成功: {type(model)}")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ❌ 模型构建失败: {e}")
        return False
    
    print("\n4. 加载预训练权重...")
    try:
        checkpoint = torch.load('/home/huajianzeng/project/SceneSplat/models/pretrained/scenesplat_main/model_best.pth', weights_only=False)
        
        # 清理权重名称
        from collections import OrderedDict
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if key.startswith("module."):
                key = key[7:]  # 去除 module. 前缀
            weight[key] = value
        
        model.load_state_dict(weight, strict=True)
        print(f"   ✅ 权重加载成功 (epoch {checkpoint.get('epoch', 'unknown')})")
    except Exception as e:
        print(f"   ❌ 权重加载失败: {e}")
        return False
    
    print("\n5. 准备输入数据...")
    try:
        # 转换为tensor (不需要batch维度，直接传入)
        coord = torch.from_numpy(coord).float().cuda()  # [N, 3]
        feat = torch.from_numpy(feat).float().cuda()    # [N, 11]
        lang_feat = torch.from_numpy(lang_feat).float().cuda()  # [N, D]
        
        print(f"   输入坐标: {coord.shape}")
        print(f"   输入特征: {feat.shape}")
        print(f"   语言特征: {lang_feat.shape}")
        
    except Exception as e:
        print(f"   ❌ 数据准备失败: {e}")
        return False
    
    print("\n6. 运行前向推理...")
    try:
        model.eval()
        with torch.no_grad():
            # 构建输入字典
            input_dict = {
                'coord': coord,
                'feat': feat,
                'lang_feat': lang_feat,
                'offset': torch.tensor([0, coord.shape[0]], device='cuda'),  # [0, N]
                'grid_size': 0.02,  # 添加网格大小
            }
            
            print("   开始推理...")
            output = model(input_dict)
            print(f"   ✅ 推理成功!")
            
            # 保存推理结果 (简化版)
            from save_inference_features_simple import save_inference_output_simple
            scene_name = os.path.basename(scene_path)
            saved_files = save_inference_output_simple(output, input_dict, scene_name)
            
            return True, output
            
    except Exception as e:
        print(f"   ❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def save_results(output, input_dict, scene_path):
    """保存推理结果到results目录"""
    print("\n7. 保存推理结果...")
    
    # 创建results目录
    results_dir = "/home/huajianzeng/project/SceneSplat/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_name = os.path.basename(scene_path)
    
    saved_files = {}
    
    try:
        # 1. 保存所有输出数据
        print("   正在分析输出结构...")
        
        if isinstance(output, dict):
            print(f"   发现输出字典，键: {list(output.keys())}")
            
            # 保存完整输出字典 (pickle格式，保持原始结构)
            output_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_full_output.pkl")
            with open(output_file, 'wb') as f:
                # 将所有tensor移到CPU再保存
                output_cpu = {}
                for k, v in output.items():
                    if isinstance(v, torch.Tensor):
                        output_cpu[k] = v.detach().cpu()
                    elif hasattr(v, 'feat') and hasattr(v, 'coord'):
                        # Point对象
                        point_cpu = type(v)()
                        for attr_name in dir(v):
                            if not attr_name.startswith('_') and hasattr(v, attr_name):
                                attr_value = getattr(v, attr_name)
                                if isinstance(attr_value, torch.Tensor):
                                    setattr(point_cpu, attr_name, attr_value.detach().cpu())
                                elif not callable(attr_value):
                                    setattr(point_cpu, attr_name, attr_value)
                        output_cpu[k] = point_cpu
                    else:
                        output_cpu[k] = v
                pickle.dump(output_cpu, f)
            
            saved_files['full_output'] = output_file
            print(f"   ✅ 完整输出已保存到: {output_file}")
            
            # 2. 提取并保存主要特征
            for key, value in output.items():
                print(f"   处理输出项: {key}")
                
                if isinstance(value, torch.Tensor):
                    # 直接的tensor输出
                    feat_array = value.detach().cpu().numpy()
                    feat_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_{key}.npy")
                    np.save(feat_file, feat_array)
                    saved_files[key] = feat_file
                    print(f"     ✅ {key} 已保存: {feat_file} (形状: {feat_array.shape})")
                
                elif hasattr(value, 'feat') and value.feat is not None:
                    # Point对象的feat
                    feat_array = value.feat.detach().cpu().numpy()
                    feat_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_{key}_feat.npy")
                    np.save(feat_file, feat_array)
                    saved_files[f'{key}_feat'] = feat_file
                    print(f"     ✅ {key}.feat 已保存: {feat_file} (形状: {feat_array.shape})")
                    
                    # Point对象的coord
                    if hasattr(value, 'coord') and value.coord is not None:
                        coord_array = value.coord.detach().cpu().numpy()
                        coord_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_{key}_coord.npy")
                        np.save(coord_array, coord_file)
                        saved_files[f'{key}_coord'] = coord_file
                        print(f"     ✅ {key}.coord 已保存: {coord_file} (形状: {coord_array.shape})")
                    
                    # Point对象的其他属性
                    for attr_name in ['grid_coord', 'serialized_code', 'index', 'offset']:
                        if hasattr(value, attr_name):
                            attr_value = getattr(value, attr_name)
                            if attr_value is not None and isinstance(attr_value, torch.Tensor):
                                attr_array = attr_value.detach().cpu().numpy()
                                attr_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_{key}_{attr_name}.npy")
                                np.save(attr_file, attr_array)
                                saved_files[f'{key}_{attr_name}'] = attr_file
                                print(f"     ✅ {key}.{attr_name} 已保存: {attr_file} (形状: {attr_array.shape})")
                
                else:
                    print(f"     ⚠️ {key}: 未知类型 {type(value)}")
        
        elif isinstance(output, torch.Tensor):
            # 直接tensor输出
            feat_array = output.detach().cpu().numpy()
            feat_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_output_tensor.npy")
            np.save(feat_file, feat_array)
            saved_files['output_tensor'] = feat_file
            print(f"   ✅ 输出tensor已保存到: {feat_file} (形状: {feat_array.shape})")
        
        # 2. 保存输入特征用于对比
        input_feat_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_input_features.npy")
        input_lang_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_input_lang_features.npy")
        
        np.save(input_feat_file, input_dict['feat'].cpu().numpy())
        np.save(input_lang_file, input_dict['lang_feat'].cpu().numpy())
        
        print(f"   ✅ 输入几何特征已保存到: {input_feat_file}")
        print(f"   ✅ 输入语言特征已保存到: {input_lang_file}")
        
        # 3. 保存输入信息
        input_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_input_info.txt")
        with open(input_file, 'w') as f:
            f.write(f"SceneSplat 语言模型推理 - 输入信息\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"时间戳: {timestamp}\n")
            f.write(f"场景路径: {scene_path}\n")
            f.write(f"场景名称: {scene_name}\n\n")
            
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    f.write(f"{key}: 形状={value.shape}, 类型={value.dtype}\n")
                else:
                    f.write(f"{key}: {type(value)} = {value}\n")
        
        print(f"   ✅ 输入信息已保存到: {input_file}")
        
        # 4. 创建推理摘要
        summary_file = os.path.join(results_dir, f"{scene_name}_{timestamp}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"SceneSplat 语言模型推理摘要\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"推理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"场景: {scene_name}\n")
            f.write(f"输入点数: {input_dict['coord'].shape[0]}\n")
            f.write(f"输入几何特征维度: {input_dict['feat'].shape[1]}\n")
            f.write(f"输入语言特征维度: {input_dict['lang_feat'].shape[1]}\n")
            
            if isinstance(output, dict) and 'point_feat' in output:
                point_feat = output['point_feat']
                if hasattr(point_feat, 'feat'):
                    f.write(f"输出特征维度: {point_feat.feat.shape[1]}\n")
                    f.write(f"输出特征范围: [{point_feat.feat.min().item():.6f}, {point_feat.feat.max().item():.6f}]\n")
                    f.write(f"输出特征均值: {point_feat.feat.mean().item():.6f}\n")
                    f.write(f"输出特征标准差: {point_feat.feat.std().item():.6f}\n")
                    
                    # 计算特征相似度
                    input_lang_on_gpu = input_dict['lang_feat']
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        point_feat.feat, input_lang_on_gpu, dim=1
                    ).mean().item()
                    f.write(f"与输入语言特征相似度: {cosine_sim:.6f}\n")
            
            f.write(f"\n生成的文件:\n")
            if 'feat_file' in locals():
                f.write(f"- 输出特征向量: {os.path.basename(feat_file)}\n")
            if 'coord_file' in locals():
                f.write(f"- 输出坐标信息: {os.path.basename(coord_file)}\n")
            if 'input_feat_file' in locals():
                f.write(f"- 输入几何特征: {os.path.basename(input_feat_file)}\n")
                f.write(f"- 输入语言特征: {os.path.basename(input_lang_file)}\n")
            f.write(f"- 输入信息: {os.path.basename(input_file)}\n")
            f.write(f"- 推理摘要: {os.path.basename(summary_file)}\n")
        
        print(f"   ✅ 推理摘要已保存到: {summary_file}")
        print(f"\n📁 所有结果文件保存在: {results_dir}")
        
        return results_dir
        
    except Exception as e:
        print(f"   ❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_output(output, input_coord, input_feat, input_lang_feat):
    """详细分析推理输出"""
    print("\n" + "=" * 50)
    print("📊 推理输出详细分析")
    print("=" * 50)
    
    if output is None:
        print("❌ 无输出结果可分析")
        return
    
    print(f"1. 输出数据结构:")
    print(f"   类型: {type(output)}")
    
    if isinstance(output, dict):
        print(f"   字典键: {list(output.keys())}")
        
        for key, value in output.items():
            print(f"\n2. 输出项: {key}")
            if isinstance(value, torch.Tensor):
                print(f"   - 形状: {value.shape}")
                print(f"   - 数据类型: {value.dtype}")
                print(f"   - 设备: {value.device}")
                print(f"   - 值范围: [{value.min().item():.6f}, {value.max().item():.6f}]")
                print(f"   - 均值: {value.mean().item():.6f}")
                print(f"   - 标准差: {value.std().item():.6f}")
                
                # 显示前几个值的样本
                if value.numel() <= 20:
                    print(f"   - 所有值: {value.flatten()[:10].cpu().tolist()}")
                else:
                    print(f"   - 前10个值: {value.flatten()[:10].cpu().tolist()}")
                    
            elif hasattr(value, 'feat') and hasattr(value, 'coord'):
                # Point对象
                print(f"   - Point对象属性:")
                if hasattr(value, 'feat') and value.feat is not None:
                    feat_tensor = value.feat
                    print(f"     * feat形状: {feat_tensor.shape}")
                    print(f"     * feat数据类型: {feat_tensor.dtype}")
                    print(f"     * feat值范围: [{feat_tensor.min().item():.6f}, {feat_tensor.max().item():.6f}]")
                    print(f"     * feat均值: {feat_tensor.mean().item():.6f}")
                    print(f"     * feat标准差: {feat_tensor.std().item():.6f}")
                
                if hasattr(value, 'coord') and value.coord is not None:
                    coord_tensor = value.coord
                    print(f"     * coord形状: {coord_tensor.shape}")
                    print(f"     * coord值范围: [{coord_tensor.min().item():.6f}, {coord_tensor.max().item():.6f}]")
                
                # 检查其他可能的属性
                attrs = ['grid_coord', 'serialized_code', 'index', 'additional']
                for attr in attrs:
                    if hasattr(value, attr):
                        attr_value = getattr(value, attr)
                        if attr_value is not None:
                            if isinstance(attr_value, torch.Tensor):
                                print(f"     * {attr}形状: {attr_value.shape}")
                            else:
                                print(f"     * {attr}类型: {type(attr_value)}")
            else:
                print(f"   - 其他类型: {type(value)}")
                print(f"   - 字符串表示: {str(value)[:100]}...")
    
    elif isinstance(output, torch.Tensor):
        print(f"   - 形状: {output.shape}")
        print(f"   - 数据类型: {output.dtype}")
        print(f"   - 设备: {output.device}")
        print(f"   - 值范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
        print(f"   - 均值: {output.mean().item():.6f}")
        print(f"   - 标准差: {output.std().item():.6f}")
    
    # 对比输入输出
    print(f"\n3. 输入输出对比:")
    print(f"   输入点数: {input_coord.shape[0]}")
    print(f"   输入特征维度: {input_feat.shape[1]}")
    print(f"   输入语言特征维度: {input_lang_feat.shape[1]}")
    
    if isinstance(output, dict) and 'point_feat' in output:
        point_feat = output['point_feat']
        if hasattr(point_feat, 'feat'):
            print(f"   输出特征维度: {point_feat.feat.shape[1] if point_feat.feat.ndim > 1 else 'scalar'}")
            print(f"   特征维度变化: {input_feat.shape[1]} -> {point_feat.feat.shape[1] if point_feat.feat.ndim > 1 else 'scalar'}")
    
    # 深度特征分析
    print(f"\n4. 特征构成分析:")
    if isinstance(output, dict) and 'point_feat' in output:
        point_feat = output['point_feat']
        if hasattr(point_feat, 'feat'):
            output_feat = point_feat.feat
            print(f"   输出特征维度: {output_feat.shape[1]}")
            print(f"   输入语言特征维度: {input_lang_feat.shape[1]}")
            
            # 检查是否与输入语言特征相似
            if output_feat.shape[1] == input_lang_feat.shape[1]:
                print(f"   ✅ 输出特征维度与输入语言特征一致 ({output_feat.shape[1]}维)")
                
                # 计算与输入语言特征的相似性
                input_lang_on_gpu = input_lang_feat.cuda()
                cosine_sim = torch.nn.functional.cosine_similarity(
                    output_feat, input_lang_on_gpu, dim=1
                ).mean().item()
                
                print(f"   与输入语言特征的余弦相似度: {cosine_sim:.6f}")
                
                if cosine_sim > 0.8:
                    print(f"   🔍 高相似度 - 输出可能主要保持了输入语言特征")
                elif cosine_sim > 0.3:
                    print(f"   🔍 中等相似度 - 输出是语言特征与几何特征的融合")
                else:
                    print(f"   🔍 低相似度 - 输出是大幅变换后的特征")
                
                # 分析特征变化
                input_lang_stats = {
                    'mean': input_lang_on_gpu.mean().item(),
                    'std': input_lang_on_gpu.std().item(),
                    'min': input_lang_on_gpu.min().item(),
                    'max': input_lang_on_gpu.max().item()
                }
                
                output_stats = {
                    'mean': output_feat.mean().item(),
                    'std': output_feat.std().item(),
                    'min': output_feat.min().item(),
                    'max': output_feat.max().item()
                }
                
                print(f"\n   输入语言特征统计:")
                print(f"     均值: {input_lang_stats['mean']:.6f}")
                print(f"     标准差: {input_lang_stats['std']:.6f}")
                print(f"     范围: [{input_lang_stats['min']:.6f}, {input_lang_stats['max']:.6f}]")
                
                print(f"\n   输出特征统计:")
                print(f"     均值: {output_stats['mean']:.6f}")
                print(f"     标准差: {output_stats['std']:.6f}")
                print(f"     范围: [{output_stats['min']:.6f}, {output_stats['max']:.6f}]")
                
                # 检查是否有几何信息的编码
                print(f"\n   几何特征融合分析:")
                print(f"   输入几何特征维度: {input_feat.shape[1]}维")
                
                # 简单的线性相关性检查
                geometry_feat_expanded = input_feat.cuda().unsqueeze(-1).expand(-1, -1, output_feat.shape[1] // input_feat.shape[1])
                geometry_feat_flat = geometry_feat_expanded.reshape(input_feat.shape[0], -1)[:, :output_feat.shape[1]]
                
                if geometry_feat_flat.shape[1] == output_feat.shape[1]:
                    geo_corr = torch.nn.functional.cosine_similarity(
                        output_feat, geometry_feat_flat, dim=1
                    ).mean().item()
                    print(f"   与几何特征的相关性: {geo_corr:.6f}")
                    
                    if abs(geo_corr) > 0.1:
                        print(f"   🔍 检测到几何特征影响")
                    else:
                        print(f"   🔍 几何特征影响较小")
            else:
                print(f"   输出维度与语言特征不同，可能是混合表示")
    
    
    # 保存分析结果
    print(f"\n6. 结果保存:")
    analysis_file = "/home/huajianzeng/project/SceneSplat/inference_analysis.txt"
    try:
        with open(analysis_file, 'w') as f:
            f.write("SceneSplat 语言模型推理输出分析\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"输出类型: {type(output)}\n")
            if isinstance(output, dict):
                for key, value in output.items():
                    f.write(f"\n{key}:\n")
                    if isinstance(value, torch.Tensor):
                        f.write(f"  形状: {value.shape}\n")
                        f.write(f"  数据类型: {value.dtype}\n")
                        f.write(f"  值范围: [{value.min().item():.6f}, {value.max().item():.6f}]\n")
                        f.write(f"  均值: {value.mean().item():.6f}\n")
                    elif hasattr(value, 'feat'):
                        f.write(f"  Point对象 - feat形状: {value.feat.shape}\n")
        print(f"   ✅ 分析结果已保存到: {analysis_file}")
    except Exception as e:
        print(f"   ⚠️ 保存失败: {e}")

if __name__ == "__main__":
    result = test_single_scene()
    if isinstance(result, tuple):
        success, output = result
        if success:
            print("\n🎉 单场景语言模型推理测试成功!")
            
            # 重新加载数据进行分析对比
            scene_path = "/home/huajianzeng/project/SceneSplat/scannet_mcmc_3dgs_preprocessed/test_grid1.0cm_chunk6x6_stride3x3/scene0708_00_0"
            coord = np.load(os.path.join(scene_path, 'coord.npy'))
            color = np.load(os.path.join(scene_path, 'color.npy'))
            opacity = np.load(os.path.join(scene_path, 'opacity.npy'))
            quat = np.load(os.path.join(scene_path, 'quat.npy'))
            scale = np.load(os.path.join(scene_path, 'scale.npy'))
            lang_feat = np.load(os.path.join(scene_path, 'lang_feat.npy'))
            
            # 使用相同的子集
            n_points = min(10000, len(coord))
            coord = coord[:n_points]
            color = color[:n_points]
            opacity = opacity[:n_points]
            quat = quat[:n_points]  
            scale = scale[:n_points]
            lang_feat = lang_feat[:n_points]
            feat = np.concatenate([color, quat, scale, opacity[:, None]], axis=1)
            
            # 转换为tensor用于对比
            coord_tensor = torch.from_numpy(coord).float()
            feat_tensor = torch.from_numpy(feat).float()
            lang_feat_tensor = torch.from_numpy(lang_feat).float()
            
            # 分析输出
            analyze_output(output, coord_tensor, feat_tensor, lang_feat_tensor)
            
        else:
            print("\n❌ 推理测试失败")
    else:
        # 兼容旧版本返回值
        success = result
        if success:
            print("\n🎉 单场景语言模型推理测试成功!")
        else:
            print("\n❌ 推理测试失败")
    
    sys.exit(0 if success else 1)