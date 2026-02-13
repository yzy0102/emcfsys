import numpy as np
import numpy as np
import pandas as pd
from skimage.measure import regionprops, regionprops_table
from PIL import Image # 虽然主要用skimage，但保留PIL以防你需要处理特殊格式

import numpy as np
import pandas as pd
from skimage.measure import label as sk_label  # 用于连通域标记
from skimage.measure import regionprops_table

def analyze_phenotypes(img, label_data, features_to_compute):
    """
    支持实例分割的表型分析函数
    :param img: 原始灰度图 (numpy array)
    :param label_data: 语义分割或实例分割图 (numpy array)
    :param features_to_compute: 用户选择的特征字典
    """
    label_data = np.squeeze(label_data)  # 确保是2D数组
    # 1. 检查并转换：将语义分割转换为实例分割
    # 如果图中只有 0 和 1（或者只有一个非零值），说明需要做连通域标记
    unique_values = np.unique(label_data)
    if len(unique_values) <= 2:
        # 使用 8 连通域算法标记每一个独立的物体
        instance_mask = sk_label(label_data > 0)
        print(f"Detected semantic mask. Found {instance_mask.max()} instances.")
    else:
        instance_mask = label_data
        print(f"Detected instance mask with {len(unique_values)-1} objects.")

    # 2. 基础属性提取
    # 增加 'centroid' 坐标，方便后续点击表格自动跳转定位
    props = ['label', 'area', 'perimeter', 'major_axis_length', 'minor_axis_length', 'centroid',]
    
    # 使用 intensity_image=img 可以提取像素强度相关的指标（如平均亮度）
    stats = regionprops_table(instance_mask, intensity_image=img, properties=props)
    
    df = pd.DataFrame(stats)
    
    # 3. 计算伸长率 (Elongation)
    if features_to_compute.get("Elongation", True):
        # 伸长率公式：$Elongation = \frac{MajorAxisLength}{MinorAxisLength}$
        df['elongation'] = df['major_axis_length'] / (df['minor_axis_length'] + 1e-6)
    
    # 4. 准备返回的列
    column_mapping = {
        "Area": "area",
        "Perimeter": "perimeter",
        "Elongation": "elongation",
        "label": "label",
        "centroid-0": "centroid-0", # y 坐标
        "centroid-1": "centroid-1"  # x 坐标
    }
    
    # 始终包含 label 和 centroid 以便交互
    selected_columns = ["label", "centroid-0", "centroid-1"]
    for key, val in features_to_compute.items():
        if val and key in column_mapping:
            if column_mapping[key] not in selected_columns:
                selected_columns.append(column_mapping[key])
    

            
    return df[selected_columns], instance_mask