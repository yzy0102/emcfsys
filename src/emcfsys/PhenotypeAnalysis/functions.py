import numpy as np
import pandas as pd
from skimage.measure import regionprops, regionprops_table
from PIL import Image # 虽然主要用skimage，但保留PIL以防你需要处理特殊格式
from scipy.stats import norm


from skimage.measure import label as sk_label  # 用于连通域标记
import cv2
from scipy import signal



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
        
    # 4. 计算电子密度 (Electron Density)
    if features_to_compute.get("Electron Density", True):
        intensity = Cal_intensity_test(instance_mask, img)
        
        # 修复潜在的长度对齐问题：确保 intensity 的长度与 df 的行数一致
        if len(intensity) == len(df):
            df['electron_density'] = intensity
        else:
            print("Warning: Label count mismatch. Electron density might be misaligned.")
            df['electron_density'] = np.nan # 如果长度不匹配，先填入 nan 防止报错
    
    
    # 5. 准备返回的列
    column_mapping = {
        "Area": "area",
        "Perimeter": "perimeter",
        "Elongation": "elongation",
        "label": "label",
        "centroid-0": "centroid-0", 
        "centroid-1": "centroid-1",  
        "Electron Density": "electron_density", 
    }
    
    # 始终包含 label 和 centroid 以便交互
    selected_columns = ["label", "centroid-0", "centroid-1"]
    
    for key, val in features_to_compute.items():
        if val and key in column_mapping:
            internal_column = column_mapping[key]
            # 确保要提取的列确实已经计算并存在于 DataFrame 中
            if internal_column not in selected_columns and internal_column in df.columns:
                selected_columns.append(internal_column)
    

            
    return df[selected_columns], instance_mask


def get_back_intensity(old_img):

    hist1 = cv2.calcHist([np.array(old_img)], [0], None, [256], [0, 256])

    hist1 = hist1[1:254]
    bins = np.arange(hist1.shape[0] + 1)
    x = np.array(hist1)
    mu = np.mean(x)
    sigma = np.std(x)
    y = norm.pdf(hist1, mu, sigma)

    xxx = bins[1:]
    yyy = y.ravel()
    z1 = np.polyfit(xxx, yyy, 100)
    p1 = np.poly1d(z1)
    yvals = p1(xxx)
    num_peak_3 = signal.find_peaks(yvals, distance=10)
    def get_tensity(num_peak_3=num_peak_3, y=y):
        num = [n - y.argmin() for n in num_peak_3[0]]
        num = np.array(num)
        num = np.where(num < 0, 255, num)
        return np.sort(num)[0] + y.argmin()
    return get_tensity(num_peak_3, y)


def Cal_intensity_test(Organelle_Instance, gray_img):
    """
    Return the electron-intensity of the Orgfanelle instances
    """
    background_intensity = get_back_intensity(gray_img)
    num_labels = Organelle_Instance.max()
    labeled_image = Organelle_Instance
    intensity = []
    separated_regions = []
    for label in range(1, num_labels + 1):
        mask = np.uint8(labeled_image == label)
        separated_region = cv2.bitwise_and(gray_img, gray_img, mask=mask)
        separated_regions.append(separated_region)

        hist = cv2.calcHist([np.array(separated_region)], [0], mask[:, :], [256], [0, 256])
        x = np.array(hist)
        mu = np.mean(x)
        sigma = np.std(x)
        y = norm.pdf(hist, mu, sigma)
        intensity.append(y.argmin() / background_intensity)


    return intensity

