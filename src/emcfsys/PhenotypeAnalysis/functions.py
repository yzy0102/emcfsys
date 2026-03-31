import numpy as np
import pandas as pd
from skimage.measure import regionprops, regionprops_table
from PIL import Image # 虽然主要用skimage，但保留PIL以防你需要处理特殊格式
from scipy.stats import norm
from shapely.geometry import Polygon, LineString


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
            
# 5. 计算形状复杂度 (Shape Complexity)
    if features_to_compute.get("Shape Complexity", True):
        complexities = []
        # 遍历每一个被标记的细胞器
        for label in df['label'].values:
            single_mask = (instance_mask == label)
            # 默认使用 50 个采样点，可以根据需要调整以平衡速度和精度
            complexity_val = calculate_shape_complexity(single_mask, num_points=50)
            complexities.append(complexity_val)
            
        df['shape_complexity'] = complexities
    
    # 5. 准备返回的列
    column_mapping = {
        "Area": "area",
        "Perimeter": "perimeter",
        "Elongation": "elongation",
        "label": "label",
        "centroid-0": "centroid-0", 
        "centroid-1": "centroid-1",  
        "Electron Density": "electron_density", 
        "Shape Complexity": "shape_complexity",
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



def calculate_shape_complexity(instance_mask, num_points=50):
    """
    计算基于可见性图的形状复杂度
    公式: Complexity = 2m / (n * (n - 1))
    :param instance_mask: 单个实例的二值化掩码 (0和1)
    :param num_points: 轮廓上等间距采样的有效点数 n，默认50点
    """
    # 1. 提取外部轮廓
    mask_uint8 = np.uint8(instance_mask) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.nan
        
    # 获取最大轮廓（过滤掉可能的噪点）
    contour = max(contours, key=cv2.contourArea).squeeze()
    
    # 如果轮廓点太少构不成多边形
    if len(contour.shape) < 2 or len(contour) < 3:
        return np.nan
        
    # 2. 构建多边形并处理潜在的自交叉无效几何体
    poly = Polygon(contour)
    if not poly.is_valid:
        poly = poly.buffer(0)
        
    # 3. 在多边形边界上等间距采样 n 个有效点
    perimeter = poly.length
    if perimeter == 0:
        return np.nan
        
    n = num_points
    distances = np.linspace(0, perimeter, n, endpoint=False)
    points = [poly.boundary.interpolate(d) for d in distances]
    
    m = 0  # 记录可见边的数量
    
    # 4. 两两连线并判断可见性 (遍历所有可能的点对)
    for i in range(n):
        for j in range(i + 1, n):
            line = LineString([points[i], points[j]])
            
            # 使用 difference 求线段与多边形的差异部分。
            # 差异部分长度接近 0，说明线段完全在多边形内部或边界上（即可见边）
            if line.difference(poly).length < 1e-5:
                m += 1
                
    # 5. 根据新公式计算复杂度
    # 防止 n 较小时除以零的错误
    if n <= 1:
        return np.nan
        
    complexity = (2 * m) / (n * (n - 1))
    
    return complexity