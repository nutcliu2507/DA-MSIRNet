import os
from skimage import io
import numpy as np

# 圖片資料夾路徑
folder_path = r"C:\Users\2507\Desktop\spa-former-main\checkpoints\predict_cb_325000_C24_1222\10-20\truth"

# 初始化最大最小值
global_min = float('inf')
global_max = float('-inf')

# 遍歷資料夾內的所有圖片
for filename in os.listdir(folder_path):
    if filename.endswith((".png", ".jpg", ".jpeg")):  # 過濾圖片格式
        img_path = os.path.join(folder_path, filename)
        img = io.imread(img_path)  # 讀取圖像為 NumPy 陣列

        # 計算當前圖片的最小 & 最大值
        img_min = np.min(img)
        img_max = np.max(img)

        # 更新全域最大最小值
        global_min = min(global_min, img_min)
        global_max = max(global_max, img_max)

print(f"Dataset Min: {global_min}, Dataset Max: {global_max}")
