import os
import cv2
import numpy as np
import random
import shutil

class gray_grad:
    def __init__(self, max_val, min_val, grad, path, out, num):
        # 输入：max=1,min小于1大于0的数字
        self.max_val = max_val
        self.min_val = min_val
        # grad:阶层数量
        self.grad = grad
        # path,指向图片来源文件夹路径
        self.path = path
        # out,指向导出文件夹
        self.out = out
        # num: 每个阶层随机抽取的图片数量
        self.num = num
        
    def _run(self):
        # 获取所有png图片
        all_images = [f for f in os.listdir(self.path) if f.lower().endswith('.png')]
        
        for i in range(self.grad + 1):
            # 计算当前阶层的阈值
            threshold = self.min_val + i * (self.max_val - self.min_val) / self.grad
            # 创立文件夹名称为pre_grad值的文件夹
            folder_name = f"{threshold:.3f}"
            folder_path = os.path.join(self.out, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            # 随机在path下拿num张图片
            selected = random.sample(all_images, min(self.num, len(all_images)))
            
            for img_name in selected:
                img_path = os.path.join(self.path, img_name)
                # 读取图片
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                    
                # 如果图片已经是灰度，否则转化为灰度
                if len(img.shape) == 3 and img.shape[2] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                else:
                    gray = img.copy()
                
                # 转化为数组元素，元素先/255
                normalized = gray.astype(np.float32) / 255.0
                
                # 把图片中不在max和阈值期间的元素变为0
                # 保留 [threshold, max_val] 范围内的像素，其余置0
                mask = (normalized >= threshold) & (normalized <= self.max_val)
                result = normalized * mask
                
                # 转换回0-255范围
                result = (result * 255).astype(np.uint8)
                
                # 保存到对应文件夹
                save_path = os.path.join(folder_path, img_name)
                cv2.imwrite(save_path, result)
                print(f"已处理: {img_name} -> {folder_name}")

# 使用示例
s = gray_grad(1, 0.4, 10, r"C:\mypy\recog_ai\imgs", r"C:\mypy\recog_ai\imgs", 5)
s._run()