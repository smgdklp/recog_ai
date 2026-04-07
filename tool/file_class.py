import os
import random
import shutil

def class_date(class_path, data_path):
    os.makedirs(os.path.join(class_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(class_path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(class_path, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(class_path, "labels", "val"), exist_ok=True)
    
    pre_png = None
    pre_txt = None
    #data是分类源，分到class路径
    # 遍历data内.png文件，如果有和它一样名称的.txt，没有对应txt的忽视
    # 则将pre_png复制到class_path/images/train/, pre_txt复制到class_path/labels/train/
    for file in os.listdir(data_path):
        if file.endswith(".png"):
            png_path = os.path.join(data_path, file)
            txt_name = file.replace(".png", ".txt")
            txt_path = os.path.join(data_path, txt_name)
            
            if os.path.exists(txt_path):
                pre_png = png_path
                pre_txt = txt_path
                
                dest_png = os.path.join(class_path, "images", "train", file)
                dest_txt = os.path.join(class_path, "labels", "train", txt_name)
                
                shutil.copy2(pre_png, dest_png)
                shutil.copy2(pre_txt, dest_txt)

def class_train(class_path, proportion):
    # proportion: 输入一个不大于1大于0的浮点数，作为训练集比率
    pre_png = None  # 做遍历png缓存容器
    pre_txt = None  # 做遍历txt缓存容器
    random_index = None  # 随机数容器
    
    train_images_dir = os.path.join(class_path, "images", "train")
    train_labels_dir = os.path.join(class_path, "labels", "train")
    val_images_dir = os.path.join(class_path, "images", "val")
    val_labels_dir = os.path.join(class_path, "labels", "val")
    
    # 遍历所有class_path/images/train/.png
    for png_file in os.listdir(train_images_dir):
        if png_file.endswith(".png"):
            random_index = random.random()
            if random_index > proportion:
                pre_png = os.path.join(train_images_dir, png_file)
                txt_file = png_file.replace(".png", ".txt")
                pre_txt = os.path.join(train_labels_dir, txt_file)
                
               
                if os.path.exists(pre_txt):
                    shutil.move(pre_png, os.path.join(val_images_dir, png_file))
                    shutil.move(pre_txt, os.path.join(val_labels_dir, txt_file))

def test(class_path):
    index = 0 
    
    # 遍历class_path/images/train/.png，检查是否在class_path/labels/train/有对应名称的txt
    train_images_dir = os.path.join(class_path, "images", "train")
    train_labels_dir = os.path.join(class_path, "labels", "train")
    
    for png_file in os.listdir(train_images_dir):
        if png_file.endswith(".png"):
            txt_name = png_file.replace(".png", ".txt")
            txt_path = os.path.join(train_labels_dir, txt_name)
            if not os.path.exists(txt_path):
                index += 1
                print(f"训练集缺失标注: {png_file}")
    
    # 遍历class_path/images/val/.png，检查是否在class_path/labels/val/有对应名称的txt
    val_images_dir = os.path.join(class_path, "images", "val")
    val_labels_dir = os.path.join(class_path, "labels", "val")
    
    for png_file in os.listdir(val_images_dir):
        if png_file.endswith(".png"):
            txt_name = png_file.replace(".png", ".txt")
            txt_path = os.path.join(val_labels_dir, txt_name)
            if not os.path.exists(txt_path):
                index += 1
                print(f"验证集缺失标注: {png_file}")
    
    if index == 0:
        print("所有图片都有对应的标注文件")
    else:
        print(f"共发现 {index} 个缺失标注的图片")
    
    return index

if __name__ == "__main__":
    class_path = r"C:\mypy\recog_ai\data"
    data_path = r"C:\mypy\recog_ai\imgs"
    class_date(class_path, data_path)
    class_train(class_path, 0.8)
    test(class_path)