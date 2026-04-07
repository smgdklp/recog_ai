
from ultralytics import YOLO

model = YOLO('yolov8n.pt')#小而美，确信

results = model.train(
    data=r"C:\mypy\recog_ai\data\data.yaml",
    max_det=1,
    epochs=200,
    batch=4,   
    lr0=0.001,
    imgsz=416,  
    workers=2,               # 数据加载线程
    device='cpu',            # 核显用cpu模式（或用'xpu'需特殊配置）
    amp=False,               # 自动混合精度，核显上建议先关闭
    optimizer='SGD',         # SGD比AdamW省内存
    patience=50,             # 早停，50轮没提升就停
    cache=False,             # 不缓存图片到内存，'ram'缓存加速
    exist_ok=True,           # 允许覆盖已有结果
    verbose=True,            # 打印训练日志
)