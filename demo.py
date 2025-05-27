from ultralytics import YOLO

if __name__ == "__main__":
   # yolo环境检测
    from ultralytics import YOLO

    # 从配置文件创建新的 YOLOv8 模型（未训练）
    model = YOLO("yolov8n.yaml")

    # 加载预训练的 YOLOv8 模型（推荐用于训练）
    model = YOLO("yolov8n.pt")

    # 使用 coco8.yaml 数据集训练模型，训练 3 个 epoch
    results = model.train(data="coco8.yaml", epochs=3)

    # 在验证集上评估模型性能
    results = model.val()

    # 使用模型对图像进行目标检测
    results = model("https://ultralytics.com/images/bus.jpg")

    # 显示带边界框的图像
    results[0].show()