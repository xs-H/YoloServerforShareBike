import cv2
import numpy as np
from utils.beautify import custom_plot

def test_custom_plot():
    # 替换为你的图片路径
    image_path = "pos_194.jpg"  # 请确保这个路径指向一张真实的图片
    output_path = "output_image.jpg"

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图片: {image_path}")

    # 获取图像尺寸
    img_height, img_width = image.shape[:2]

    # YOLO 标签数据
    yolo_labels = [
        [2, 0.688358, 0.656326, 0.210483, 0.686131, 0.972656],
        [2, 0.334435, 0.690389, 0.222956, 0.610706, 0.969727],
        [0, 0.332291, 0.485097, 0.113427, 0.207421, 0.961914],
        [1, 0.32508, 0.79927, 0.199569, 0.381995, 0.959961],
        [1, 0.686799, 0.65146, 0.169946, 0.340633, 0.959473],
        [0, 0.690697, 0.393552, 0.0841933, 0.164234, 0.95752]
    ]

    # 类别名称
    names = ["safety_helmet", "reflective_vest", "person", "head", "ordinary_clothes"]

    # 转换 YOLO 标签为 custom_plot 所需的格式
    boxes = []
    confs = []
    labels = []

    for label in yolo_labels:
        class_id, center_x, center_y, width, height, conf = label
        # 将归一化坐标转换为像素坐标
        x1 = int((center_x - width / 2) * img_width)
        y1 = int((center_y - height / 2) * img_height)
        x2 = int((center_x + width / 2) * img_width)
        y2 = int((center_y + height / 2) * img_height)
        boxes.append([x1, y1, x2, y2])
        confs.append(conf)
        labels.append(names[int(class_id)])

    # 调用美化绘制函数
    result_image = custom_plot(
        image,
        boxes,
        confs,
        labels,
        use_chinese_mapping=True,  # 使用中文标签
        beautify=True,  # 启用美化模式（圆角标签）
        font_size=22,
        line_width=4,
        label_padding=(20, 18),
        radius=6,
        text_color=(0, 0, 0)
    )

    # 保存结果
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存至: {output_path}")

    # 显示结果（可选）
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_custom_plot()