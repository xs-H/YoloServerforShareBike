from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# ======================= 全局配置 =======================
FONT_PATH = "LXGWWenKai-Bold.ttf"  # 中文字体
FONT_SIZE = 44  # 默认字体大小
LINE_WIDTH = 8  # 默认线宽
LABEL_PADDING = (30, 18)  # 默认标签内边距（水平，垂直）
RADIUS = 10  # 默认圆角半径
TEXT_COLOR = (0, 0, 0)  # 默认文本颜色（RGB，黑色）

LABEL_MAPPING = {
    "safety_helmet": "安全帽",
    "reflective_vest": "反光背心",
    "person": "人员",
    "head": "头部",
    "ordinary_clothes": "普通服装"
}
COLOR_MAPPING = {
    "safety_helmet": (50, 205, 50),  # 鲜绿色
    "reflective_vest": (255, 255, 0),  # 亮黄色
    "person": (65, 105, 255),  # 橙色
    "head": (0, 0, 255),  # 红色
    "ordinary_clothes": (226, 43, 138)  # 紫色
}
text_size_cache = {}

# ======================= 美化函数 =======================
def draw_text(image, text, position, font_obj, fill=TEXT_COLOR):
    """使用 PIL 绘制中英文文本"""
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font_obj, fill=fill)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def get_text_size(text, font_obj):
    """计算文本尺寸（带缓存）"""
    cache_key = f"{text}_{font_obj.size}"
    if cache_key in text_size_cache:
        return text_size_cache[cache_key]
    temp_image = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_image)
    bbox = draw.textbbox((0, 0), text, font=font_obj)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    text_size_cache[cache_key] = (width, height)
    return (width, height)

def draw_top_rounded_label(image, x1, y1, x2, y2, color, radius=RADIUS):
    """绘制顶部圆角标签框"""
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2), color, -1)
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y1 + radius), color, -1)
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    return image

def custom_plot(
    image,
    boxes,
    confs,
    labels,
    use_chinese_mapping=True,
    beautify=True,
    font_size=FONT_SIZE,
    line_width=LINE_WIDTH,
    label_padding=LABEL_PADDING,
    radius=RADIUS,
    text_color=TEXT_COLOR
):
    """
    绘制检测框和标签

    参数:
        image: 输入图像（numpy数组，BGR）
        boxes: 检测框列表，[[x1,y1,x2,y2], ...]
        confs: 置信度列表
        labels: 标签列表（类别名称）
        use_chinese_mapping: 是否使用中文标签（默认True）
        beautify: 是否启用美化模式（默认True）
        font_size: 字体大小（默认22）
        line_width: 检测框线宽（默认4）
        label_padding: 标签内边距（水平，垂直，默认(30,18)）
        radius: 标签圆角半径（默认8）
        text_color: 文本颜色（RGB，默认黑色）

    返回:
        绘制后的图像（numpy数组，BGR）
    """
    result_image = image.copy()
    font = ImageFont.truetype(FONT_PATH, font_size)
    padding_x, padding_y = label_padding

    for box, conf, label in zip(boxes, confs, labels):
        x1, y1, x2, y2 = map(int, box)
        color = COLOR_MAPPING.get(label, (0, 255, 0))
        label_text = f"{LABEL_MAPPING.get(label, label)} {conf:.0%}" if use_chinese_mapping else f"{label} {conf:.0%}"
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, line_width)

        if beautify:
            text_width, text_height = get_text_size(label_text, font)
            text_width += padding_x
            label_height = text_height + padding_y
            label_y1 = max(0, y1 - label_height)
            label_y2 = y1
            label_x1 = int(x1 - line_width / 2)
            label_x2 = int(label_x1 + text_width)
            result_image = draw_top_rounded_label(result_image, label_x1, label_y1, label_x2, label_y2, color, radius)
            text_x = label_x1 + padding_x // 2
            text_y = label_y1 + (label_height - text_height) // 2
            result_image = draw_text(result_image, label_text, (text_x, text_y), font, text_color)
        else:
            text_width, text_height = get_text_size(label_text, font)
            text_x = x1
            text_y = y1 - text_height - 10 if y1 > text_height + 10 else y1 + 10
            result_image = draw_text(result_image, label_text, (text_x, text_y), font, fill=color)

    return result_image