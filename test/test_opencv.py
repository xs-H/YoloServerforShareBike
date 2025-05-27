import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont # 引入Pillow用于中文支持

# --- 辅助函数：绘制可定制圆角的填充矩形 (OpenCV) ---
def draw_filled_rounded_rect(image_np, pt1, pt2, color_bgr, radius,
                             top_left_round=True, top_right_round=True,
                             bottom_left_round=True, bottom_right_round=True):
    """
    使用 OpenCV 绘制颜色填充的圆角矩形，可控制每个角的圆角状态。

    Args:
        image_np (numpy.array): OpenCV 图像对象 (BGR 格式)。
        pt1 (tuple): 矩形左上角坐标 (x1, y1)。
        pt2 (tuple): 矩形右下角坐标 (x2, y2)。
        color_bgr (tuple): 填充颜色 (B, G, R)。
        radius (int): 圆角半径。
        top_left_round (bool): 左上角是否圆角。
        top_right_round (bool): 右上角是否圆角。
        bottom_left_round (bool): 左下角是否圆角。
        bottom_right_round (bool): 右下角是否圆角。
    """
    x1, y1 = pt1
    x2, y2 = pt2
    color = color_bgr
    thickness = -1 # 填充

    # 绘制矩形主体（覆盖大部分直角区域）
    # 水平矩形部分
    cv2.rectangle(image_np, (x1 + (radius if top_left_round else 0), y1),
                  (x2 - (radius if top_right_round else 0), y2), color, thickness)
    # 垂直矩形部分
    cv2.rectangle(image_np, (x1, y1 + (radius if top_left_round else 0)),
                  (x2, y2 - (radius if bottom_left_round else 0)), color, thickness)

    # 绘制圆角部分 (填充圆形)
    if top_left_round:
        cv2.circle(image_np, (x1 + radius, y1 + radius), radius, color, thickness, cv2.LINE_AA)
    if top_right_round:
        cv2.circle(image_np, (x2 - radius, y1 + radius), radius, color, thickness, cv2.LINE_AA)
    if bottom_left_round:
        cv2.circle(image_np, (x1 + radius, y2 - radius), radius, color, thickness, cv2.LINE_AA)
    if bottom_right_round:
        cv2.circle(image_np, (x2 - radius, y2 - radius), radius, color, thickness, cv2.LINE_AA)


# --- 辅助函数：绘制可定制圆角的边框矩形 (OpenCV) ---
def draw_bordered_rounded_rect(image_np, pt1, pt2, color_bgr, thickness, radius,
                               top_left_round=True, top_right_round=True,
                               bottom_left_round=True, bottom_right_round=True):
    """
    使用 OpenCV 绘制带边框的圆角矩形，可控制每个角的圆角状态。
    这是由直线和椭圆弧组合而成。

    Args:
        image_np (numpy.array): OpenCV 图像对象 (BGR 格式)。
        pt1 (tuple): 矩形左上角坐标 (x1, y1)。
        pt2 (tuple): 矩形右下角坐标 (x2, y2)。
        color_bgr (tuple): 边框颜色 (B, G, R)。
        thickness (int): 边框厚度。
        radius (int): 圆角半径。
        top_left_round (bool): 左上角是否圆角。
        top_right_round (bool): 右上角是否圆角。
        bottom_left_round (bool): 左下角是否圆角。
        bottom_right_round (bool): 右下角是否圆角。
    """
    x1, y1 = pt1
    x2, y2 = pt2
    color = color_bgr
    line_type = cv2.LINE_AA

    # 绘制直线段
    # 上边
    cv2.line(image_np, (x1 + radius if top_left_round else x1, y1),
             (x2 - radius if top_right_round else x2, y1), color, thickness, line_type)
    # 下边
    cv2.line(image_np, (x1 + radius if bottom_left_round else x1, y2),
             (x2 - radius if bottom_right_round else x2, y2), color, thickness, line_type)
    # 左边
    cv2.line(image_np, (x1, y1 + radius if top_left_round else y1),
             (x1, y2 - radius if bottom_left_round else y2), color, thickness, line_type)
    # 右边
    cv2.line(image_np, (x2, y1 + radius if top_right_round else y1),
             (x2, y2 - radius if bottom_right_round else y2), color, thickness, line_type)

    # 绘制圆角弧
    if top_left_round:
        cv2.ellipse(image_np, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, line_type)
    if top_right_round:
        cv2.ellipse(image_np, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, line_type)
    if bottom_left_round:
        cv2.ellipse(image_np, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, line_type)
    if bottom_right_round:
        cv2.ellipse(image_np, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, line_type)


# --- 核心函数：绘制YOLO风格检测框 ---
def draw_detection_box(image_np, bbox, box_color_bgr, box_thickness=2, corner_radius=10):
    """
    使用 OpenCV 绘制YOLO风格的检测框（无填充，左上角直角，右上、左下、右下圆角）。

    Args:
        image_np (numpy.array): OpenCV 图像对象 (BGR 格式)。
        bbox (tuple): 检测框的坐标 (x_min, y_min, x_max, y_max)。
        box_color_bgr (tuple): 检测框的线条颜色 (B, G, R)。
        box_thickness (int): 检测框线条厚度。
        corner_radius (int): 圆角半径。
    """
    draw_bordered_rounded_rect(image_np, bbox[0:2], bbox[2:4],
                               box_color_bgr, box_thickness, corner_radius,
                               top_left_round=False, # 左上角直角
                               top_right_round=True,
                               bottom_left_round=True,
                               bottom_right_round=True)


# --- 核心函数：绘制YOLO风格标签框和文字 ---
def draw_label_box(image_np, bbox, label_text, label_color_bgr, text_color_bgr,
                   font_face_cv2, font_scale_cv2, text_thickness_cv2, # OpenCV 字体参数
                   font_path_pil, font_size_pil, # Pillow 字体参数
                   label_box_height=30, corner_radius=10,
                   horizontal_padding=10, vertical_padding=5, # 明确内边距
                   use_chinese_font=False, # 控制是否使用Pillow绘制中文
                   box_thickness_det=2): # 从检测框获取的厚度，用于对齐
    """
    使用 OpenCV 绘制YOLO风格的标签框（带填充，上半部分圆角，下半部分直角）并在其中添加文字。
    标签框的宽度根据文字内容自适应。支持中文/英文切换字体渲染方式。

    Args:
        image_np (numpy.array): OpenCV 图像对象 (BGR 格式)。
        bbox (tuple): 对应的检测框坐标 (x_min, y_min, x_max, y_max)，用于定位标签框。
        label_text (str): 要显示在标签框内的文字。
        label_color_bgr (tuple): 标签框的填充颜色 (B, G, R)。
        text_color_bgr (tuple): 文字颜色 (B, G, R)。
        font_face_cv2 (int): OpenCV 字体类型 (例如 cv2.FONT_HERSHEY_SIMPLEX)。
        font_scale_cv2 (float): OpenCV 字体大小缩放因子。
        text_thickness_cv2 (int): OpenCV 文字线条厚度。
        font_path_pil (str): Pillow 字体文件的路径 (例如 'C:/Windows/Fonts/msyh.ttc')。
        font_size_pil (int): Pillow 字体大小。
        label_box_height (int): 标签框的高度。
        corner_radius (int): 标签框上部圆角的半径。
        horizontal_padding (int): 文字左右两边的水平内边距。
        vertical_padding (int): 文字上下两边的垂直内边距。
        use_chinese_font (bool): 如果为True，使用Pillow绘制文字（支持中文）；否则使用OpenCV内置字体。
        box_thickness_det (int): 检测框的线条厚度，用于计算标签框的精确X对齐。
    """
    x_min_det, y_min_det, _, _ = bbox # 标签框只依赖检测框的左上角坐标

    # 1. 计算文字尺寸 (根据选择的字体渲染方式)
    if use_chinese_font:
        try:
            font_pil = ImageFont.truetype(font_path_pil, font_size_pil)
        except IOError:
            print(f"警告：无法加载字体文件 '{font_path_pil}'。将使用Pillow默认字体。")
            font_pil = ImageFont.load_default()

        # Pillow的textbbox获取文字的边界框
        # 注意：Pillow的textbbox在不同版本和系统可能有细微差异，这里获取的是一个近似值
        dummy_img = Image.new('RGB', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox_text_pil = dummy_draw.textbbox((0, 0), label_text, font=font_pil)
        text_width = bbox_text_pil[2] - bbox_text_pil[0]
        text_height = bbox_text_pil[3] - bbox_text_pil[1]
    else:
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font_face_cv2, font_scale_cv2, text_thickness_cv2)
        # OpenCV的getTextSize返回的text_height通常是文字的实际高度，baseline是基线到文字底部的距离
        # Pillow的textbbox返回的text_height通常包含基线，所以这里不需要特别减去baseline for height calculation
        # 但在计算y坐标时需要考虑baseline

    # 2. 根据文字尺寸和内边距计算标签框的实际宽度和高度
    # 标签框的高度现在可以通过文字高度 + 2*vertical_padding 来动态计算
    # 或者保持固定的 label_box_height，如果它更大
    calculated_label_height = text_height + 2 * vertical_padding
    final_label_box_height = max(label_box_height, calculated_label_height) # 取两者中较大值

    label_box_width = text_width + 2 * horizontal_padding
    if label_box_width < 2 * corner_radius: # 确保宽度至少能容纳两个半圆，避免圆角绘制异常
        label_box_width = 2 * corner_radius

    # 3. 计算标签框的最终坐标
    # 关键调整：标签框的x_min基于检测框的x_min和其厚度进行精确对齐
    # 由于边框线有一部分在框内，一部分在框外，通常视觉对齐需要考虑线宽的一半
    label_box_x_min = int(x_min_det - box_thickness_det / 2.0)
    label_box_y_min = y_min_det - final_label_box_height
    label_box_x_max = label_box_x_min + label_box_width
    label_box_y_max = y_min_det # 标签框下边与检测框上边对齐

    # 4. 绘制填充的标签框
    draw_filled_rounded_rect(image_np, (label_box_x_min, label_box_y_min), (label_box_x_max, label_box_y_max),
                             label_color_bgr, corner_radius,
                             top_left_round=True, # 左上角圆角
                             top_right_round=True, # 右上角圆角
                             bottom_left_round=False, # 左下角直角
                             bottom_right_round=False) # 右下角直角

    # 5. 在标签框内添加文字 (根据选择的字体渲染方式)
    if use_chinese_font:
        # 将OpenCV图像转换为Pillow图像，进行文字绘制，再转换回来
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        draw_pil = ImageDraw.Draw(image_pil)

        text_color_rgb = (text_color_bgr[2], text_color_bgr[1], text_color_bgr[0]) # BGR转RGB

        # 计算文字的起始坐标，使其在标签框内居中
        # Pillow的文本绘制坐标是左上角
        text_x = label_box_x_min + (label_box_width - text_width) // 2
        text_y = label_box_y_min + (final_label_box_height - text_height) // 2
        draw_pil.text((text_x, text_y), label_text, font=font_pil, fill=text_color_rgb)

        image_np[:] = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR) # 将修改后的Pillow图像写回OpenCV图像
    else:
        # 计算文字的起始坐标，使其在标签框内居中
        # OpenCV的putText的y坐标是文字基线
        text_x = label_box_x_min + (label_box_width - text_width) // 2
        text_y = label_box_y_min + (final_label_box_height - text_height) // 2 + text_height - baseline // 2

        cv2.putText(image_np, label_text, (text_x, text_y), font_face_cv2, font_scale_cv2,
                    text_color_bgr, text_thickness_cv2, cv2.LINE_AA)


# --- 主程序开始 ---

# 1. 创建一个空白图像作为画布
image_width, image_height = 800, 600
image_np = np.zeros((image_height, image_width, 3), dtype=np.uint8)
image_np.fill(200) # 填充为浅灰色背景 (BGR: 200, 200, 200)

# 2. 模拟一个YOLO检测框的BBox (x_min, y_min, x_max, y_max)
detection_bbox = (200, 200, 600, 500) # (左上x, 左上y, 右下x, 右下y)

# 3. 设置绘制参数
label_text_english = "Virus 100%"
label_text_chinese = "病毒检测 100.0%" # 示例中文文本

# 选择要显示的文本
current_label_text = label_text_chinese # 切换此处以测试中英文

label_fill_color_bgr = (95, 173, 101) # 标签框填充颜色 (BGR)
box_line_color_bgr = (95, 173, 101) # 检测框线条颜色 (BGR)
text_color_bgr = (0, 0, 0) # 文字颜色 (BGR)

box_thickness = 3          # 检测框线条厚度
detection_box_corner_radius = 5 # 检测框的圆角半径

# 标签框相关参数
label_box_min_height = 45  # 标签框最小高度
label_box_corner_radius = 8 # 标签框上部圆角的半径
label_horizontal_padding = 10 # 标签文字左右内边距
label_vertical_padding = 5    # 标签文字上下内边距

# OpenCV内置字体参数 (用于英文或use_chinese_font=False时)
font_face_cv2 = cv2.FONT_HERSHEY_SIMPLEX # 最常用的OpenCV字体
font_scale_cv2 = 1.0 # 字体大小缩放
text_thickness_cv2 = 2 # 文字线条厚度

# Pillow字体参数 (用于中文或use_chinese_font=True时)
# 字体文件路径（请替换为您的实际路径）
# Windows: "C:/Windows/Fonts/msyh.ttc" (微软雅黑), "C:/Windows/Fonts/simhei.ttf" (黑体)
# macOS: "/System/Library/Fonts/Arial Unicode.ttf" (支持中文), "/System/Library/Fonts/PingFang.ttc" (苹方)
# Linux: "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" (英文), "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc" (文泉驿等)
font_path_pil = "LXGWWenKai-Bold.ttf" # 例如：微软雅黑
font_size_pil = 20 # Pillow字体大小

# 控制是否使用Pillow绘制中文 (重要开关)
# 如果 current_label_text 包含中文，设置为 True，否则设置为 False
use_chinese_font = any('\u4e00' <= char <= '\u9fff' for char in current_label_text)


# 4. 绘制检测框
draw_detection_box(image_np, detection_bbox, box_line_color_bgr, box_thickness, detection_box_corner_radius)

# 5. 绘制标签框和文字
draw_label_box(image_np, detection_bbox, current_label_text, label_fill_color_bgr, text_color_bgr,
               font_face_cv2, font_scale_cv2, text_thickness_cv2, # OpenCV 字体参数
               font_path_pil, font_size_pil, # Pillow 字体参数
               label_box_min_height, label_box_corner_radius,
               label_horizontal_padding, label_vertical_padding, # 内边距
               use_chinese_font, box_thickness) # 传递厚度参数


# 6. 显示图像
cv2.imshow("YOLO Style Detection with Label Box (Final Version)", image_np)
cv2.waitKey(0) # 等待用户按键
cv2.destroyAllWindows() # 关闭所有OpenCV窗口