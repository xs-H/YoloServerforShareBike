import cv2
import numpy as np
from pathlib import Path
import time
from utils import custom_plot

def process_frame(frame, result, args, display_width, display_height, beautify_params):
    """处理单帧：动态调整美化参数、绘制、显示、保存"""
    # 绘制帧
    if args.beautify:
        annotated_frame = custom_plot(
            frame,
            boxes=result.boxes.xyxy.cpu().numpy(),
            confs=result.boxes.conf.cpu().numpy(),
            labels=[result.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()],
            **beautify_params
        )
    else:
        try:
            annotated_frame = result.plot(line_width=beautify_params["line_width"])
        except Exception as e:
            print(f"Warning: YOLO line_width may not be supported: {e}")
            annotated_frame = result.plot()

    # 缩放显示
    annotated_frame = cv2.resize(annotated_frame, (display_width, display_height))
    return annotated_frame