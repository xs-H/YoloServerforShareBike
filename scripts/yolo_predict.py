#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_predict.py
# @Time      :2025/5/24 14:50
# @Author    :雨霓同学
# @Function  :基于 YOLO 的安全帽检测推理脚本（极简版，支持摄像头/视频流、图像、文件夹、多保存参数、统一输出目录、动态美化参数）

from ultralytics import YOLO
from pathlib import Path
import argparse
import cv2
import numpy as np
import os
from utils.beautify import custom_plot
from utils.infer_fream import process_frame

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="基于 YOLO 的安全帽检测推理")
    parser.add_argument("--weights", type=str, default="train6-20250523-175204-yolo11m-best.pt", help="模型权重路径")
    parser.add_argument("--source", type=str, default="0", help="输入源（图像/文件夹/视频/摄像头ID，如 '0'）")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU 阈值")
    parser.add_argument("--save", type=bool, default=True, help="保存预测结果（图像或视频）")
    parser.add_argument("--save_txt", type=bool, default=False, help="保存预测结果为 TXT")
    parser.add_argument("--save_conf", type=bool, default=False, help="在 TXT 中包含置信度值")
    parser.add_argument("--save_frame", type=bool, default=False, help="保存摄像头/视频每帧图像")
    parser.add_argument("--save_crop", type=bool, default=False, help="保存检测框裁剪图像")
    parser.add_argument("--display-size", type=str, default="720p", choices=["360p", "720p", "1280p", "2K", "4K"], help="摄像头/视频显示分辨率")
    parser.add_argument("--beautify", type=bool, default=True, help="启用美化绘制（圆角标签、中文支持）")
    parser.add_argument("--font-size", type=int, default=22, help="美化字体大小（覆盖自动调整）")
    parser.add_argument("--line-width", type=int, default=4, help="美化线宽（覆盖自动调整）")
    parser.add_argument("--label-padding-x", type=int, default=10, help="美化标签水平内边距（覆盖自动调整）")
    parser.add_argument("--label-padding-y", type=int, default=8, help="美化标签垂直内边距（覆盖自动调整）")
    parser.add_argument("--radius", type=int, default=8, help="美化圆角半径（覆盖自动调整）")
    return parser.parse_args()


def main():
    """主函数，执行 YOLO 模型推理"""
    args = parse_args()

    # 分辨率映射
    resolution_map = {
        "360p": (640, 360),
        "720p": (1280, 720),
        "1280p": (1920, 1080),
        "2K": (2560, 1440),
        "4K": (3840, 2160)
    }
    display_width, display_height = resolution_map[args.display_size]

    # 美化参数（基值）
    beautify_params = {
        "use_chinese_mapping": True,
        "beautify": True,
        "font_size": 22,
        "line_width": 4,
        "label_padding": (30, 18),
        "radius": 8,
        "text_color": (0, 0, 0)
    }

    # 路径标准化
    model_path = Path(args.weights)
    if not model_path.is_absolute():
        model_path =  Path(r"C:\Users\Matri\Desktop\Safe\yoloserver\models\checkpoints")  / args.weights
    source = args.source

    # 验证路径（跳过摄像头ID）
    if not source.isdigit():
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"输入源不存在: {source_path}")
        source = str(source_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载模型
    model = YOLO(str(model_path))

    # 设置显示窗口
    window_name = "YOLO Safety Helmet Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)

    # 流式推理（摄像头或视频）
    if source.isdigit() or source.endswith((".mp4", ".avi", ".mov")):
        # 初始化视频捕获
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开{'摄像头' if source.isdigit() else '视频'}: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # 流式推理
        video_writer = None
        frames_dir = None
        save_dir = None
        for idx, result in enumerate(model.predict(
            source=source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            show=False,
            project="runs/predict",
            name="exp",
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=args.save_crop,
            stream=True
        )):
            # 第一帧初始化保存路径
            if idx == 0:
                save_dir = Path(result.save_dir)
                if args.save_frame:
                    frames_dir = save_dir / "0_frames"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                if args.save:
                    video_path = save_dir / "output.mp4"
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (display_width, display_height)
                    )

            # 获取帧
            frame = result.orig_img

            # 处理帧
            annotated_frame = process_frame(frame, result, args, display_width, display_height, beautify_params)

            # 保存视频
            if video_writer:
                video_writer.write(annotated_frame)

            # 保存帧图像
            if frames_dir:
                cv2.imwrite(str(frames_dir / f"{idx}.jpg"), annotated_frame)

            # 显示
            cv2.imshow(window_name, annotated_frame)

            # 按 q 或 Esc 退出
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:
                break

        # 释放资源
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"{'摄像头' if source.isdigit() else '视频'}推理完成，结果已保存至: {save_dir or '未保存'}")
    else:
        # 非流式推理（图像/文件夹）
        results = model.predict(
            source=source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=args.save_crop,
            show=False,
            project="runs/predict",
            name="exp"
        )
        # 美化输出
        save_dir = Path(results[0].save_dir)
        for result in results:
            annotated_frame = process_frame(
                result.orig_img, result, args, display_width, display_height, beautify_params
            )
            if args.save:
                cv2.imwrite(str(save_dir / result.path.name), annotated_frame)
            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:
                break
        cv2.destroyAllWindows()
        print(f"推理完成，结果已保存至: {save_dir}")

if __name__ == "__main__":
    main()