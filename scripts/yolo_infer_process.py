#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_infer_process.py
# @Time      :2025/5/27
# @Author    :雨霓同学
# @Function  :基于 YOLO 的安全帽检测推理脚本（融合多线程，修复文件夹和视频输出）

import logging
from ultralytics import YOLO
from pathlib import Path
import argparse
import cv2
import numpy as np
import os
import queue
import threading
import time
from utils.utils import (setup_logging, load_yaml_config, merge_configs, log_training_params, rename_log_file,
                  YOLO_SERVICE_DIR, CONFIGS_DIR)
from utils.infer_fream import process_frame
from utils.infer_stream import stream_inference

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
    parser.add_argument("--save_conf", type=bool, default=True, help="在 TXT 中包含置信度值")
    parser.add_argument("--save_frame", type=bool, default=True, help="保存摄像头/视频每帧图像")
    parser.add_argument("--save_crop", type=bool, default=True, help="保存检测框裁剪图像")
    parser.add_argument("--display-size", type=str, default="720p", choices=["360p", "720p", "1280p", "2K", "4K"],
                        help="摄像头/视频显示分辨率")
    parser.add_argument("--beautify", type=bool, default=True, help="启用美化绘制（圆角标签、中文支持）")
    parser.add_argument("--use-chinese", type=bool, default=True, help="启用中文映射")
    parser.add_argument("--font-size", type=int, default=22, help="美化字体大小（以 720p 为基准，自动缩放）")
    parser.add_argument("--line-width", type=int, default=4, help="美化线宽（以 720p 为基准，自动缩放）")
    parser.add_argument("--label-padding-x", type=int, default=10, help="美化标签水平内边距（以 720p 为基准，自动缩放）")
    parser.add_argument("--label-padding-y", type=int, default=10, help="美化标签垂直内边距（以 720p 为基准，自动缩放）")
    parser.add_argument("--radius", type=int, default=8, help="美化圆角半径（以 720p 为基准，自动缩放）")
    parser.add_argument("--log_encoding", type=str, default="utf-8-sig", help="日志编码格式")
    parser.add_argument("--use_yaml", type=bool, default=True, help="是否使用 YAML 配置")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别")
    parser.add_argument('--extra_args', nargs='*', default=[], help="额外 YOLO 参数，格式: --key value")
    return parser.parse_args()

class VideoProcessor:
    def __init__(self, model, args, yolo_args, project_args, logger):
        self.model = model
        self.args = args
        self.yolo_args = yolo_args
        self.project_args = project_args
        self.logger = logger

        # 初始化视频捕获
        self.cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
        if not self.cap.isOpened():
            self.logger.error(f"无法打开{'摄像头' if args.source.isdigit() else '视频'}: {args.source}")
            raise RuntimeError(f"无法打开{'摄像头' if args.source.isdigit() else '视频'}: {args.source}")

        # 设置摄像头参数
        self.target_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.resolution_map = {
            "360p": (640, 360), "720p": (1280, 720), "1280p": (1920, 1080),
            "2K": (2560, 1440), "4K": (3840, 2160)
        }
        self.target_width, self.target_height = self.resolution_map[project_args.display_size]
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        # 视频写入器
        self.save_dir = None
        self.video_writer = None
        self.frames_dir = None
        self.crops_dir = None
        self.frame_count = 0
        self.preprocess_times = []
        self.inference_times = []
        self.postprocess_times = []

        # 多线程通信
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.frame_interval = 1.0 / self.target_fps
        self.last_frame_time = time.time()

        # 美化参数
        self.beautify_params = {
            "use_chinese_mapping": project_args.use_chinese,
            "beautify": project_args.beautify,
            "font_size": project_args.font_size,
            "line_width": project_args.line_width,
            "label_padding": (project_args.label_padding_x, project_args.label_padding_y),
            "radius": project_args.radius,
            "text_color": (0, 0, 0)
        }

    def capture_and_process(self):
        """采集和处理线程"""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("无法读取摄像头帧")
                time.sleep(0.001)
                continue

            # YOLO推理
            start_time = time.time()
            yolo_args_dict = vars(self.yolo_args).copy()
            yolo_args_dict['stream'] = False
            yolo_args_dict['verbose'] = False
            yolo_args_dict['save'] = False  # 禁用 YOLO 默认保存
            yolo_args_dict['save_txt'] = False
            yolo_args_dict['save_crop'] = False
            yolo_args_dict.pop('source', None)
            results = self.model(frame, **yolo_args_dict)
            inference_time = time.time() - start_time

            # 处理帧
            annotated_frame = process_frame(frame, results[0], self.project_args,
                                         self.target_width, self.target_height, self.beautify_params)
            resized = cv2.resize(annotated_frame, (self.target_width, self.target_height))

            # 记录时间
            self.preprocess_times.append(results[0].speed['preprocess'])
            self.inference_times.append(results[0].speed['inference'])
            self.postprocess_times.append(results[0].speed['postprocess'])

            # 保存帧图像
            if self.frames_dir and self.project_args.save_frame:
                frame_path = self.frames_dir / f"{self.frame_count}.jpg"
                cv2.imwrite(str(frame_path), resized)
                self.logger.debug(f"保存帧图像: {frame_path}")

            # 保存裁剪图像（手动实现）
            if self.crops_dir and self.project_args.save_crop:
                for i, box in enumerate(results[0].boxes):
                    cls = int(box.cls)
                    conf = box.conf.item()
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    crop = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    crop_path = self.crops_dir / f"{self.frame_count}_{i}_cls{cls}_conf{conf:.2f}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    self.logger.debug(f"保存裁剪图像: {crop_path}")

            # 保存 TXT
            if self.save_dir and self.project_args.save_txt:
                txt_path = self.save_dir / f"{self.frame_count}.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    for box in results[0].boxes:
                        cls = int(box.cls)
                        conf = box.conf.item() if self.project_args.save_conf else None
                        xywh = box.xywh[0].cpu().numpy()
                        line = f"{cls} {xywh[0]:.1f} {xywh[1]:.1f} {xywh[2]:.1f} {xywh[3]:.1f}"
                        if conf is not None:
                            line += f" {conf:.2f}"
                        f.write(line + '\n')
                    self.logger.debug(f"保存 TXT: {txt_path}")

            # 写入队列
            try:
                self.frame_queue.put_nowait((resized, time.time()))
            except queue.Full:
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                try:
                    self.frame_queue.put_nowait((resized, time.time()))
                except queue.Full:
                    pass

            self.frame_count += 1

            # 帧率控制
            now = time.time()
            elapsed = now - self.last_frame_time
            sleep_time = max(0, self.frame_interval - elapsed - 0.001)
            time.sleep(sleep_time)
            self.last_frame_time = now

    def run(self):
        """主线程：显示和保存视频"""
        worker = threading.Thread(target=self.capture_and_process)
        worker.daemon = True
        worker.start()

        self.logger.info("等待首帧...")
        start_time = time.time()
        while self.frame_queue.empty() and not self.stop_event.is_set():
            time.sleep(0.01)
            if time.time() - start_time > 5:
                self.logger.error("无法获取首帧，退出")
                self.stop_event.set()
                break

        # 初始化保存路径
        if not self.stop_event.is_set():
            self.save_dir = Path(YOLO_SERVICE_DIR) / "runs" / "detect" / f"infer_{int(time.time())}"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"推理结果保存目录: {self.save_dir}")
            if self.project_args.save_frame:
                self.frames_dir = self.save_dir / "0_frames"
                self.frames_dir.mkdir(parents=True, exist_ok=True)
            if self.project_args.save_crop:
                self.crops_dir = self.save_dir / "crops"
                self.crops_dir.mkdir(parents=True, exist_ok=True)
            if self.project_args.save:
                video_path = self.save_dir / "output.mp4"
                self.video_writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*'H264'),  # 尝试 H264 编码器
                    self.target_fps,
                    (self.target_width, self.target_height)
                )
                if not self.video_writer.isOpened():
                    self.logger.error(f"视频写入器初始化失败: {video_path}")
                    self.stop_event.set()

        window_name = "YOLO Safety Helmet Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.target_width, self.target_height)

        frame_count = 0
        start_time = time.time()
        try:
            while not self.stop_event.is_set():
                latest_frame = None
                while not self.frame_queue.empty():
                    latest_frame, _ = self.frame_queue.get_nowait()

                if latest_frame is not None:
                    if self.video_writer and self.project_args.save and self.video_writer.isOpened():
                        self.video_writer.write(latest_frame)
                        self.logger.debug(f"写入视频帧: {frame_count}")
                    cv2.imshow(window_name, latest_frame)
                    frame_count += 1
                else:
                    self.logger.debug("无新帧，跳过显示")

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    self.logger.info(f"退出键: {'q' if key == ord('q') else 'ESC'}")
                    self.stop_event.set()
                    break

        except Exception as e:
            self.logger.error(f"主循环异常: {e}")
            self.stop_event.set()

        finally:
            self.stop_event.set()
            worker.join(timeout=0.5)
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            if self.cap.isOpened():
                self.cap.release()
            if self.video_writer and self.video_writer.isOpened():
                self.video_writer.release()
            cv2.destroyAllWindows()

            # 计算并记录时间
            total_preprocess = sum(self.preprocess_times)
            total_inference = sum(self.inference_times)
            total_postprocess = sum(self.postprocess_times)
            total_time = total_preprocess + total_inference + total_postprocess
            avg_preprocess = total_preprocess / frame_count if frame_count > 0 else 0
            avg_inference = total_inference / frame_count if frame_count > 0 else 0
            avg_postprocess = total_postprocess / frame_count if frame_count > 0 else 0
            avg_total = total_time / frame_count if frame_count > 0 else 0

            self.logger.info(
                f"视频/摄像头推理总时间: 总帧数={frame_count}, "
                f"总预处理={total_preprocess:.2f}ms, "
                f"总推理={total_inference:.2f}ms, "
                f"总后处理={total_postprocess:.2f}ms, "
                f"总耗时={total_time:.2f}ms (约 {total_time / 1000:.2f}秒)"
            )
            self.logger.info(
                f"单帧平均时间: "
                f"预处理={avg_preprocess:.2f}ms, "
                f"推理={avg_inference:.2f}ms, "
                f"后处理={avg_postprocess:.2f}ms, "
                f"总计={avg_total:.2f}ms"
            )
            self.logger.info(f"{'摄像头' if self.args.source.isdigit() else '视频'}推理完成，结果已保存至: {self.save_dir or '未保存'}")
            if self.save_dir:
                rename_log_file(self.logger, self.save_dir, Path(self.args.weights).stem, self.project_args.log_encoding)

def main():
    """主函数，执行 YOLO 模型推理"""
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    model_name = Path(args.weights).stem

    # 配置日志
    logger = setup_logging(
        base_path=YOLO_SERVICE_DIR,
        log_type="infer",
        model_name=model_name,
        encoding=args.log_encoding,
        log_level=log_level,
        temp_log=True
    )
    logger.info("===== YOLOv8 安全帽检测推理开始 =====")

    try:
        # 加载 YAML 配置
        yaml_config = {}
        if args.use_yaml:
            yaml_config = load_yaml_config(config_path=CONFIGS_DIR / "infer.yaml", config_type="infer")

        # 合并参数
        yolo_args, project_args = merge_configs(args, yaml_config, mode="infer")
        log_training_params(project_args, Path(args.weights), CONFIGS_DIR, logger, mode="infer")

        # 验证路径
        model_path = Path(project_args.weights)
        if not model_path.is_absolute():
            model_path = YOLO_SERVICE_DIR / "models" / "checkpoints" / project_args.weights
        source = project_args.source
        if not source.isdigit():
            source_path = Path(source)
            if not source_path.exists():
                logger.error(f"输入源不存在: {source_path}")
                raise FileNotFoundError(f"输入源不存在: {source_path}")
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            raise FileNotFoundError(f"模型文件不存在: {project_args.weights}")

        # 加载模型
        logger.info(f"加载模型: {model_path}")
        model = YOLO(str(model_path))

        # 流式推理（摄像头或视频）
        if source.isdigit() or source.endswith((".mp4", ".avi", ".mov")):
            processor = VideoProcessor(model, project_args, yolo_args, project_args, logger)
            processor.run()
        else:
            # 非流式推理（图像/文件夹）
            yolo_args_dict = vars(yolo_args).copy()
            yolo_args_dict.pop('source', None)
            yolo_args_dict['save'] = False  # 禁用 YOLO 默认保存
            yolo_args_dict['save_txt'] = False
            yolo_args_dict['save_crop'] = False
            results = model.predict(**yolo_args_dict)
            save_dir = Path(results[0].save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"推理结果保存目录: {save_dir}")

            resolution_map = {
                "360p": (640, 360), "720p": (1280, 720), "1280p": (1920, 1080),
                "2K": (2560, 1440), "4K": (3840, 2160)
            }
            display_width, display_height = resolution_map[project_args.display_size]
            beautify_params = {
                "use_chinese_mapping": project_args.use_chinese,
                "beautify": project_args.beautify,
                "font_size": project_args.font_size,
                "line_width": project_args.line_width,
                "label_padding": (project_args.label_padding_x, project_args.label_padding_y),
                "radius": project_args.radius,
                "text_color": (0, 0, 0)
            }

            window_name = "YOLO Safety Helmet Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, display_width, display_height)

            for result in results:
                annotated_frame = process_frame(
                    result.orig_img, result, project_args, display_width, display_height, beautify_params
                )
                if project_args.save:
                    output_path = save_dir / Path(result.path).name
                    cv2.imwrite(str(output_path), annotated_frame)
                    logger.debug(f"保存图像: {output_path}")
                if project_args.save_txt:
                    txt_path = save_dir / f"{Path(result.path).stem}.txt"
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        for box in result.boxes:
                            cls = int(box.cls)
                            conf = box.conf.item() if project_args.save_conf else None
                            xywh = box.xywh[0].cpu().numpy()
                            line = f"{cls} {xywh[0]:.1f} {xywh[1]:.1f} {xywh[2]:.1f} {xywh[3]:.1f}"
                            if conf is not None:
                                line += f" {conf:.2f}"
                            f.write(line + '\n')
                    logger.debug(f"保存 TXT: {txt_path}")
                if project_args.save_crop:
                    crops_dir = save_dir / "crops"
                    crops_dir.mkdir(parents=True, exist_ok=True)
                    for i, box in enumerate(result.boxes):
                        cls = int(box.cls)
                        conf = box.conf.item()
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        crop = result.orig_img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                        crop_path = crops_dir / f"{Path(result.path).stem}_{i}_cls{cls}_conf{conf:.2f}.jpg"
                        cv2.imwrite(str(crop_path), crop)
                        logger.debug(f"保存裁剪图像: {crop_path}")
                cv2.imshow(window_name, annotated_frame)
                logger.info(
                    f"图像 {Path(result.path).name}: 预处理={result.speed['preprocess']:.2f}ms, "
                    f"推理={result.speed['inference']:.2f}ms, "
                    f"后处理={result.speed['postprocess']:.2f}ms"
                )
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q') or key == 27:
                    logger.info("用户按 q 或 Esc 退出推理")
                    break
            cv2.destroyAllWindows()
            logger.info(f"推理完成，结果已保存至: {save_dir}")
            rename_log_file(logger, save_dir, model_name, project_args.log_encoding)

    except Exception as e:
        logger.error(f"推理失败: {e}", exc_info=True)
        raise
    finally:
        logger.info("===== YOLOv8 安全帽检测推理结束 =====")

if __name__ == "__main__":
    main()