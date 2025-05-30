# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_inference_teaV0.py
# @Time      :2025/5/26 11:10
# @Author    :雨霓同学
# @Function  : 简单版本-支持原生推理

from ultralytics import YOLO
from pathlib import Path
import argparse
import cv2
from utils.utils import (setup_logging, load_yaml_config, merge_configs,
                log_training_params,rename_log_file, YOLO_SERVICE_DIR, CONFIGS_DIR)
from utils.infer_fream import process_frame
import logging
def args_parser():
    """
    命令行解析参数
    :return: 解析后的命名行参数
    """
    parser = argparse.ArgumentParser(description="基于 YOLO 的安全帽推理脚本")
    parser.add_argument("--weights", type=str, default="train6-20250523-175204-yolo11m-best.pt", help="模型权重路径")
    parser.add_argument("--source", type=str, default=r"..\data\test\images", help="输入源（图像/文件夹/视频/摄像头ID，如 '0'）")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU 阈值")
    parser.add_argument("--save", type=bool, default=True, help="保存预测图像")
    parser.add_argument("--save_dir", type=str, default="runs/val", help="保存预测图像的目录")
    parser.add_argument("--save_txt", type=bool, default=True, help="保存预测结果为 TXT")
    parser.add_argument("--save_conf", type=bool, default=True, help="在 TXT 中包含置信度值")
    parser.add_argument("--save_frames", type=bool, default=True, help="保存摄像头/视频每帧图像")
    parser.add_argument("--save_crop", type=bool, default=True, help="保存检测框裁剪图像")
    parser.add_argument("--show",  type=bool, default=True, help="显示结果")
    # 自定义参数
    parser.add_argument("--display-size",  type=str, default="720p",
                        choices=["360p", "720p", "1280p", "2K", "4K"], help="显示窗口大小")
    parser.add_argument("--beautify", type=bool, default=True, help="美化结果")
    parser.add_argument("--use-chinese", type=bool, default=True, help="使用中文映射")
    parser.add_argument("--font-size", type=int, default=20, help="美化字体大小（以 720p 为基准，自动缩放）")
    parser.add_argument("--line-width", type=int, default=4, help="美化线宽（以 720p 为基准，自动缩放）")
    parser.add_argument("--label-padding-x", type=int, default=30, help="美化标签水平内边距（以 720p 为基准，自动缩放）")
    parser.add_argument("--label-padding-y", type=int, default=18, help="美化标签垂直内边距（以 720p 为基准，自动缩放）")
    parser.add_argument("--radius", type=int, default=8, help="美化圆角半径（以 720p 为基准，自动缩放）")
    parser.add_argument("--text-color", type=str, default="black", help="美化文本颜色")
    parser.add_argument("--log-encoding", type=str, default="utf-8-sig", help="日志编码格式")
    parser.add_argument("--use_yaml", type=bool, default=True, help="是否使用 YAML 配置")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别")
    return parser.parse_args()


def main():
    """
    主函数，执行推理
    :return:
    """
    args = args_parser()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    model_name = Path(args.weights).stem
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
        yaml_config = {}
        if args.use_yaml:
            yaml_config = load_yaml_config(config_path=CONFIGS_DIR / "infer.yaml", config_type="infer")
        #  合并参数
        yolo_args, project_args = merge_configs(args, yaml_config, mode="infer")

        # 分辨率的映射
        resolution_map = {
            "360p": (640, 360),
            "720p": (1280, 720),
            "1280p": (1920, 1080),
            "2K": (2560, 1440),
            "4K": (3840, 2160)
        }
        display_width, display_height = resolution_map[args.display_size]
        # 整理美化参数
        beautify_params = {
            "use_chinese_mapping": project_args.use_chinese,
            "beautify": project_args.beautify,
            "font_size": project_args.font_size,
            "line_width": project_args.line_width,
            "label_padding": (project_args.label_padding_x,project_args.label_padding_y),
            "radius": project_args.radius,
            "text_color": (0, 0, 0)
        }

        # 路径标准化
        model_path = Path(project_args.weights)
        if not model_path.is_absolute():
            model_path = YOLO_SERVICE_DIR / "models" / "checkpoints" / project_args.weights
        source = project_args.source

        if not source.isdigit():
            source_path = Path(source)
            if not source_path.exists():
                logger.error(f"输入源不存在: {source_path}")
                raise FileNotFoundError(f"输入源不存在: {source_path}")
            source = str(source_path)
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            raise FileNotFoundError(f"模型文件不存在: {project_args.weights}")

        # 加载模型
        logger.info(f"加载模型: {model_path}")
        model = YOLO(str(model_path))

        # 设置显示窗口
        window_name = "YOLO Safety Helmet Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if source.isdigit() or source.endswith((".mp4", "*.avi", '*.mov')):
            cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
            if not cap.isOpened():
                logger.error(f"无法打开{'摄像头' if source.isdigit() else '视频'}: {source}")
                raise RuntimeError(f"无法打开{'摄像头' if source.isdigit() else '视频'}: {source}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            # 初始化时间收集列表
            preprocess_times = []
            inference_times = []
            postprocess_times = []
            frame_count = 0

            # 流式推理
            video_writer = None
            frames_dir = None
            save_dir = None

            # 参数设置：复制yolo_args参数，修改为流模式
            yolo_args_dict = vars(yolo_args).copy()
            yolo_args_dict['stream'] = True
            yolo_args_dict['show'] = False
            yolo_args_dict['save'] = False
            for idx, result in enumerate(model.predict(**yolo_args_dict)):
                # 第一帧初始化保存路径
                if idx == 0:
                    save_dir = Path(result.save_dir)
                    logger.info(f"推理结果保存目录: {save_dir}")
                    if project_args.save_frames:
                        frames_dir = save_dir / "0_frames"
                        frames_dir.mkdir(parents=True, exist_ok=True)
                    if project_args.save:
                        video_path = save_dir / "output.mp4"
                        video_writer = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (display_width, display_height)
                        )
                frame = result.orig_img
                annotated_frame = process_frame(frame, result, args, display_width, display_height, beautify_params)

                if video_writer:
                    video_writer.write(annotated_frame)
                if frames_dir:
                    frame_path = frames_dir / f"{idx}.jpg"
                    cv2.imwrite(str(frame_path), annotated_frame)
                    logger.debug(f"保存帧图像: {frame_path}")
                # 收集推理时间
                preprocess_times.append(result.speed['preprocess'])
                inference_times.append(result.speed['inference'])
                postprocess_times.append(result.speed['postprocess'])
                frame_count += 1
                # 显示
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q') or key == 27:
                    logger.info("用户按 q 或 Esc 退出推理")
                    break
            # 计算总时间和平均时间
            total_preprocess = sum(preprocess_times)
            total_inference = sum(inference_times)
            total_postprocess = sum(postprocess_times)
            total_time = total_preprocess + total_inference + total_postprocess
            avg_preprocess = total_preprocess / frame_count if frame_count > 0 else 0
            avg_inference = total_inference / frame_count if frame_count > 0 else 0
            avg_postprocess = total_postprocess / frame_count if frame_count > 0 else 0
            avg_total = total_time / frame_count if frame_count > 0 else 0

            # 日志记录时间
            logger.info(
                f"视频/摄像头推理总时间: 总帧数={frame_count}, "
                f"总预处理={total_preprocess:.2f}ms, "
                f"总推理={total_inference:.2f}ms, "
                f"总后处理={total_postprocess:.2f}ms, "
                f"总耗时={total_time:.2f}ms (约 {total_time / 1000:.2f}秒)"
            )
            logger.info(
                f"单帧平均时间: "
                f"预处理={avg_preprocess:.2f}ms, "
                f"推理={avg_inference:.2f}ms, "
                f"后处理={avg_postprocess:.2f}ms, "
                f"总计={avg_total:.2f}ms"
            )
            # 释放资源
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            logger.info(f"{'摄像头' if source.isdigit() else '视频'}推理完成，结果已保存至: {save_dir or '未保存'}")
            # 重命名文件
            if save_dir:
                rename_log_file(logger, save_dir, model_name, project_args.log_encoding)
        else:
            # 非流式推理（图像/文件夹）
            yolo_args_dict = vars(yolo_args).copy()
            yolo_args_dict.pop('stream', None)
            yolo_args_dict['show'] = False
            yolo_args_dict['save'] = False
            results = model.predict(**yolo_args_dict)
            save_dir = Path(results[0].save_dir)
            logger.info(f"推理结果保存目录: {save_dir}")
            for result in results:
                display_height,display_width = result.orig_shape[:2]
                annotated_frame = process_frame(
                    result.orig_img, result, project_args, display_width, display_height, beautify_params
                )
                if project_args.save:
                    output_path = save_dir / Path(result.path).name
                    cv2.imwrite(str(output_path), annotated_frame)
                    logger.debug(f"保存图像: {output_path}")
                # cv2.imshow(window_name, annotated_frame)
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
        logger.error(f"参数解析错误: {e}")
        raise
    finally:
        logger.info("===== YOLOv8 安全帽检测推理结束 =====")


if __name__ == "__main__":
    main()


