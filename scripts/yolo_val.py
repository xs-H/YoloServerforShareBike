#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_val_teacher.py
# @Time      :2025/5/26 09:11
# @Author    :雨霓同学
# @Function  :
from ultralytics import YOLO
from pathlib import Path
import argparse
import logging
from utils import (CONFIGS_DIR,
                CHECKPOINTS_DIR,
                YOLO_SERVICE_DIR,
                setup_logging,  # 设置日志
                merge_configs,  # 合并配置文件
                log_execution_time,  # 记录执行时间
                load_yaml_config,  # 加载默认配置文件
                log_training_params,  # 记录参数来源
                )


def parse_args():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser(description="基于 YOLO 的安全帽检测验证")
    parser.add_argument("--weights", type=str, default="train6-20250523-175204-yolo11m-best.pt", help="模型权重路径")
    parser.add_argument("--data", type=str, default="data.yaml", help="数据集路径")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--device", type=str, default="0", help="计算设备")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--iou",  type=float, default=0.45, help="IOU 阈值")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--split",type=str, default="test", help="数据集分割")
    parser.add_argument("--plots",  type=bool, default=True, help="绘制验证结果")
    parser.add_argument("--save_txt", type=bool, default=True, help="保存验证为TXT")
    parser.add_argument("--save_conf", type=bool, default=None, help="在 TXT 中包含置信度值")
    # 项目自定义参数
    parser.add_argument("--log_encoding", type=str, default="utf-8-sig", help="日志文件编码")
    parser.add_argument("--use_yaml", type=bool, default=True, help="是否使用 YAML 配置")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别（DEBUG/INFO/WARNING/ERROR）")
    # 支持额外的 YOLO 参数
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="额外的 YOLO 参数（如 --max_det 100）")
    return parser.parse_args()

@log_execution_time(logger_name="YOLO_Validation")
def validate_model(model, yolo_args, logger):
    """
    执行验证函数
    :param model: 模型实例
    :param yolo_args: YOLO的验证参数
    :param logger: 日志记录器
    :return: 验证的结果
    """
    results = model.val(**vars(yolo_args))
    return results

def main():
    """
    主函数： 执行Yolo模型验证并记录日志
    :return:
    """
    # 获取CIL的参数
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    model_name = Path(args.weights).stem  # 提取模型名（如 yolo11m）
    # 初始化日志记录器
    logger = setup_logging(
        base_path=YOLO_SERVICE_DIR,
        log_type="val",
        model_name=model_name,
        encoding=args.log_encoding,
        log_level=log_level,
        temp_log=False  # 验证无需临时日志
    )
    logger.info("===== YOLO 安全帽检测验证开始 =====")
    try:
        # 加载 YAML 配置
        yaml_config = {}
        if args.use_yaml:
            yaml_config = load_yaml_config(config_path=CONFIGS_DIR / "val.yaml", config_type="val")

        # 合并命令行和 YAML 参数
        yolo_args, project_args = merge_configs(args, yaml_config, mode='val')
        # print(yolo_args)  # 只有Yolo本身的参数
        # print(project_args)  # 包含所有的项目参数

        # 检查一下模型文件
        model_path = Path(project_args.weights)
        if not model_path.is_absolute():
            model_path = CHECKPOINTS_DIR / project_args.weights
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        logger.info(f"加载模型: {model_path}")

        # 检查数据集配置文件
        data_path = Path(project_args.data)
        if not data_path.is_absolute():
            data_path = CONFIGS_DIR / project_args.data
        if not data_path.exists():
            logger.error(f"数据集配置文件不存在: {data_path}")
            raise FileNotFoundError(f"数据集配置文件不存在: {data_path}")
        logger.info(f"数据集配置文件: {data_path}")

        # 记录参数、设备和数据集信息
        log_training_params(project_args, model_path, CONFIGS_DIR, logger,  mode="val")

        # 加载yolo模型
        model = YOLO(str(model_path))

        # 执行验证
        results = validate_model(model, yolo_args, logger)

        # 记录验证结果
        logger.info("========= 验证结果 =========")
        metrics = results.results_dict
        log_lines = [
            f"验证结果:",
            f"- 精确率: {metrics['metrics/precision(B)']:.4f}",
            f"- 召回率: {metrics['metrics/recall(B)']:.4f}",
            f"- mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}",
            f"- mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}"
        ]
        logger.info("\n".join(log_lines))


    except Exception as e:
        logger.error(f"验证失败: {e}", exc_info=True)
        raise
    finally:
        logger.info("===== YOLO 安全帽检测验证结束 =====")



if __name__ == "__main__":
    main()
