#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :verify_dataset_log_v2.py
# @Time      :2025/5/21 12:40
# @Author    :雨霓同学
# @Function  :验证YOLO数据集配置及相关文件

from os.path import split
import yaml
from pathlib import Path
import logging
import random
import sys
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent))  # 添加 yoloserver 目录到 sys.path
from utils.utils import setup_logging


# 配置验证模式（取消注释以选择验证模式）
VERIFY_MODE = 'SAMPLE'  # 抽样验证，比例由 sample_ratio 控制
# VERIFY_MODE = 'FULL'  # 全量验证

# 抽样验证的参数
SAMPLE_RATIO = 0.1  # 抽样比例
MIN_SAMPLES = 10  # 最小抽样数量


# 配置日志
# def setup_logging(base_path, log_type="testdataset"):
#     log_dir = base_path / "logging" / log_type
#     log_dir.mkdir(parents=True, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_file = log_dir / f"{log_type}_{timestamp}.log"
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file, encoding='utf-8'),
#             logging.StreamHandler()
#         ]
#     )
#     return logging.getLogger(__name__)


def verify_dataset_config(yaml_path):
    """
    验证YOLO数据集配置,检查data.yaml和对应的图像、标签文件
    :param yaml_path: data.yaml的路径
    :return: 验证通过返回True,否则False
    """
    yaml_path = Path(yaml_path).resolve()
    logger.info(f"验证 data.yaml 的路径为: {yaml_path}")

    if not yaml_path.exists():
        logger.error(f"data.yaml 文件不存在: {yaml_path}")
        return False

    # 读取 YAML 配置
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"YAML 文件解析失败: {e}")
        return False

    class_names = config.get('names', [])
    nc = config.get('nc', 0)

    if len(class_names) != nc:
        logger.error(f"class_names 长度与 nc 不匹配: {len(class_names)} != {nc}")
        return False
    logger.info(f"class_names 长度与 nc 匹配，类别数量: {nc}, 类别为: {class_names}")

    # 验证数据集，支持 train, val, test (test 可选)
    splits = ['train', 'val', 'test'] if 'test' in config else ['train', 'val']
    for split in splits:
        split_path = (yaml_path.parent / config[split]).resolve()
        logger.info(f"验证 {split} 路径为: {split_path}")
        if not split_path.exists():
            logger.error(f"图像目录 {split_path} 路径不存在")
            return False

        # 获取图像文件
        img_paths = (
                list(split_path.glob('*.[jJ][pP][gG]')) +
                list(split_path.glob('*.[pP][nN][gG]')) +
                list(split_path.glob('*.[jJ][pP][eE][gG]')) +
                list(split_path.glob('*.[bB][mM][pP]'))
        )
        if not img_paths:
            logger.error(f"图像目录 {split_path} 下没有图像文件")
            return False
        logger.info(f"{split} 路径下图像数量为: {len(img_paths)}")

        # 动态抽样大小
        sample_size = max(MIN_SAMPLES, int(len(img_paths) * SAMPLE_RATIO))
        if VERIFY_MODE == 'FULL':
            logger.info(f"{split} 进行全量验证，检测数量为: {len(img_paths)}")
            sample_paths = img_paths
        else:
            logger.info(f"{split} 进行抽样验证，抽样数量为: {sample_size}")
            sample_paths = random.sample(img_paths, min(sample_size, len(img_paths)))

        # 验证图像和标签文件
        for img_path in sample_paths:
            img_path_resolve = img_path.resolve()
            logger.debug(f"验证图像文件: {img_path_resolve}")
            if not img_path_resolve.exists():
                logger.error(f"图像文件 {img_path_resolve} 不存在")
                return False

            label_path = split_path.parent / split / (img_path.stem + '.txt')
            logger.debug(f"验证标签文件: {label_path}")
            if not label_path.exists():
                logger.debug(f"标签文件 {label_path} 不存在（正常，空标签）")
                continue

            # 验证 YOLO 格式的标签文件内容
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
            if not lines:
                logger.debug(f"标签文件 {label_path} 为空（正常）")
                continue

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    logger.error(f"标签文件 {label_path} 内容格式错误: {line}")
                    return False
                try:
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= nc:
                        logger.error(f"标签文件 {label_path} 内容错误: 类别 ID {class_id} 超出 nc {nc} 范围")
                        return False
                    coords = [float(x) for x in parts[1:]]
                    if not all(0 <= x <= 1 for x in coords):
                        logger.error(f"标签文件 {label_path} 内容错误: 坐标 {coords} 超出范围 [0, 1]: {line}")
                        return False
                except ValueError:
                    logger.error(f"标签文件 {label_path} 包含无效值: {line}")
                    return False

    logger.info("数据集验证通过")
    return True


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parent.parent  # 项目根目录
    logger = setup_logging(base_path, log_type="testdataset")
    logger.info(f"项目根目录为: {base_path}")
    yaml_path = base_path / "configs" / "data.yaml"
    # 配置验证模式（取消注释以选择验证模式）
    # VERIFY_MODE = 'SAMPLE'  # 抽样验证，比例由 sample_ratio 控制
    VERIFY_MODE = 'FULL'  # 全量验证

    # 抽样验证的参数
    SAMPLE_RATIO = 0.1  # 抽样比例
    MIN_SAMPLES = 10  # 最小抽样数量

    if verify_dataset_config(yaml_path):
        logger.info("测试完成,顺利通过测试")
    else:
        logger.error("测试完成,数据存在异常")