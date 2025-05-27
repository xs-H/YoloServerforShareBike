#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :dataset.py
# @Time      :2025/5/24
# @Author    :雨霓同学
# @Function  :获取 YOLO 数据集的基本信息
import yaml
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)


def get_dataset_info(data_config_path, mode='train'):
    """
    从 YOLO 格式的数据配置文件中获取数据集信息。

    Args:
        data_config_path (str or Path): 数据配置文件路径（如 data.yaml）。
        mode (str): 数据集模式，'train' 获取训练集信息，'val' 获取测试集信息，默认为 'train'。

    Returns:
        tuple: (类别数, 类别名称列表, 样本数量, 样本来源)。

    Raises:
        FileNotFoundError: 如果 data_config_path 不存在。
        yaml.YAMLError: 如果 YAML 文件解析失败。
    """
    data_config_path = Path(data_config_path)
    try:
        # 加载 YAML 配置文件
        with open(data_config_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f) or {}

        # 获取类别信息
        nc = data_config.get('nc', 0)
        names = data_config.get('names', [])

        # 选择目标路径（train 或 test）
        split_key = 'train' if mode == 'train' else 'test'
        if split_key not in data_config:
            if mode == 'val':
                logger.warning(f"'{split_key}' 路径未在 {data_config_path} 中定义，尝试使用 'val' 路径")
                split_key = 'val'
            if split_key not in data_config:
                logger.warning(f"'{split_key}' 路径未定义，返回空信息")
                return nc, names, 0, "未知"

        # 解析路径
        split_path = Path(data_config[split_key])
        if not split_path.is_absolute():
            split_path = data_config_path.parent / split_path

        # 统计样本数量
        samples = 0
        sample_source = "未知"
        images_dir = split_path
        if images_dir.exists():
            # 支持的图像格式
            image_extensions = ['.jpg', '.png', '.jpeg', '.bmp', '.webp']
            samples = sum(
                len([f for f in os.listdir(images_dir) if f.lower().endswith(tuple(image_extensions))]) for _ in [1])
            sample_source = f"图片文件 ({', '.join(image_extensions)})"
            if samples == 0:
                logger.warning(f"未找到图像文件，检查路径: {images_dir}")
        else:
            logger.warning(f"图像目录不存在: {images_dir}")

        logger.info(f"{mode} 数据集信息：类别数={nc}, 样本数={samples}, 来源={sample_source}")
        return nc, names, samples, sample_source

    except FileNotFoundError:
        logger.error(f"数据集配置文件不存在: {data_config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"解析 YAML 文件失败: {data_config_path}, 错误: {e}")
        raise
    except Exception as e:
        logger.error(f"获取数据集信息失败: {e}")
        return 0, [], 0, "未知"


if __name__ == "__main__":
    print(get_dataset_info("../configs/data.yaml", mode='train'))
    print(get_dataset_info("../configs/data.yaml", mode='val'))
    print(get_dataset_info("../configs/data.yaml", mode='test'))