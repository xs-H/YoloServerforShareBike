#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :utils.py
# @Time      :2025/5/22 09:31
# @Author    :雨霓同学
# @Function  :
import os
import shutil
import argparse
import torch
import logging
from datetime import datetime
import platform
from pathlib import Path
from .paths import CONFIGS_DIR, CHECKPOINTS_DIR,RUNS_DIR, PRETRAINED_MODELS_DIR,YOLO_SERVICE_DIR  # 相对导入
from .configs import DEFAULT_TRAIN_CONFIG, COMMENTED_TRAIN_CONFIG
from .dataset import get_dataset_info
from functools import wraps
import time

VALID_YOLO_ARGS = set(DEFAULT_TRAIN_CONFIG.keys())

import yaml
logger = logging.getLogger(__name__)
def setup_logging(base_path, log_type="testdataset", model_name=None, encoding='utf-8-sig', log_level=logging.INFO):
    """
    配置日志，保存到 logging/{log_type} 子目录。

    Args:
        base_path (Path): 项目根目录
        log_type (str): 日志类型，如 train、infer
        model_name (str, optional): 模型名称，用于日志文件名
        encoding (str): 日志文件编码，默认为 utf-8-sig
        log_level (int): 日志级别，默认为 logging.INFO

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    log_dir = base_path / "logging" / log_type
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_type}_{timestamp}_{model_name}.log" if model_name else f"{log_type}_{timestamp}.log"
    log_file = log_dir / log_filename
    logger = logging.getLogger("YOLO_Training")
    logger.setLevel(log_level)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding=encoding)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    # 记录日志元数据
    logger.info(f"日志目录: {log_dir}")
    logger.info(f"日志文件: {log_file} (编码: {encoding})")
    logger.info(f"日志类型: {log_type}")
    logger.info(f"日志级别: {logging.getLevelName(log_level)}")
    logger.info("日志初始化完成")

    return logger

# 获得详细的训练设备信息
def get_device_info(device):
    """
    获取详细的训练设备信息，包括GPU、CPU、内存等。
    Args: device(str):设备标识信息如 0 （GOU）或者CPU
    Returns:
        str: 包含训练设备信息的字符串列表。
    """
    device_info = []
    device_info.append(f"系统的基本信息如下：{platform.platform()}")
    device_info.append(f"操作系统版本信息及详细信息： {platform.system()} & {platform.version()}")
    device_info.append(f"CPU核心数： {platform.processor()}")
    device_info.append(f"内存信息： {platform.machine()}")
    device_info.append(f"Python版本： {platform.python_version()}")
    device_info.append(f"架构类型：{platform.machine()}")
    if device.upper() != "CPU" and torch.cuda.is_available():
        try:
            device_idx = int(device) if device.isdigit() else 0
            device_info.append(f"Pytorch版本： {torch.__version__}")
            device_info.append(f"CUDA版本： {torch.version.cuda or '未知'} ") # type: ignore
            device_info.append(f"显卡是否可用：{'可用' if torch.cuda.is_available() else '不可用'}")
            device_info.append(f"GPU设备型号： {torch.cuda.get_device_name(device_idx)}")
            device_info.append(f"GPU设备数量： {torch.cuda.device_count()}")
            device_info.append(f"CUDA算力：{torch.cuda.get_device_capability()}")
            device_info.append(f"GPU显存信息： {torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3):.2f} GB")
        except Exception as e:
            device_info.append(f"获取设备信息失败：{e}")
            device_info.append(f"设备标识：{device}")
    else:
        device_info.append(f"设备标识：{device}")
        device_info.append(f"Pytorch版本： {torch.__version__}")
        device_info.append(f"CPU型号：  {platform.processor() or '未知'} ")
    return device_info

# 加载yaml配置文件
def load_yaml_config(config_path, config_type="train"):
    """
    加载YAML配置文件。

    Args:
        congig_path (str): 配置文件路径。
        config_type (str): 配置文件类型，如 train、infer。

    Returns:
        dict: 加载的YAML配置文件。
    """
    # 统一转换为 Path 对象
    if config_path is None:
        config_path = CONFIGS_DIR / f"{config_type}.yaml"
    elif isinstance(config_path, str):
        config_path = CONFIGS_DIR / config_path  # 相对路径，拼接 CONFIGS_DIR
    else:
        config_path = Path(config_path)  # 确保是 Path 对象

    logger = logging.getLogger("YOLO_Training")
    logger.info(f"尝试加载配置文件: {config_path}")
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path},即将生成默认配置文件")
        if config_type == "train":
            logger.info("生成默认训练配置文件")
            generate_default_train_yaml(config_path)
        elif config_type == "infer":
            generate_default_infer_yaml(config_path)
    try:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            config = yaml.safe_load(f) or {}
            logger.info(f"加载配置文件成功: {config_path}")
            return config
    except yaml.YAMLError as e:
        logger.error(f"加载配置文件失败: {config_path},原因: {e}")
        return {}

def generate_default_train_yaml(config_path):
    """
    生成默认的配置文件,包含所有Yolo的训练参数和注释
    :param config_path: 配置文件的路径,如: configs/train.yaml
    :return: None
    """
    logger = logging.getLogger("YOLO_Training")
    config = DEFAULT_TRAIN_CONFIG
    commented_config = COMMENTED_TRAIN_CONFIG

    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(config_path, "w", encoding="utf-8-sig") as f:
            f.write(commented_config)
            logger.info(f"生成默认配置文件成功: {config_path}")
    except Exception as e:
        logger.error(f"生成默认配置文件失败: {config_path},原因: {e}")



def generate_default_infer_yaml(config_path):
    pass

def copy_checkpoint_models(train_dir, model_filename, logger):
    """
    复制训练好的模型到指定的目录下
    :param train_dir: 训练模型的地址
    :param model_filename: 模型的名称
    :param logger:
    :return:
    """
    if not train_dir or not isinstance(train_dir, Path):
        logger.error("无效训练目录, 跳过模型复制")
        return
    # date_str = datetime.now().strftime("%Y%m%d%_%H%M")
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_filename.replace('.pt', '')
    train_suffix = train_dir.name
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    if not os.access(CHECKPOINTS_DIR, os.W_OK):
        logger.error(f"检查点目录: {CHECKPOINTS_DIR}不可写, 跳过模型复制")
        raise OSError
    for model_type in ['best','last']:
        src_pat = train_dir / "weights" / f"{model_type}.pt"
        if src_pat.exists():
            checkpoint_name = f"{date_str}-{model_name}-{train_suffix}-{model_type}.pt"
            checkpoint_path = CHECKPOINTS_DIR / checkpoint_name
            shutil.copy(src_pat, checkpoint_path)
            logger.info(f"{model_type}.pt 已复制到: {checkpoint_path}")
        else:
            logger.warning(f"{model_type}.pt 未找到: {src_pat}")


def merge_configs(args, yaml_config):
    """
    合并命令行参数和 YAML 配置文件参数，并分离 YOLO 专用参数和项目自定义参数。

    功能概述：
    - 合并命令行（CLI）和 YAML 配置文件参数，CLI 参数优先级高于 YAML。
    - 分离出仅用于 YOLO 训练的参数（yolo_args）和包含所有参数的项目参数（project_args）。
    - 标记每个参数的来源（CLI 或 YAML），便于日志记录和追溯。
    - 标准化路径参数（data、project、weights）为绝对路径，确保路径正确性。
    - 验证关键参数（如 epochs、imgsz）的合法性，防止无效输入。
    - 记录 CLI 覆盖 YAML 的参数冲突，便于调试。
    - 支持任意 CLI 参数（通过 argparse.REMAINDER），提高灵活性。

    参数说明：
    :param args: 命令行输入的参数对象 (argparse.Namespace)，由 parse_args() 生成，包含 CLI 输入的参数，可能包括 extra_args（未知参数）。
    :param yaml_config: 从 YAML 配置文件（如 configs/train.yaml）加载的字典，包含默认或补充参数。
    :return: 返回元组 (yolo_args, project_args)
        - yolo_args: Namespace 对象，仅包含 YOLO 训练有效的参数（如 epochs、imgsz、data），传递给 model.train。
        - project_args: Namespace 对象，包含所有参数（YOLO 参数、项目自定义参数如 log_encoding、特殊参数如 use_yaml）。

    处理逻辑：
    1. 初始化两个 Namespace 对象：yolo_args（YOLO 参数）和 project_args（所有参数）。
    2. 合并 CLI 参数（包括 extra_args），过滤 None 值，写入 yolo_args 和 project_args。
    3. 补充 YAML 参数（仅当 CLI 未指定时），写入 yolo_args 和 project_args。
    4. 标记参数来源（CLI 或 YAML），通过 _specified 后缀记录。
    5. 验证关键 YOLO 参数（epochs、imgsz 等）的合法性。
    6. 标准化路径参数（data、project、weights）为绝对路径。
    7. 记录 CLI 覆盖 YAML 的参数冲突。

    注意事项：
    - VALID_YOLO_ARGS 是一个预定义的集合（在 utils.py 中），包含 YOLOv8 支持的训练参数（如 epochs、imgsz、lr0）。
    - YOLO_SERVICE_DIR 和 PRETRAINED_MODELS_DIR 是项目根目录和预训练模型目录（Path 对象），用于路径标准化。
    - 日志记录器（logger）用于记录路径标准化、参数冲突和错误信息。
    - extra_args 是通过 argparse.REMAINDER 捕获的未知 CLI 参数（如 --lr0 0.01），需手动解析。
    """
    # 初始化日志记录器，用于记录路径标准化、参数冲突和错误
    logger = logging.getLogger("YOLO_Training")

    # 初始化两个 Namespace 对象：
    # - project_args: 存储所有参数（CLI + YAML + 项目自定义），用于日志记录和项目管理
    # - yolo_args: 仅存储 YOLO 训练有效的参数，传递给 model.train
    project_args = argparse.Namespace()
    yolo_args = argparse.Namespace()

    # --- 第一阶段：合并命令行参数 ---
    # 从 CLI 参数中过滤掉值为 None 的参数，只保留用户显式指定的参数
    # 例如：python yolo_train.py --epochs 2 不会包含 batch=None
    cmd_args = {k: v for k, v in vars(args).items() if k != 'extra_args' and v is not None}

    # 遍历常规 CLI 参数，写入 project_args 和 yolo_args
    for key, value in cmd_args.items():
        # 将参数写入 project_args（所有参数都记录）
        setattr(project_args, key, value)
        # 如果参数属于 VALID_YOLO_ARGS，则写入 yolo_args
        if key in VALID_YOLO_ARGS:
            setattr(yolo_args, key, value)

    # 处理额外的 CLI 参数（通过 argparse.REMAINDER 捕获，如 --lr0 0.01）
    # extra_args 格式为列表：['--lr0', '0.01', '--mosaic', '0.5']
    if hasattr(args, 'extra_args'):
        # 确保 extra_args 成对出现（参数名和值）
        if len(args.extra_args) % 2 != 0:
            logger.error("额外参数格式错误，必须成对出现（如 --lr0 0.01）")
            raise ValueError("额外参数格式错误，必须成对出现")
        for i in range(0, len(args.extra_args), 2):
            key = args.extra_args[i].lstrip('--')  # 去掉 -- 前缀
            value = args.extra_args[i + 1]
            # 尝试将值转换为适当类型（例如字符串 '0.01' 转为 float）
            try:
                # 如果值是数字，转换为 int 或 float
                if value.replace('.', '', 1).isdigit():
                    value = float(value) if '.' in value else int(value)
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'  # 转换为布尔值
            except ValueError:
                logger.warning(f"无法转换额外参数 {key} 的值 {value}，保持字符串类型")
            # 写入 project_args 和 yolo_args（如果适用）
            setattr(project_args, key, value)
            if key in VALID_YOLO_ARGS:
                setattr(yolo_args, key, value)
            # 标记为 CLI 参数
            setattr(project_args, f"{key}_specified", True)

    # --- 第二阶段：合并 YAML 配置参数（命令行参数优先） ---
    # 遍历 YAML 参数，仅当 CLI 未指定时才使用 YAML 值
    for key, value in yaml_config.items():
        if not hasattr(project_args, key):
            # 写入 project_args
            setattr(project_args, key, value)
            # 如果参数属于 VALID_YOLO_ARGS，写入 yolo_args
            if key in VALID_YOLO_ARGS:
                setattr(yolo_args, key, value)
        else:
            # 如果 CLI 覆盖了 YAML 参数，记录冲突
            if getattr(project_args, key) != value:
                logger.info(f"参数 {key} 被命令行覆盖: CLI={getattr(project_args, key)}, YAML={value}")

    # --- 第三阶段：标记参数来源 ---
    # 为 CLI 参数（包括 extra_args）标记来源为 True
    for key in cmd_args:
        setattr(project_args, f"{key}_specified", True)
    # 为 extra_args 参数标记来源
    if hasattr(args, 'extra_args'):
        for i in range(0, len(args.extra_args), 2):
            key = args.extra_args[i].lstrip('--')
            setattr(project_args, f"{key}_specified", True)
    # 为 YAML 参数标记来源为 False（仅当未被 CLI 覆盖）
    for key in yaml_config:
        if not hasattr(project_args, f"{key}_specified"):
            setattr(project_args, f"{key}_specified", False)

    # --- 第四阶段：验证关键参数 ---
    # 验证 YOLO 参数的合法性，确保传递给 model.train 的参数有效
    if hasattr(yolo_args, 'epochs') and (yolo_args.epochs <= 0):
        logger.error("训练轮数 (epochs) 必须为正整数")
        raise ValueError("训练轮数 (epochs) 必须为正整数")
    if hasattr(yolo_args, 'imgsz') and (yolo_args.imgsz <= 0 or yolo_args.imgsz % 32 != 0):
        logger.error("图像尺寸 (imgsz) 必须为正整数且为 32 的倍数")
        raise ValueError("图像尺寸 (imgsz) 必须为正整数且为 32 的倍数")
    if hasattr(yolo_args, 'batch') and yolo_args.batch is not None and yolo_args.batch <= 0:
        logger.error("批次大小 (batch) 必须为正整数")
        raise ValueError("批次大小 (batch) 必须为正整数")
    if hasattr(yolo_args, 'device') and yolo_args.device is not None:
        if yolo_args.device != 'cpu' and not yolo_args.device.isdigit():
            logger.error("计算设备 (device) 必须为 'cpu' 或 GPU 索引（如 '0'）")
            raise ValueError("计算设备 (device) 必须为 'cpu' 或 GPU 索引")

    # --- 第五阶段：标准化路径参数 ---
    # 标准化数据集路径（data），将相对路径转换为绝对路径
    # 例如：--data data.yaml -> C:\Users\Matri\Desktop\Safe\yoloserver\configs\data.yaml
    if hasattr(args, 'data') and args.data and not Path(args.data).is_absolute():
        # 假设 data.yaml 位于 yoloserver/configs/ 目录
        data_path = YOLO_SERVICE_DIR / "configs" / args.data
        # 更新原始 args（确保后续逻辑一致）
        args.data = str(data_path)
        # 同步更新 project_args 和 yolo_args
        setattr(project_args, 'data', str(data_path))
        if 'data' in VALID_YOLO_ARGS:
            setattr(yolo_args, 'data', str(data_path))
        # 记录标准化后的路径到日志，便于调试
        logger.info(f"标准化数据集路径: {args.data}")

    # --- 第六阶段：返回结果 ---
    # 返回两个 Namespace 对象：
    # - yolo_args: 仅包含 YOLO 训练参数，传递给 model.train
    # - project_args: 包含所有参数，用于日志记录和项目管理
    return yolo_args, project_args

def log_training_params(args, model_path, configs_dir, logger, modee="train"):
    """
    :param args:
    :param model_path:
    :param configs_dir:
    :param logger:
    :param modee:
    :return:
    """
    logger.info(f"=========此次 训练过程中 YOLO参数 =========")
    logger.info(f"模型文件: {model_path}")
    logger.info(f"配置文件目录: {args.data}")
    if modee == "train":
        for key, value in vars(args).items():
            if not key.endswith('_specified'):  # 跳过标记字段
                source = '命令行' if getattr(args, f'{key}_specified', False) else 'YAML'
                logger.info(f"- {key}: {value} (来源: {source})")
    logger.info(f"=========此次 训练过程中 设备信息 =========")
    for info in get_device_info(args.device):
        logger.info(info)
    # 数据集信息
    logger.info(f"=========此次 训练过程中 数据集信息 =========")
    nc, names, train_samples, sample_source = get_dataset_info(args.data)
    logger.info(f"数据集类别数: {nc}")
    logger.info(f"数据集类别名称: {", ".join(names) if names else '未知'}")
    logger.info(f"数据集样本数: {train_samples} 基于: {sample_source}")
    logger.info(f"==============================")

def log_execution_time(logger_name="YOLO_Training"):
    """
    装饰器：记录函数执行时间并写入日志。

    Args:
        logger_name (str): 日志记录器的名称，默认为 YOLO_Training。

    Returns:
        callable: 装饰器函数。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name)
            start_time = time.time()
            logger.info(f"开始执行: {func.__name__}")
            result = func(*args, **kwargs)
            total_time = time.time() - start_time
            logger.info(f"{func.__name__} 完成，总耗时: {total_time:.2f} 秒 (约 {total_time / 60:.2f} 分钟)")
            return result
        return wrapper
    return decorator

def rename_log_file(logger, save_dir, model_name, log_encoding):
    """
    重命名日志文件

    Args:
        logger (logging.Logger): 日志记录器实例。
        save_dir (Path): 日志保存目录。
        model_name (str): 模型名称。
        log_encoding (str): 日志文件编码。

    Returns:
    """
    if not save_dir.exists():
        logger.error(f"日志保存目录不存在: {save_dir}")
        return

    # 获取当前日志文件名
    log_file = next((f for f in save_dir.glob("*.log") if f.is_file()), None)
    if not log_file:
        logger.warning("未找到日志文件，无法重命名")
        return

    # 构建新的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_log_name = f"{timestamp}_{model_name}.log"
    new_log_path = save_dir / new_log_name

    # 重命名日志文件
    log_file.rename(new_log_path)
    logger.info(f"日志文件已重命名为: {new_log_path} (编码: {log_encoding})")



if __name__ == '__main__':
    device_info = get_device_info("0")
    # for info in device_info:
    #     print(info)
