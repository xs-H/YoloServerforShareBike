# -*- coding:utf-8 -*-
# yoloserver/utils/paths.py
# 定义项目路径常量，确保跨平台兼容和目录初始化

from pathlib import Path
import logging

# 获取日志记录器（避免循环导入，延迟调用 setup_logging）
logger = logging.getLogger("YOLO_Training")

# 项目根目录（yoloserver/）
# 基于 __file__ 动态解析，指向 SafeTest/yoloserver/
YOLO_SERVICE_DIR = Path(__file__).parent.parent


logger.info(f"初始化路径: YOLO_SERVICE_DIR = {YOLO_SERVICE_DIR}")

# 配置文件目录（yoloserver/configs/）
# 存放 train.yaml, data.yaml 等
CONFIGS_DIR = YOLO_SERVICE_DIR / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"创建目录: CONFIGS_DIR = {CONFIGS_DIR}")

# 预训练模型目录（yoloserver/models/pretrained/）
# 存放 yolov8n.pt 等模型
PRETRAINED_MODELS_DIR = YOLO_SERVICE_DIR / "models" / "pretrained"
PRETRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"创建目录: PRETRAINED_MODELS_DIR = {PRETRAINED_MODELS_DIR}")

# 检查点目录（yoloserver/runs/detect/）
# 存放训练检查点，如 runs/detect/train/
CHECKPOINTS_DIR = YOLO_SERVICE_DIR / "models" / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"创建目录: CHECKPOINTS_DIR = {CHECKPOINTS_DIR}")

# 训练输出目录（yoloserver/runs/）
# 存放所有训练结果，如 runs/detect/
RUNS_DIR = YOLO_SERVICE_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"创建目录: RUNS_DIR = {RUNS_DIR}")




