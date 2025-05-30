# scripts/yolo_train.py
from ultralytics import YOLO
from pathlib import Path
from utils.paths import (CONFIGS_DIR, PRETRAINED_MODELS_DIR,YOLO_SERVICE_DIR)
from utils.utils import load_yaml_config
# from utils import log_execution_time
import argparse
import logging
from utils.utils import setup_logging, merge_configs, log_training_params, copy_checkpoint_models 
from utils.utils import rename_log_file 



def parse_args():
    parser = argparse.ArgumentParser(description="基于YOLO的共享单车检测训练")
    # YOLO 核心参数
    parser.add_argument("--data", type=str, default="./data/data.yaml", help="数据集路径")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--batch", type=int, default=64, help="批次大小")
    parser.add_argument("--device", type=str, default=None, help="计算设备")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--project", type=str, default=None, help="输出目录")
    parser.add_argument("--name", type=str, default=None, help="运行名称")
    parser.add_argument("--exist_ok", action="store_true", help="是否覆盖现有目录")
    parser.add_argument("--model", type=str, default="./configs/yolov8.yaml", help="自定义模型配置文件")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="预训练模型")
    # 可选高频 YOLO 参数
    parser.add_argument("--lr0", type=float, default=None, help="初始学习率")
    parser.add_argument("--optimizer", type=str, default=None, help="优化器类型 (如 SGD, Adam)")
    # 项目自定义参数
    parser.add_argument("--log_encoding", type=str, default="utf-8-sig", help="日志编码格式")
    parser.add_argument("--use_yaml", type=bool, default=True, help="是否使用 YAML 配置")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别")
    return parser.parse_args()

# @log_execution_time(logger_name="YOLO_Training")
def train_model(model, yolo_args, logger):
    """
    执行 YOLO 模型训练。

    Args:
        model (YOLO): YOLO 模型实例。
        yolo_args (argparse.Namespace): YOLO 训练参数。
        logger (logging.Logger): 日志记录器。

    Returns:
        results: 训练结果。
    """
    results = model.train(**vars(yolo_args))
    return results

def main():
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    model_name = args.model.replace('./configs/', '')  
    model_name = model_name.replace('.yaml', '')  # 去掉文件扩展名
    logger = setup_logging(
         base_path=YOLO_SERVICE_DIR,
         log_type="train",
         model_name=model_name,
         encoding=args.log_encoding,
         log_level=log_level
     )
    logger.info("===== YOLOv8 训练开始 =====")
    try:
        yaml_config = {}
        if args.use_yaml:
            yaml_config = load_yaml_config(config_path=CONFIGS_DIR / "train.yaml", config_type="train")
        yolo_args, project_args = merge_configs(args, yaml_config)

        model_pretrained_path = PRETRAINED_MODELS_DIR / project_args.weights
        log_training_params(project_args, model_pretrained_path, CONFIGS_DIR, logger)

        # logger.info(f"加载模型: {model_pretrained_path}")
        if not model_pretrained_path.exists():
            logger.error(f"模型文件不存在: {model_pretrained_path}")
            raise FileNotFoundError(f"模型文件不存在: {model_pretrained_path}")
        # model = YOLO(str(model_pretrained_path))
        logger.info(f"加载模型: 配置 {args.model}； 权重 {model_pretrained_path}")
        model = YOLO(args.model).load(model_pretrained_path)  # 加载自定义模型配置和预训练权重
        if not model:
            logger.error(f"模型加载失败: {args.model}")
        logger.info(f"模型加载成功: 配置 {args.model}； 权重 {model_pretrained_path}")



        if not Path(project_args.data).exists():
            logger.error(f"数据集配置文件不存在: {project_args.data}")
            raise FileNotFoundError(f"数据集配置文件不存在: {project_args.data}")
        else:
            logger.info(f"数据集配置文件存在: {project_args.data}")

        # device = yolo_args.device if hasattr(yolo_args, 'device') else ("0" if torch.cuda.is_available() else "cpu")
        # logger.info(f"使用设备: {device}")

        # logger.info(f"开始训练: epochs={yolo_args.epochs}, imgsz={yolo_args.imgsz}, batch={yolo_args.batch}")
        # 核心训练
        # results = model.train(**vars(yolo_args))
        # 调用封装的训练函数
        results = train_model(model, yolo_args, logger)

        if results and hasattr(results, 'save_dir'):
            copy_checkpoint_models(results.save_dir, project_args.weights, logger)
            rename_log_file(logger, results.save_dir, model_name, args.log_encoding)
        else:
            logger.warning("训练未生成有效的目录,跳过模型复制")
        logger.info("训练完成")
        logger.info(f"模型保存至: {results.save_dir}")

        metrics = results.results_dict
        log_lines = [
            f"训练结果:",
            f"- 精确率: {metrics['metrics/precision(B)']:.4f}",
            f"- 召回率: {metrics['metrics/recall(B)']:.4f}",
            f"- mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}",
            f"- mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}"
        ]
        logger.info("\n".join(log_lines))
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        raise
    finally:
        logger.info("===== YOLOv8 训练结束 =====")


if __name__ == "__main__":
    main()