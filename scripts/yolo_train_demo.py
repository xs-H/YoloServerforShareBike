#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_train_demo.py
# @Time      :2025/5/21 14:46
# @Author    :雨霓同学
# @Function  :
import argparse
from ultralytics import YOLO


def parse_args():
    """
    解析命令行参数,配置训练所需要的超参数
    :return: argparse.Namespace 包含所有命令行的参数对象
    """
    parser = argparse.ArgumentParser(description="YOLOv8 训练脚本，用于安全帽检测任务。支持命令行参数配置超参数，日志、设备信息、数据集统计和模型管理功能移到 utils 包。")
    parser.add_argument("--data", type=str, default="../configs/data.yaml", help="数据集配置文件路径")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="预训练模型路径")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--epochs", type=int, default=4, help="训练轮数")
    parser.add_argument("--device", type=str, default="0", help="计算设备，如 0（GPU）或 cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--run-name", type=str, default="train", help="运行名称")
    return parser.parse_args()

def main(args):
    """
    主训练函数,执行模型的加载和训练
    :param args: 命令行参数对象,包含训练配置
    :return: None
    """
    model = YOLO(args.weights)
    model.train(data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                device=args.device,
                project="run/detect",
                name=args.run_name,
                )



if __name__ == "__main__":
    args = parse_args()
    main(args)
