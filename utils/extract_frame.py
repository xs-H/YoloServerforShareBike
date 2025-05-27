import cv2  # Opencv库,用于提取视频
import os
import argparse  # 命令行解析参数

def extract_frames(video_path, output_dir, frame_interval):
    """
    从视频中按指定的间隔提取帧并保存为图像
    :param video_path: 输入视频文件路径
    :param output_dir: 输出图像目录
    :param frame_interval: 帧提取间隔
    :return:
    """
    # 创建一个输出的目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误:无法打开视频文件:{video_path}")
        return
    # 获取视频属性
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获得视频的帧率
    print(f"视频总帧数: {total_frames}, 帧率:{fps:.2f}FPS")

    # 初始化计数器
    frame_count = 0
    saved_count = 0

    # 逐帧读取视频
    while True:
        ret, fream = cap.read() # 读取下一帧
        if not ret:
            break
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.png")
            cv2.imwrite(output_path, fream)
            saved_count += 1
            print(f"已经保存第{saved_count}帧到{output_path}")
        frame_count += 1
    # 释放视频资源
    cap.release()
    print(f"完成!共提取了{saved_count}帧到{output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="主要用于从视频中提取图像用于Yolo训练")
    parser.add_argument("--video", required=True, help="输入视频的文件路径")
    parser.add_argument("--output", required=True, help="输出的目录，用于保存提取的帧")
    parser.add_argument("--interval", type=int, default=10, help="帧提取间隔")

    args = parser.parse_args()
    extract_frames(args.video, args.output, args.interval)
    # extract_frames("../demos.mp4",'../extract', 10)