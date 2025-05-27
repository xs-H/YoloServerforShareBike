import cv2
import time
import queue
import threading
import random


class VideoProcessor:
    def __init__(self):
        # 硬件初始化
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("摄像头初始化失败")

        # 固定输出参数
        self.target_fps = 30  # 目标30FPS
        self.target_width, self.target_height = 1280, 720

        # 设置摄像头分辨率（匹配输出，避免不必要缩放）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # 视频写入器
        self.writer = cv2.VideoWriter(
            'output.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.target_fps,
            (self.target_width, self.target_height)
        )
        if not self.writer.isOpened():
            self.cap.release()
            raise RuntimeError("视频写入器初始化失败")

        # 多线程通信
        self.frame_queue = queue.Queue(maxsize=2)  # 小队列防积压
        self.stop_event = threading.Event()

        # 时间控制
        self.frame_interval = 1.0 / self.target_fps  # 33.3ms per frame
        self.last_frame_time = time.time()

    def capture_and_process(self):
        """采集和处理线程"""
        while not self.stop_event.is_set():
            # 采集帧
            ret, frame = self.cap.read()
            if not ret:
                print("警告：无法读取摄像头帧")
                time.sleep(0.001)  # 避免空循环过快
                continue

            # 模拟处理延迟（16-30ms，可调整为更大值）
            # processing_time = 0.016 + 0.014 * (time.time() % 1)  # 16-30ms
            # time.sleep(processing_time)
            # 模拟更大延迟（100-500ms）
            processing_time = random.uniform(0.015, 0.040)  # 100-500ms
            time.sleep(processing_time)

            # 处理帧
            resized = cv2.resize(frame, (self.target_width, self.target_height))
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(resized, f"Proc: {processing_time * 1000:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(resized, timestamp, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 非阻塞写入队列
            try:
                self.frame_queue.put_nowait((resized, time.time()))
            except queue.Full:
                # 清空旧帧，保留最新帧
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                try:
                    self.frame_queue.put_nowait((resized, time.time()))
                except queue.Full:
                    pass

            # 帧率控制
            now = time.time()
            elapsed = now - self.last_frame_time
            sleep_time = max(0, self.frame_interval - elapsed - 0.001)
            time.sleep(sleep_time)
            self.last_frame_time = now

    def run(self):
        # 启动工作线程
        worker = threading.Thread(target=self.capture_and_process)
        worker.daemon = True
        worker.start()

        # 等待首帧，确保队列非空
        print("等待首帧...")
        start_time = time.time()
        while self.frame_queue.empty() and not self.stop_event.is_set():
            time.sleep(0.01)
            if time.time() - start_time > 5:  # 超时5秒
                print("错误：无法获取首帧，退出")
                self.stop_event.set()
                break

        # 帧率控制变量
        start_time = time.time()
        frame_count = 0

        try:
            while not self.stop_event.is_set():
                # 获取最新帧
                latest_frame = None
                while not self.frame_queue.empty():
                    latest_frame, _ = self.frame_queue.get_nowait()

                if latest_frame is not None:
                    # 写入视频文件
                    self.writer.write(latest_frame)

                    # 显示实时参数
                    fps_text = f"Target FPS: {self.target_fps} | Frame: {frame_count}"
                    cv2.putText(latest_frame, fps_text, (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.imshow("Output", latest_frame)
                    frame_count += 1

                # 键盘退出控制（q或ESC）
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    print(f"退出键: {'q' if key == ord('q') else 'ESC'}")
                    self.stop_event.set()
                    break

        except Exception as e:
            print(f"主循环异常: {e}")
            self.stop_event.set()

        finally:
            # 停止线程并清理资源
            self.stop_event.set()
            worker.join(timeout=0.5)

            # 清空队列
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            # 释放资源
            if self.cap.isOpened():
                self.cap.release()
            if self.writer.isOpened():
                self.writer.release()
            cv2.destroyAllWindows()

            # 计算并打印平均帧率
            elapsed_time = time.time() - start_time
            if elapsed_time > 0 and frame_count > 0:
                print(f"视频已保存，平均帧率: {frame_count / elapsed_time:.1f}FPS")
            else:
                print("视频已保存，未记录有效帧率")


if __name__ == "__main__":
    try:
        vp = VideoProcessor()
        vp.run()
    except Exception as e:
        print(f"程序异常退出: {e}")