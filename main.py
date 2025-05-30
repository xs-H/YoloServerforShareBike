import sys
from pathlib import Path
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog, QLabel, QTextEdit, QMessageBox
from PySide6.QtGui import QIcon, QPixmap, QImage
from PySide6.QtCore import QThread, Signal, Qt
import cv2
import numpy as np
import logging
from yoloside6 import Ui_MainWindow
from utils.infer_stream import stream_inference

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SafeUI")

class InferenceThread(QThread):
    frame_ready = Signal(np.ndarray, np.ndarray, object)  # 原始帧, 推理帧, YOLO Result 对象
    progress_updated = Signal(int)
    error_occurred = Signal(str)

    def __init__(self, source, weights, project_args, main_window):
        super().__init__()
        self.source = source
        self.weights = weights
        self.project_args = project_args
        self.main_window = main_window  # 引用 MainWindow 以获取 UI 滑块值
        self.running = True
        self.paused = False
        self.is_camera = source == "0"
        self.is_image = Path(source).is_file() and source.lower().endswith(('.jpg', '.jpeg', '.png'))
        self.is_directory = Path(source).is_dir()
        self.cap = None

    def run(self):
        try:
            total_frames = 0
            frame_interval = 1000 if (self.is_image or self.is_directory) else None
            if self.is_camera or (not self.is_image and not self.is_directory):
                self.cap = cv2.VideoCapture(0 if self.is_camera else self.source)
                if not self.cap.isOpened():
                    self.error_occurred.emit(f"无法打开{'摄像头' if self.is_camera else '视频'}")
                    return
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_camera else 0
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                frame_interval = 1000 / fps
            else:
                source_path = Path(self.source)
                image_files = [source_path] if self.is_image else sorted(source_path.glob("*.[jp][pn][gf]"))
                total_frames = len(image_files)
                if total_frames == 0:
                    self.error_occurred.emit("目录中无图片文件")
                    return

            idx = 0
            for raw_frame, annotated_frame, result in stream_inference(
                self.weights, self.source, self.project_args, self.get_yolo_args(),
                pause_callback=lambda: self.paused or not self.running
            ):
                if not self.running:
                    break
                if self.paused:
                    logger.debug("InferenceThread 暂停")
                    self.msleep(100)
                    continue
                self.frame_ready.emit(raw_frame, annotated_frame, result)
                if not self.is_camera:
                    idx += 1
                    progress = int(idx / total_frames * 100) if total_frames > 0 else 0
                    self.progress_updated.emit(progress)
                self.msleep(int(frame_interval) if frame_interval else 10)
        except Exception as e:
            self.error_occurred.emit(f"推理失败: {str(e)}")
            logger.error(f"InferenceThread 错误: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None
            logger.info("InferenceThread 已清理")

    def get_yolo_args(self):
        return {
            'conf': self.main_window.ui.conf_num.value(),
            'iou': self.main_window.ui.iou_number.value(),
            'imgsz': 640,
            'stream': True,
            'save_txt': self.project_args.get('save_txt', False),
            'save_conf': self.project_args.get('save_conf', False),
            'save_crop': self.project_args.get('save_crop', False)
        }

    def terminate(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        super().terminate()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.inference_thread = None
        self.source = None
        self.is_camera = False
        self.is_image = False
        self.is_directory = False
        self.model_path = None
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        self.ui.model_name.setReadOnly(True)
        self.ui.model_name.setPlaceholderText("请选择模型文件...")
        self.ui.model_name.setStyleSheet("QLineEdit { border: 1px solid gray; padding: 2px; text-overflow: ellipsis; }")
        self.ui.model_name.setMaximumWidth(200)

        icon_path = Path(__file__).parent / "icons" / "folder.png"
        if icon_path.exists():
            self.ui.model_select.setIcon(QIcon(str(icon_path)))
            self.ui.model_select.setText("")
        else:
            self.ui.model_select.setText("选择模型")

        self.ui.upload_image.setScaledContents(True)
        self.ui.upload_image.setText("上传预览")
        self.ui.finall_result.setScaledContents(True)
        self.ui.finall_result.setText("检测结果")

        self.ui.video_progressBar.setValue(0)
        self.ui.video_progressBar.setTextVisible(True)

        self.ui.conf_num.setRange(0.0, 1.0)
        self.ui.conf_num.setSingleStep(0.05)
        self.ui.conf_num.setValue(0.25)
        self.ui.conf_slider.setMinimum(0)
        self.ui.conf_slider.setMaximum(100)
        self.ui.conf_slider.setValue(25)
        self.ui.iou_number.setRange(0.0, 1.0)
        self.ui.iou_number.setSingleStep(0.05)
        self.ui.iou_number.setValue(0.45)
        self.ui.iou_slider.setMinimum(0)
        self.ui.iou_slider.setMaximum(100)
        self.ui.iou_slider.setValue(45)
        self.ui.save_data.setChecked(True)
        self.ui.detection_quantity.setText("共有共享单车: 0辆")
        self.ui.detection_time.setText("0 ms")
        self.ui.detection_result.setText("无检测结果")

        self.statusBar = self.statusBar()
        self.status_label = QLabel("就绪")
        self.fps_label = QLabel("FPS: 0")
        self.statusBar.addWidget(self.status_label)
        self.statusBar.addWidget(self.fps_label)

        try:
            self.ui.log_display = QTextEdit()
            self.ui.log_display.setReadOnly(True)
            self.ui.log_display.setMaximumHeight(100)
            self.ui.verticalLayout.addWidget(self.ui.log_display)
        except AttributeError:
            logger.warning("无法添加日志显示，请检查 UI 布局")

        self.update_button_states()

    def connect_signals(self):
        self.ui.model_select.clicked.connect(self.select_model)
        self.ui.video.clicked.connect(self.select_video)
        self.ui.image.clicked.connect(self.select_image)
        self.ui.dirs.clicked.connect(self.select_dirs)
        self.ui.camera.clicked.connect(self.select_camera)
        self.ui.yolo_start.clicked.connect(self.start_inference)
        self.ui.video_start.clicked.connect(self.start_video)
        self.ui.video_stop.clicked.connect(self.stop_video)
        self.ui.video_termination.clicked.connect(self.terminate_video)
        self.ui.conf_num.valueChanged.connect(self.sync_conf_slider)
        self.ui.conf_slider.valueChanged.connect(self.sync_conf_num)
        self.ui.iou_number.valueChanged.connect(self.sync_iou_slider)
        self.ui.iou_slider.valueChanged.connect(self.sync_iou_num)

    def sync_conf_slider(self, value):
        self.ui.conf_slider.setValue(int(value * 100))
        logger.debug(f"更新 conf 值: {value}")

    def sync_conf_num(self, value):
        self.ui.conf_num.setValue(value / 100.0)
        logger.debug(f"更新 conf 值: {value / 100.0}")

    def sync_iou_slider(self, value):
        self.ui.iou_slider.setValue(int(value * 100))
        logger.debug(f"更新 iou 值: {value}")

    def sync_iou_num(self, value):
        self.ui.iou_number.setValue(value / 100.0)
        logger.debug(f"更新 iou 值: {value / 100.0}")

    def select_model(self):
        try:
            # default_dir = Path(__file__).parent.parent / "yoloserver" / "weights"
            default_dir = Path(__file__).parent / "weights"

            default_dir = default_dir.resolve()
            default_dir.mkdir(parents=True, exist_ok=True)

            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择 YOLO 模型文件", str(default_dir), "YOLO 模型文件 (*.pt);;所有文件 (*.*)"
            )
            if file_path:
                self.model_path = file_path
                self.ui.model_name.setText(Path(file_path).name)
                logger.info(f"选择的模型: {self.model_path}")
            else:
                self.model_path = None
                self.ui.model_name.setText("")
                logger.info("未选择模型")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择模型失败")
            logger.error(f"选择模型失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择模型失败: {str(e)}")

    def select_video(self):
        try:
            # default_dir = Path(__file__).parent.parent / "yoloserver" / "inputs"
            default_dir = Path(__file__).parent / "inputs"

            default_dir = default_dir.resolve()
            default_dir.mkdir(parents=True, exist_ok=True)

            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", str(default_dir), "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*.*)"
            )
            if file_path:
                self.terminate_video()
                self.source = file_path
                self.is_camera = False
                self.is_image = False
                self.is_directory = False
                self.show_preview(file_path, is_video=True)
                logger.info(f"选择的视频: {self.source}")
            else:
                self.source = None
                self.is_image = False
                self.is_directory = False
                self.ui.upload_image.setText("上传预览")
                logger.info("未选择视频")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择视频失败")
            logger.error(f"选择视频失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择视频失败: {str(e)}")

    def select_image(self):
        try:
            # default_dir = Path(__file__).parent.parent / "yoloserver" / "inputs"
            default_dir = Path(__file__).parent / "inputs"

            default_dir = default_dir.resolve()
            default_dir.mkdir(parents=True, exist_ok=True)

            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片文件", str(default_dir), "图片文件 (*.jpg *.jpeg *.png);;所有文件 (*.*)"
            )
            if file_path:
                self.terminate_video()
                self.source = file_path
                self.is_camera = False
                self.is_image = True
                self.is_directory = False
                self.show_preview(file_path, is_video=False)
                logger.info(f"选择的图片: {self.source}")
            else:
                self.source = None
                self.is_image = False
                self.is_directory = False
                self.ui.upload_image.setText("上传预览")
                logger.info("未选择图片")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择图片失败")
            logger.error(f"选择图片失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择图片失败: {str(e)}")

    def select_dirs(self):
        try:
            # default_dir = Path(__file__).parent.parent / "yoloserver" / "inputs"
            default_dir = Path(__file__).parent / "inputs"

            default_dir = default_dir.resolve()
            default_dir.mkdir(parents=True, exist_ok=True)

            dir_path = QFileDialog.getExistingDirectory(self, "选择图片或视频目录", str(default_dir))
            if dir_path:
                self.terminate_video()
                self.source = dir_path
                self.is_camera = False
                self.is_image = False
                self.is_directory = True
                for img_path in Path(dir_path).glob("*.[jp][pn][gf]"):
                    self.show_preview(str(img_path), is_video=False)
                    break
                else:
                    self.ui.upload_image.setText(f"已选择目录: {Path(dir_path).name}（无图片预览）")
                logger.info(f"选择的目录: {self.source}")
            else:
                self.source = None
                self.is_image = False
                self.is_directory = False
                self.ui.upload_image.setText("上传预览")
                logger.info("未选择目录")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择目录失败")
            logger.error(f"选择目录失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择目录失败: {str(e)}")

    def select_camera(self):
        try:
            self.terminate_video()
            self.source = "0"
            self.is_camera = True
            self.is_image = False
            self.is_directory = False
            self.ui.upload_image.setText("摄像头已选择，点击开始播放")
            logger.info("选择输入: 摄像头")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择摄像头失败")
            logger.error(f"选择摄像头失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择摄像头失败: {str(e)}")

    def show_preview(self, file_path, is_video=False):
        try:
            if is_video:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        self.ui.upload_image.setText("无法读取视频")
                        return
                else:
                    self.ui.upload_image.setText("无法打开视频")
                    return
            else:
                frame = cv2.imread(file_path)
                if frame is None:
                    self.ui.upload_image.setText("无法读取图片")
                    return

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.ui.upload_image.setPixmap(pixmap.scaled(self.ui.upload_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logger.debug(f"显示预览: {file_path}, shape: {h}x{w}")
        except Exception as e:
            self.status_label.setText("预览失败")
            logger.error(f"显示预览失败: {str(e)}")
            self.ui.log_display.append(f"错误: 显示预览失败: {str(e)}")

    def start_inference(self):
        try:
            if not self.model_path:
                self.status_label.setText("请先选择模型文件")
                return
            if not self.source:
                self.status_label.setText("请先选择输入源")
                return
            self.start_video()
        except Exception as e:
            self.status_label.setText(f"错误: 开始推理失败")
            logger.error(f"开始推理失败: {str(e)}")
            self.ui.log_display.append(f"错误: 开始推理失败: {str(e)}")

    def start_video(self):
        try:
            if not self.source:
                self.status_label.setText("请先选择输入源")
                self.ui.upload_image.setText("请先选择视频、摄像头、图片或目录")
                return

            if self.inference_thread and self.inference_thread.isRunning():
                self.inference_thread.paused = False
                self.status_label.setText("正在推理")
                logger.info("推理已恢复")
                self.update_button_states()
                return

            self.inference_thread = InferenceThread(self.source, self.model_path, self.project_args(), self)
            self.inference_thread.frame_ready.connect(self.update_frames)
            self.inference_thread.progress_updated.connect(self.update_progress)
            self.inference_thread.error_occurred.connect(self.show_error)
            self.inference_thread.finished.connect(self.video_finished)
            self.inference_thread.start()
            self.status_label.setText("正在推理")
            logger.info("推理已开始")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 开始推理失败")
            logger.error(f"开始推理失败: {str(e)}")
            self.ui.log_display.append(f"错误: 开始推理失败: {str(e)}")

    def stop_video(self):
        try:
            if self.inference_thread and self.inference_thread.isRunning():
                self.inference_thread.paused = True
                self.status_label.setText("已暂停")
                logger.info("推理已暂停")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 暂停失败")
            logger.error(f"暂停失败: {str(e)}")
            self.ui.log_display.append(f"错误: 暂停失败: {str(e)}")

    def terminate_video(self):
        try:
            logger.info("开始终止线程")
            if self.inference_thread and self.inference_thread.isRunning():
                self.inference_thread.running = False
                self.inference_thread.quit()
                self.inference_thread.wait(500)
                self.inference_thread.terminate()
                self.inference_thread = None
                logger.info("推理已终止")
            if not self.is_image:
                self.ui.upload_image.setText("上传预览")
                self.ui.finall_result.setText("检测结果")
            self.ui.video_progressBar.setValue(0)
            self.ui.detection_quantity.setText("共有共享单车: 0辆")
            self.ui.detection_time.setText("0 ms")
            self.ui.detection_result.setText("无检测结果")
            self.status_label.setText("就绪")
            self.update_button_states()
            logger.info("UI 已重置")
        except Exception as e:
            self.status_label.setText(f"错误: 停止失败")
            logger.error(f"停止失败: {str(e)}")
            self.ui.log_display.append(f"错误: 停止失败: {str(e)}")

    def closeEvent(self, event):
        try:
            logger.info("开始关闭窗口")
            self.terminate_video()
            event.accept()
            logger.info("窗口已关闭")
        except Exception as e:
            logger.error(f"关闭窗口失败: {str(e)}")
            self.ui.log_display.append(f"错误: 关闭窗口失败: {str(e)}")
            event.ignore()

    def project_args(self):
        return {
            'save': self.ui.save_data.isChecked(),
            'save_frame': self.ui.save_data.isChecked(),
            'save_txt': self.ui.save_data.isChecked(),
            'save_conf': self.ui.save_data.isChecked(),
            'save_crop': self.ui.save_data.isChecked(),
            'display_size': '720p',
            'beautify': True,
            'use_chinese': True,
            'font_size': 22,
            'line_width': 4,
            'label_padding_x': 30,
            'label_padding_y': 18,
            'radius': 8,
            'log_encoding': 'utf-8-sig',
        }

    def update_frames(self, raw_frame, annotated_frame, result):
        try:
            start_time = cv2.getTickCount()
            frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.ui.upload_image.setPixmap(pixmap.scaled(self.ui.upload_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.ui.finall_result.setPixmap(pixmap.scaled(self.ui.finall_result.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


            total_time = 0.0
            if result is not None and hasattr(result, 'boxes'):
                others = sum(1 for box in result.boxes if int(box.cls) == 3)
                shared_bike_haluo = sum(1 for box in result.boxes if int(box.cls) == 2)
                shared_bike_meituan = sum(1 for box in result.boxes if int(box.cls) == 0)
                shared_bike_qingcong = sum(1 for box in result.boxes if int(box.cls) == 1)
                total = others + shared_bike_qingcong + shared_bike_meituan + shared_bike_haluo
                total_time = sum(result.speed.values()) if hasattr(result, 'speed') else 0.0
            self.ui.detection_quantity.setText(f"共有共享单车: {total}辆")
            self.ui.detection_time.setText(f"{total_time:.2f} ms")
            self.ui.detection_result.setText(f"""共检测到自行车有:  {total} \n
    其中哈啰共享单车的数量有: {shared_bike_haluo} 辆\n
    其中美团共享单车的数量有: {shared_bike_meituan} 辆\n
    其中青葱共享单车的数量有: {shared_bike_qingcong} 辆\n
    其中不是共享单车的数量有: {others} 辆\n
    当前帧检测耗时: {total_time:.2f} ms
""")

            end_time = cv2.getTickCount()
            frame_time = ((end_time - start_time) / cv2.getTickFrequency()) * 1000
            fps = 1000 / frame_time if frame_time > 0 else 0
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.status_label.setText("正在推理")
            logger.debug(f"帧更新耗时: {frame_time:.2f}ms")
        except Exception as e:
            self.status_label.setText("更新帧失败")
            logger.error(f"更新帧失败: {str(e)}")
            self.ui.log_display.append(f"错误: 更新帧失败: {str(e)}")

    def show_error(self, error_msg):
        self.status_label.setText(f"错误: {error_msg}")
        self.ui.upload_image.setText(error_msg)
        self.ui.finall_result.setText(error_msg)
        self.ui.detection_quantity.setText("共有共享单车: 0辆")
        self.ui.detection_time.setText("耗时: 0 ms")
        self.ui.detection_result.setText("无检测结果")
        self.terminate_video()
        logger.error(f"错误: {error_msg}")
        self.ui.log_display.append(f"错误: {error_msg}")
        QMessageBox.critical(self, "错误", error_msg)

    def video_finished(self):
        logger.info("推理完成")
        if self.is_image:
            self.status_label.setText("推理完成（单张图像）")
        else:
            self.terminate_video()
        logger.info("播放已完成")

    def update_progress(self, progress):
        self.ui.video_progressBar.setValue(progress)
        logger.debug(f"进度更新: {progress}%")

    def update_button_states(self):
        has_source = getattr(self, 'source', None) is not None
        has_model = getattr(self, 'model_path', None) is not None
        is_running = bool(getattr(self, 'inference_thread', None) and self.inference_thread.isRunning())
        is_paused = bool(is_running and getattr(self.inference_thread, 'paused', False))
        self.ui.yolo_start.setEnabled(bool(has_source and has_model and not is_running))
        self.ui.video_start.setEnabled(bool(has_source and has_model and (not is_running or is_paused)))
        self.ui.video_stop.setEnabled(bool(is_running and not is_paused))
        self.ui.video_termination.setEnabled(bool(is_running))
        self.ui.video_progressBar.setEnabled(bool(has_source and not getattr(self, 'is_camera', False)))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())