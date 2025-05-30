
import pyttsx3
import time
import logging
import numpy as np


logger = logging.getLogger(__name__)

def init_tts():
    """
    初始化语音合成引擎，配置语速，音量，并优先选择中文语音
    :return:
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        # 尝试选择中文语音
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'zh' in voice.id.lower() or 'chinese' in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        return engine
    except Exception as e:
        logger.error(f"初始化语音引擎失败: {e}")
        return None

def process_tts_detection(result, tts_enabled, tts_duration, tts_interval, tts_engine, tts_state):
    """
    处理Yolo检测结果，检测未佩戴安全的情况并触发提醒
    :param result: 模型推理的结果
    :param tts_enabled: 是否启用语音功能
    :param tts_duration: 触发警报需要的时间
    :param tts_interval: 语音提醒冷却，
    :param tts_engine: 语音合成引擎示例
    :param tts_state: 状态字典，存储未佩戴安全帽的开始时间和上次语音提醒时间
    :return:
    """
    if not tts_enabled or not tts_engine:
        return
    # 获取检测结果的类别
    classes = result.boxes.cls.cpu().numpy() if result.boxes else []
    has_person = 2 in classes
    has_head = 3 in classes
    has_safety_helmet = 0 in classes
    # 获取当前时间，用于判断冷却状态
    current_time = time.time()
    # 检查冷却状态
    in_cooldown = tts_state.get('last_tts_time') and (
        current_time - tts_state['last_tts_time'] < tts_interval
    )
    # 如果不存在冷却状态，处理安全帽未戴的情况
    if not in_cooldown:
        if has_person and has_head and not has_safety_helmet:
            # 首次检测未佩戴
            if tts_state.get('no_helmet_start_time') is None:
                tts_state['no_helmet_start_time'] = current_time
                logger.debug("检测到未佩戴安全帽，开始计时")
            # 检查持续时间
            elif current_time - tts_state['no_helmet_start_time'] >= tts_duration:
                logger.info(f"连续 {tts_duration}s 检测到未佩戴安全帽，开始触发语音提醒")
                try:
                    tts_engine.say("您已进入工地检测区域，请规范佩戴安全帽")
                    tts_engine.runAndWait()
                    tts_state['last_tts_time'] = current_time
                    tts_state['no_helmet_start_time'] = None  # 重置
                except Exception as e:
                    logger.error(f"语音播放失败: {e}")
        else:
            # 未检测到未佩戴，重置计时
            if tts_state.get('no_helmet_start_time') is not None:
                logger.debug("未佩戴状态中断，重置计时")
                tts_state['no_helmet_start_time'] = None








if __name__ == "__main__":
    run_code = 0
