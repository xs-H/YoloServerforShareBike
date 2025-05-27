#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test_tts_tea.py
# @Time      :2025/5/27 14:06
# @Author    :雨霓同学
# @Function  :
import pyttsx3  # pip install pyttsx3

def init_tts():
    """
    初始化语音合成
    :return:
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150) # 设置语速
        engine.setProperty('volume', 0.9)  # 设置音量
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'zh' in voice.id.lower() or 'chinese' in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        return engine
    except Exception as e:
        print(f"初始化语音引擎失败: {e}")
        return None

def test_tts():
    engine = init_tts()
    if not engine:
        print("无法初始化语音引擎")
        return
    text = "您已进入工地检测区域，请您规范佩戴安全帽"
    try:
        engine.say(text)
        engine.runAndWait()
        print('语音播放成功')
    except Exception as e:
        print(f"语音合成失败: {e}")


if __name__ == "__main__":
    test_tts()
