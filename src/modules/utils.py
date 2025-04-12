# -*- coding: utf-8 -*-
import time
import cv2

class FPSController:
    """帧率控制工具"""
    def __init__(self, target_fps):
        self.target_delay = 1.0 / target_fps
        self.prev_time = time.time()
        
    def wait_frame(self):
        """维持目标帧率"""
        curr_time = time.time()
        elapsed = curr_time - self.prev_time
        wait_time = max(0, self.target_delay - elapsed)
        
        if wait_time > 0:
            key = cv2.waitKey(int(wait_time*1000))
        else:
            key = cv2.waitKey(1)
            
        self.prev_time = time.time()
        return key

class VideoWriter:
    """视频记录工具"""
    def __init__(self, filename, frame_size, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        
    def write_frame(self, frame):
        self.writer.write(frame)
        
    def release(self):
        self.writer.release()