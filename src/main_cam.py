# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0,"/root/yolov12")
import cv2
import numpy as np
import time
from modules.dehazing import DehazingProcessor
from modules.fusion import MultimodalFuser
from modules.alignment import ImageAligner
from yolov12.ultralytics.models.yolo.model import YOLO 

class FireRescueSystem:
    def __init__(self):
        # 初始化硬件参数
        self.frame_size = (640, 480)
        self.fps_target = 30
        
        # 初始化算法模块
        self.dehazer = DehazingProcessor()
        self.fuser = MultimodalFuser()
        self.aligner = ImageAligner()
        
        # 初始化YOLOv12模型
        self.model = YOLO('models/yolov12n.pt')
        self.class_names = ['person', 'fire', 'obstacle']  # 示例类别
        
        # 初始化视频流
        self.cap_ir = cv2.VideoCapture(0)  # 红外摄像头
        self.cap_thermal = cv2.VideoCapture(1)  # 热成像摄像头
        self._setup_cameras()

    def _setup_cameras(self):
        """配置摄像头参数"""
        for cap in [self.cap_ir, self.cap_thermal]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps_target)

    def _process_single_frame(self, frame_ir, frame_thermal):
        """
        单帧处理流水线：
        1. 图像对齐 -> 2. 去烟处理 -> 3. 多模态融合 -> 4. 目标检测
        """
        # Step 1: 图像对齐
        aligned_thermal = self.aligner.align(frame_thermal, frame_ir)
        
        # Step 2: 去烟处理（可选）
        dehazed_ir, _ = self.dehazer.process(frame_ir)
        
        # Step 3: 多模态融合
        fused_frame = self.fuser.fuse(dehazed_ir, aligned_thermal)
        
        # Step 4: 目标检测
        results = self.model.predict(
            fused_frame, 
            imgsz=320,
            conf=0.6,  # 置信度阈值
            classes=[0],  # 只检测人员类别
            verbose=False
        )
        
        return results[0].plot(), fused_frame

    def run(self):
        """主运行循环"""
        fps_counter = 0
        start_time = time.time()
        
        while True:
            # 读取双路视频流
            ret_ir, frame_ir = self.cap_ir.read()
            ret_thermal, frame_thermal = self.cap_thermal.read()
            
            if not ret_ir or not ret_thermal:
                break

            # 镜像翻转
            frame_ir = cv2.flip(frame_ir, 1)
            frame_thermal = cv2.flip(frame_thermal, 1)

            # 处理流水线
            annotated_frame, processed_frame = self._process_single_frame(frame_ir, frame_thermal)
            
            # 显示处理结果
            self._display_output(annotated_frame, processed_frame, fps_counter, start_time)
            
            # 退出机制
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放资源
        self.cap_ir.release()
        self.cap_thermal.release()
        cv2.destroyAllWindows()

    def _display_output(self, frame, processed, fps_counter, start_time):
        """显示输出结果"""
        # FPS计算
        fps_counter += 1
        if (time.time() - start_time) > 1:
            fps = fps_counter / (time.time() - start_time)
            fps_counter = 0
            start_time = time.time()
        
        # 叠加显示信息
        info_text = f'FPS: {fps:.1f} | Mode: IR+Thermal Fusion'
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 并排显示原始帧和处理帧
        display_frame = np.hstack([processed, frame])
        cv2.imshow('Fire Rescue System', display_frame)

if __name__ == "__main__":
    system = FireRescueSystem()
    system.run()