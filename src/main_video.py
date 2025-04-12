
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
    def __init__(self, 
                 ir_video="/root/video/swir.mp4",
                 thermal_video="/root/video/lwir.mp4",
                 output_dir="/root/video/results"):
        # 初始化视频参数
        self.ir_video = ir_video
        self.thermal_video = thermal_video
        self.output_dir = output_dir
        self.frame_size = (640, 480)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化算法模块
        self.dehazer = DehazingProcessor()
        self.fuser = MultimodalFuser()
        self.aligner = ImageAligner()
        
        # 初始化YOLOv12模型
        # self.model = YOLO('models/yolov12n.pt')
        # self.model = YOLO('/root/models/last.pt')
        self.model = YOLO('/root/src/models/best.pt')
        # self.class_names = ['person', 'fire', 'obstacle']
        self.class_names = ['person']
        
        # 初始化视频流
        self.cap_ir = cv2.VideoCapture(self.ir_video)
        self.cap_thermal = cv2.VideoCapture(self.thermal_video)
        
        # 初始化结果记录
        self.result_file = open(f"{self.output_dir}/detection_results.txt", "w")
        self.frame_counter = 0

    def _process_single_frame(self, frame_ir, frame_thermal):
        """
        处理单帧的完整流程：
        1. 对齐 -> 2. 去烟 -> 3. 融合 -> 4. 检测
        """
        # 对齐处理
        aligned_thermal = self.aligner.align(frame_thermal, frame_ir)
        
        # 去烟处理
        dehazed_ir, _ = self.dehazer.process(frame_ir)
        
        # 多模态融合
        fused_frame = self.fuser.fuse(dehazed_ir, aligned_thermal)
        
        # 目标检测
        results = self.model.predict(
            fused_frame, 
            imgsz=320,
            conf=0.6,
            classes=[0],  # 只检测人员
            verbose=False
        )
        
        # 记录检测结果
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(
                f"{self.class_names[cls_id]} {conf:.2f} {x1} {y1} {x2} {y2}"
            )
        
        # 保存带标注的帧
        annotated_frame = results[0].plot()
        output_path = f"{self.output_dir}/processed_{self.frame_counter:04d}.jpg"
        cv2.imwrite(output_path, annotated_frame)
        
        return detections, fused_frame

    def run(self):
        """主处理流程"""
        while True:
            # 同步读取双路视频帧
            ret_ir, frame_ir = self.cap_ir.read()
            ret_thermal, frame_thermal = self.cap_thermal.read()
            
            # 任一视频结束则停止
            if not (ret_ir and ret_thermal):
                break
            
            # 调整帧尺寸
            frame_ir = cv2.resize(frame_ir, self.frame_size)
            frame_thermal = cv2.resize(frame_thermal, self.frame_size)
            
            # 处理流水线
            detections, _ = self._process_single_frame(frame_ir, frame_thermal)
            
            # 写入检测结果
            self.result_file.write(f"Frame {self.frame_counter}:\n")
            self.result_file.write("\n".join(detections) + "\n\n")
            
            self.frame_counter += 1

        # 释放资源
        self.cap_ir.release()
        self.cap_thermal.release()
        self.result_file.close()
        print(f"处理完成！共处理 {self.frame_counter} 帧")
        print(f"结果保存在：{self.output_dir}")

if __name__ == "__main__":
    system = FireRescueSystem(
        ir_video="/root/video/swir.mp4",
        thermal_video="/root/video/lwir.mp4",
        output_dir="/root/video/results"
    )
    system.run()