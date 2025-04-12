# -*- coding: utf-8 -*-
import time 
import os
import sys
sys.path.insert(0, "/root/yolov12")
import cv2
import numpy as np
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
        self.model = YOLO('/root/src/models/best.pt')
        self.class_names = ['person']
        
        # 初始化视频流
        self.cap_ir = cv2.VideoCapture(self.ir_video)
        self.cap_thermal = cv2.VideoCapture(self.thermal_video)
        
        # 检查视频是否成功打开
        if not self.cap_ir.isOpened() or not self.cap_thermal.isOpened():
            raise ValueError("无法打开输入视频，请检查路径是否正确！")
        
        # 获取帧率和总帧数
        self.fps = int(self.cap_ir.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap_ir.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps == 0:
            self.fps = 20  # 默认帧率
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(
            f"{self.output_dir}/output_with_boxes.mp4", 
            fourcc, 
            self.fps, 
            self.frame_size
        )
        
        # 初始化结果记录
        self.result_file = open(f"{self.output_dir}/detection_results.txt", "w")
        self.frame_counter = 0

    def _process_single_frame(self, frame_ir, frame_thermal, dynamic_conf):
        """
        处理单帧的完整流程：
        1. 对齐 -> 2. 去烟 -> 3. 融合 -> 4. 检测
        """
        # 对齐处理
        align_start = time.time()
        aligned_thermal = self.aligner.align(frame_thermal, frame_ir)
        align_time = time.time() - align_start
        
        # 去烟处理
        dehaze_start = time.time()
        dehazed_ir, _ = self.dehazer.process(frame_ir)
        dehaze_time = time.time() - dehaze_start
        
        # 多模态融合
        fusion_start = time.time()
        fused_frame = self.fuser.fuse(dehazed_ir, aligned_thermal)
        fusion_time = time.time() - fusion_start
        
        # 目标检测
        detect_start = time.time()
        results = self.model.predict(
            fused_frame, 
            imgsz=640,
            conf=dynamic_conf,
            classes=[0],
            verbose=False
        )
        detect_time = time.time() - detect_start
        
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
        
        # 打印单帧处理时间
        if self.frame_counter % 30 == 0:
            print(f"Frame {self.frame_counter} 处理时间: "
                f"对齐={align_time:.3f}s, 去烟={dehaze_time:.3f}s, "
                f"融合={fusion_time:.3f}s, 检测={detect_time:.3f}s")
        
        return detections, annotated_frame

    def run(self):
        """主处理流程"""
        while True:
            # 同步读取双路视频帧
            ret_ir, frame_ir = self.cap_ir.read()
            ret_thermal, frame_thermal = self.cap_thermal.read()
            
            if not (ret_ir and ret_thermal):
                break
            
            # 动态调整置信度阈值
            dynamic_conf = 0.3 if self.frame_counter > self.total_frames - self.fps * 5 else 0.4
            
            # 调整帧尺寸
            frame_ir = cv2.resize(frame_ir, self.frame_size)
            frame_thermal = cv2.resize(frame_thermal, self.frame_size)
            
            # 处理流水线
            detections, annotated_frame = self._process_single_frame(frame_ir, frame_thermal, dynamic_conf)
            
            # 写入检测结果
            self.result_file.write(f"Frame {self.frame_counter}:\n")
            self.result_file.write("\n".join(detections) + "\n\n")
            
            if annotated_frame is not None:
                self.output_video.write(annotated_frame)
            
            self.frame_counter += 1

        # 释放资源
        self.cap_ir.release()
        self.cap_thermal.release()
        self.output_video.release()
        self.result_file.close()
        print(f"处理完成！共处理 {self.frame_counter} 帧")
        print(f"结果保存在：{self.output_dir}")

if __name__ == "__main__":
    start_time = time.time()
    
    system = FireRescueSystem(
        ir_video="/root/video/swir.mp4",
        thermal_video="/root/video/lwir.mp4",
        output_dir="/root/video/results"
    )
    system.run()
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"平均每帧处理时间: {total_time/system.frame_counter:.4f} 秒")