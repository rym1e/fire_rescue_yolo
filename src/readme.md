Fire_Rescue_System/

├── main.py                  # 系统主入口

├── configs/

│   └── params.yaml          # 超参数配置文件

├── modules/

│   ├── dehazing.py          # 去烟算法模块（基于dcpbfgs.py）

│   ├── fusion.py            # 双模态融合模块（基于Final_st.py）

│   ├── alignment.py         # 图像对齐模块（基于CV_Match(1).py）

│   └── utils.py             # 工具函数

├── models/

│   └── yolov12n.pt          # YOLOv12预训练模型

└── requirements.txt         # 依赖库列表

SWIR视频帧 → 调整尺寸 → 对齐处理 → 去烟处理 ↘

                                          融合 → 检测 → 结果输出
                                          
LWIR视频帧 → 调整尺寸 → 对齐处理 → 保持原貌 ↗

/root/video/results/

├── detection_results.txt  # 检测结果文本

├── processed_0000.jpg     # 带标注的结果帧

├── processed_0001.jpg

└── ...
