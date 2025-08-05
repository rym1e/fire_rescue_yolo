# 一、项目概要：

* 一个基于yolov12-ultralytics的视频去烟处理系统，输入浓烟环境下的同一图片的短波和长波红外版本，利用双模态耦合，结合去烟算法进行处理，最终合成一个去烟后的视频
  
# 二、系统架构

 * 系统采用模块化设计（图1），包含以下核心组件：

* 数据输入层
* ├─ 近红外视频流（SWIR）
* └─ 远红外视频流（LWIR）
* 处理层
* ├─ 图像预处理模块
* │   ├─ 双流对齐
* │   └─ 分辨率标准化
* ├─ 去烟增强模块
* ├─ 多模态融合模块
* └─ 目标检测模块
* 输出层
* ├─ 实时视频标注流
* ├─ 检测结果日志
* └─ 处理效能报告

# 三、关键技术实现

* 3.1 基于暗通道先验的去烟算法
*实现原理：
* - 暗通道提取：J_dark(x)=min_{y∈Ω(x)}(min_{c∈{r,g,b}} J^c(y))
* - 大气光估计：选取暗通道前0.1%最亮像素均值
* - 透射率计算：t(x)=1-ω·J_dark(x)/A
* - 场景复原：J(x)=[I(x)-A]/t(x)+A
  - 
* 代码实现：
python:
  class DehazingProcessor:
      def process(self, image):
          rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          filtered = self.bilateral_filter(rgb_image)  # 双边滤波去噪
          dark = self._dark_channel(filtered, 15)      # 15x15窗口暗通道
          atmospheric_light = self._estimate_atmospheric_light(filtered, dark)
          transmission = 1 - 0.95*self._dark_channel(filtered/atmospheric_light, 15)
          dehazed = self._recover_scene_radiance(filtered, transmission, atmospheric_light, 0.1)
* 3.2 多模态图像融合
* 融合策略：
* 1. 显著性检测：S(x,y)=√(Sobel_x^2 + Sobel_y^2)
* 2. 引导滤波分解：
*   - 低频层：L=GF(I)（半径10，ε=1e-6）
*   - 高频层：H=I-L
* 3. 加权融合：
* Fused=W·H_ir+(1-W)·H_lwir + W·L_ir+(1-W)·L_lwir
* （W=Threshold(S,0.5)）
* 实现效果：在浓烟环境下，融合后图像特征可见度提升42%（图2）。
* 3.3 轻量化目标检测
* 模型配置：
* - 基础网络：YOLOv12n（Nano版）
* - 输入尺寸：320×320
* - 类别过滤：仅保留人体检测
* - 硬件加速：OpenVINO推理引擎
* 性能优化：
*   python
    self.model = YOLO('models/yolov12n.pt').export(format='openvino')
    self.model.predict(..., 
        imgsz=320, 
        conf=0.6, 
        classes=[0],  # 人员类别
        half=True)    # FP16量化
      

