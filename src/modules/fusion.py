# -*- coding: utf-8 -*-
import cv2
import numpy as np

class MultimodalFuser:
    """基于Final_st.py的多模态融合模块"""
    def __init__(self, r=10, eps=1e-6):
        self.r = r  # 引导滤波半径
        self.eps = eps  # 正则化参数
        
    def fuse(self, ir_img, thermal_img):
        """
        执行双模态融合
        :param ir_img: 红外图像 (去烟后)
        :param thermal_img: 热成像图像 (对齐后)
        :return: 融合后的图像
        """
        # 转换为灰度
        gray_ir = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY) if len(ir_img.shape)==3 else ir_img
        gray_thermal = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY) if len(thermal_img.shape)==3 else thermal_img
        
        # 计算显著性图
        saliency = self._compute_saliency(thermal_img)
        weight_map = self._compute_weight(saliency)
        
        # 引导滤波提取低频分量
        low_ir = self._guided_filter(gray_ir, gray_ir)
        low_thermal = self._guided_filter(gray_thermal, gray_thermal)
        
        # 计算高频分量
        high_ir = gray_ir - low_ir
        high_thermal = gray_thermal - low_thermal
        
        # 加权融合
        fused_high = weight_map*high_ir + (1-weight_map)*high_thermal
        fused_low = weight_map*low_ir + (1-weight_map)*low_thermal
        fused = fused_high + fused_low
        
        # 后处理
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        return cv2.cvtColor(fused, cv2.COLOR_GRAY2BGR)
    
    def _guided_filter(self, I, p):
        """引导滤波实现"""
        mean_I = cv2.boxFilter(I, cv2.CV_64F, (self.r, self.r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (self.r, self.r))
        corr_I = cv2.boxFilter(I*I, cv2.CV_64F, (self.r, self.r))
        corr_Ip = cv2.boxFilter(I*p, cv2.CV_64F, (self.r, self.r))
        
        var_I = corr_I - mean_I*mean_I
        cov_Ip = corr_Ip - mean_I*mean_p
        
        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a*mean_I
        
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (self.r, self.r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (self.r, self.r))
        
        return mean_a*I + mean_b
    
    def _compute_saliency(self, image):
        """计算显著性图"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        saliency = np.sqrt(sobelx**2 + sobely**2)
        return cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
    
    def _compute_weight(self, saliency, threshold=0.5):
        """生成权重图"""
        _, weight = cv2.threshold(saliency, threshold, 1, cv2.THRESH_BINARY)
        return weight