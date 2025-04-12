# -*- coding: utf-8 -*-

import cv2
import numpy as np

class DehazingProcessor:
    """去烟算法封装（基于dcpbfgs.py）"""
    def __init__(self, omega=0.95, t0=0.1, window_size=15):
        self.params = {
            'omega': omega,       # 去雾程度系数
            't0': t0,             # 传输率下限阈值
            'window_size': window_size  # 暗通道窗口尺寸
        }
    
    def process(self, image):
        """
        去烟处理主流程
        :param image: 输入图像 (BGR格式)
        :return: 去烟图像, 暗通道图
        """
        # 颜色空间转换
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 双边滤波
        filtered = self.bilateral_filter(rgb_image)
        
        # 去烟处理
        dehazed, dark, _ = self.dehaze(filtered)
        
        # 转换回BGR
        return cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR), dark
    
    def bilateral_filter(self, image, d=9, sigma_color=75, sigma_space=75):
        """双边滤波去噪"""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def dehaze(self, image):
        """
        暗通道去雾主算法
        :param image: RGB格式输入图像
        :return: 去雾图像, 暗通道图, 透射率图
        """
        # 获取算法参数
        omega = self.params['omega']
        t0 = self.params['t0']
        window_size = self.params['window_size']
        
        # 1. 计算暗通道
        dark = self._dark_channel(image, window_size)
        
        # 2. 估计大气光
        atmospheric_light = self._estimate_atmospheric_light(image, dark)
        
        # 3. 估计透射率
        transmission = self._estimate_transmission(image, atmospheric_light, omega, window_size)
        
        # 4. 复原场景辐射
        dehazed_image = self._recover_scene_radiance(image, transmission, atmospheric_light, t0)
        
        return dehazed_image, dark, transmission

    def _dark_channel(self, image, window_size):
        """计算暗通道"""
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
        return cv2.erode(min_channel, kernel)

    def _estimate_atmospheric_light(self, image, dark_channel):
        """估计大气光值"""
        num_pixels = image.shape[0] * image.shape[1]
        num_brightest = int(max(num_pixels * 0.001, 1))  # 取前0.1%最亮像素
        dark_vec = dark_channel.ravel()
        indices = dark_vec.argsort()[-num_brightest:]
        
        brightest_pixels = image.reshape(-1, 3)[indices]
        return brightest_pixels.mean(axis=0)

    def _estimate_transmission(self, image, atmospheric_light, omega, window_size):
        """估计透射率"""
        norm_image = image / atmospheric_light
        transmission = 1 - omega * self._dark_channel(norm_image, window_size)
        return np.clip(transmission, 0, 1)

    def _recover_scene_radiance(self, image, transmission, atmospheric_light, t0):
        """复原场景辐射"""
        transmission = np.clip(transmission, t0, 1)
        recovered = (image - atmospheric_light) / transmission[..., None] + atmospheric_light
        return np.clip(recovered, 0, 255).astype(np.uint8)