# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np

class ImageAligner:
    """基于CV_Match(1).py的图像对齐模块"""
    def __init__(self, match_method=cv2.TM_CCOEFF_NORMED):
        self.match_method = match_method
        
    def align(self, source_img, target_img):
        """
        执行图像对齐
        :param source_img: 待对齐图像（热成像）
        :param target_img: 目标图像（红外）
        :return: 对齐后的热成像图像
        """
        if source_img.shape != target_img.shape:
            source_img = cv2.resize(source_img, (target_img.shape[1], target_img.shape[0]))
            
        # 转换为灰度图
        gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY) if len(source_img.shape)==3 else source_img
        gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) if len(target_img.shape)==3 else target_img
        
        # 执行模板匹配
        result = cv2.matchTemplate(gray_source, gray_target, self.match_method)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        
        # 裁剪对齐区域
        h, w = gray_target.shape
        aligned = source_img[max_loc[1]:max_loc[1]+h, max_loc[0]:max_loc[0]+w]
        
        return aligned