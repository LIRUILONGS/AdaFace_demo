#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   service.py
@Time    :   2023/07/03 04:58:29
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   人脸识别 服务层
"""

# here put the import lib


# here put the import lib

import cv2
import imutils
from deepface import DeepFace
from decimal import Decimal
from imutils import paths
import os
import sys
sys.path.append("..") 



def represent(img_path, model_name, detector_backend, enforce_detection, align):
    """
    @Time    :   2023/05/22 12:11:58
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   None
                 Args:
                   
                 Returns:
                   void
    """
    result = {}

    return result


def verify(
    img1_path, img2_path, model_name, detector_backend, distance_metric, enforce_detection, align
):
    obj = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )
    return obj
