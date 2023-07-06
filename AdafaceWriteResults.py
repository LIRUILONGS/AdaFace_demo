#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   AdafaceWriteResults.py
@Time    :   2023/07/03 22:46:13
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   输出识别结果
"""

# here put the import lib




# here put the import lib
import net
import torch
import os
from pathlib import Path
from face_alignment import align
import numpy as np
import pandas as pd
import pickle
import time
import imutils 
from imutils import paths
import cv2
from os import path
from tqdm import tqdm
from PIL import Image
import hashlib
import glob
from redis_uits import RedisClient
import json
import utils


class AdafaceWriteResults:
    __instance = None
    


    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self,key_Y="face_Y",key_N="face_N",w_path="./"):
        """
        @Time    :   2023/06/17 22:41:40
        @Author  :   liruilonger@gmail.com
        @Version :   3.0
        @Desc    :   初始化处理，加载模型，特征文件加载
        """
        
        # redis
        self.rc = RedisClient()
        self.key_Y = key_Y
        self.key_N = key_N
        self.w_path = w_path


    def to_file(self):
        fs_Y =  list(self.rc.hgetall(self.key_Y))
        
        for fn in fs_Y:
           b64 =  self.rc.hget(self.key_Y,fn)
           utils.save_image_from_base64(b64.decode('utf-8'),'./',fn.decode('utf-8'))
           self.rc.hdel(self.key_Y,fn)

        fs_N =  list(self.rc.hgetall(self.key_N))
        for fn in fs_N:
           b64 =  self.rc.hget(self.key_N,fn)
           utils.save_image_from_base64(b64.decode('utf-8'),'./',fn.decode('utf-8'))
           self.rc.hdel(self.key_N,fn)

def to_file(key_Y,key_N):
    ada = AdafaceWriteResults(key_Y,key_N)
    ada.to_file()

    
if __name__ == '__main__':


    ada = AdafaceWriteResults()
    ada.to_file()

    #AdafaceRecognition.single_re(ada,test_image_path)
    
    
               
        



    

