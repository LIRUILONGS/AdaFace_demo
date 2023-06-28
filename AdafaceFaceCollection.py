#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   AdafaceFaceCollection.py
@Time    :   2023/06/28 04:27:44
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   AdafaceFaceCollection 人脸检测读取指定文件夹，写入数据库
"""


# here put the import lib
import net
import torch
import os
import base64
from pathlib import Path
from face_alignment import align
import numpy as np
import pandas as pd
import pickle
import time
import imutils
from imutils import paths
import cv2
import uuid
from os import path
from tqdm import tqdm
from PIL import Image
import hashlib
import glob
import base64
import utils
import multiprocessing
import concurrent.futures
import logging
import functools
from concurrent.futures import ThreadPoolExecutor
from redis_uits import RedisClient
import threading



logging.basicConfig(level=logging.DEBUG)

class AdafaceFaceCollection:
    __instance = None
    adaface_models = {
        'ir_101': "pretrained/adaface_ir101_webface12m.ckpt",
    }

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, face_path, adaface_model_name='adaface_model', architecture='ir_101', face_image='img_dir'):
        """
        @Time    :   2023/06/17 22:41:40
        @Author  :   liruilonger@gmail.com
        @Version :   3.0
        @Desc    :   初始化处理，加载模型，特征文件加载
        """
        # 生成的特征的 模型 K，对应数据库特征列表名称
        self.adaface_model_name = adaface_model_name

        # 采集的的文件位置
        self.face_path = face_path

        # 使用的模型文件
        self.architecture = architecture

        # 存放图片的 set
        self.face_image = face_image

        # redis
        self.rc = RedisClient()

        self.load_pretrained_model()

    def load_pretrained_model(self):
        """
        @Time    :   2023/06/19 23:50:58
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   加载模型
                     Args:
                       
                     Returns:
                       void
        """

        assert self.architecture in self.adaface_models.keys()
        self.model = net.build_model(self.architecture)
        statedict = torch.load(self.adaface_models[self.architecture], map_location=torch.device('cpu'))['state_dict']
        model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
        self.model.load_state_dict(model_statedict)
        self.model.eval()
        print("😍😊😍😊😍😊😍😊😍😊 模型 加载完成")
        return self


def to_input(self, pil_rgb_image):
    """
    @Time    :   2023/06/17 22:42:30
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   识别图片预处理
                 Args:
                   
                 Returns:
                   void
    """
    tensor = None
    try:
        np_img = np.array(pil_rgb_image)
        brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
        # tensor = torch.tensor([brg_img.transpose(2, 0,1)]).float()
        tensor = torch.tensor(np.array([brg_img.transpose(2, 0, 1)])).float()
    except Exception:
        # print("识别图片预处理异常,图片自动忽略")
        pass
    return tensor


def build_vector_db(self):
    """
    @Time    :   2023/06/19 23:02:16
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   提取人脸特征到数据库
    """
    employees = list(paths.list_images(self.face_path))
    # 根据CPU 核数对文件列表做处理
    cpu_count = multiprocessing.cpu_count()
    print(cpu_count,len(employees) )
    # 这里的 4 为 单个进程需要的核数
    ps = cpu_count// 3
    chunk_size = len(employees) // ps
    print(chunk_size) 
    image_chunks = [employees[i:i+chunk_size] for i in range(0, len(employees), chunk_size)]
    temp_employees = []
    for i in range(len(image_chunks)):
      flattened_list = [item for sublist in image_chunks[i:] +image_chunks[:i] for item in sublist]
      temp_employees.append((self,flattened_list)) 
       
    

    with ThreadPoolExecutor(max_workers=ps,thread_name_prefix="face_thread_") as executor:
        futures = [executor.submit(face_detector, *params) for params in temp_employees]
        results = [future.result() for future in futures]
        print(futures)


def face_detector(self,employees):
        """
        @Time    :   2023/06/28 05:48:12
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   根据文件列表进行人脸检测
                     Args:
                       人脸列表
                     Returns:
                       void
        """

        el = len(employees)
        if el == 0:
            raise ValueError("没有任何图像在  ", self.face_path, "  文件夹! 验证此路径中是否存在 .jpg 或 .png 文件。")
        pbar = tqdm(
            range(0, el),
            desc="采集特征中: "+ threading.current_thread().name +"：⚒️⚒️⚒️",
            mininterval=0.1,
            maxinterval=1.0,
            smoothing=0.1,
            colour='green',
            postfix=" ⚒️"
        )
        for index in pbar:
            employee = employees[index]
            md5_str = None
            try:
                md5_str = utils.get_file_md5(employee)
            except:
                print(f"读取文件失败{employee}")
                continue
            # 如果处理过直接跳出去
            if md5_str is not None:
                if self.rc.sismember(self.face_image, md5_str):
                    print(f"处理过的图片：{employee}")
                    continue
                else:
                    self.rc.sadd(self.face_image, md5_str)

            img_representation = get_represents(self,employee)
            if img_representation is not []:
                pbar = tqdm(
                    range(0, len(img_representation)),
                    desc="单张人脸采集中：🔬🔬 ",
                    colour='red',
                    postfix="🔬🔬")
                for i in pbar:
                    my_tuple = img_representation[i]
                    self.rc.rpush(self.adaface_model_name, (pickle.dumps(my_tuple)))

        return self

def get_represents(self, path):
        """
        @Time    :   2023/06/18 06:03:09
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   获取多个脸部特征向量
                     Args:
                       
                     Returns:
                       features_t --> [(特征向量,人脸 Image.Image)]   返回特征向量和对应人脸 Image.Image 对象 的 list
        """
        features_t = []
        try:
            aligned_rgb_imgs = align.get_aligned_face(path)
        except Exception:
            pass
            # print(f"无法提取rgb 图像: {path}")
            return features_t
        if aligned_rgb_imgs is not None:
            for aligned_rgb_img in aligned_rgb_imgs:
                bgr_tensor_input = to_input(self,aligned_rgb_img)
                if bgr_tensor_input is not None:
                    with torch.no_grad():
                        feature, _ = self.model(bgr_tensor_input)

                    features_t.append((feature, aligned_rgb_img))
                else:
                    # print(f"无法提取脸部特征向量: {path}")
                    pass
        return features_t
    

def face_detector_warpper(cls_instance, employees):
    print("face_detector_warpper")
    return cls_instance.face_detector(employees)
 

if __name__ == '__main__':
    test_image_path = 'W:\python_code\deepface\\temp\\temp'
    face_c = AdafaceFaceCollection(face_path=test_image_path, adaface_model_name="adaface_model", face_image="img_dir")
    test_image_path = 'W:\python_code\deepface\\temp\\temp'

    build_vector_db(face_c)
    #face_c.test()
