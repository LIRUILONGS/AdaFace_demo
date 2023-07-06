#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   AdafaceFaceIdentification.py
@Time    :   2023/06/28 22:10:50
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   Adaface 人脸识别,数据库获取数据识别
"""



# here put the import lib
import os
import numpy as np
import pandas as pd
import pickle
import time
import cv2
from tqdm import tqdm
from PIL import Image
import face_yaw_pitc_roll
import glob
from redis_uits import RedisClient
import utils

class AdafaceFaceIdentification:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, db_path,adaface_model_name='adaface_model'):
        """
        @Time    :   2023/06/17 22:41:40
        @Author  :   liruilonger@gmail.com
        @Version :   3.0
        @Desc    :   初始化处理，加载模型，特征文件加载
        """
        
        self.adaface_model_name = adaface_model_name
        self.db_path = db_path
        self.features = []
        # redis
        self.rc = RedisClient()
        file_name = f"representations_adaface_{adaface_model_name}.pkl"
        self.file_name = file_name.replace("-", "_").lower()
        self.read_vector_pkl()


                                                                                                                                                                                                                                                                                                                                                                                                                                                 

    def read_vector_pkl(self):
        """
        @Time    :   2023/06/16 12:10:47
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   读取特征向量文件
        """

        with open(f"{self.db_path}/{self.file_name}", "rb") as f:
                representations = pickle.load(f)
        self.df = pd.DataFrame(representations, columns=["identity", f"{self.adaface_model_name}_representation"])
        print("😍😊😍😊😍😊😍😊😍😊 特征文件  加载完成")
        return self

    def stranger_weight_removals(self,image,threshold=0.15):
        """
        @Time    :   2023/06/19 02:23:08
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   陌生人去重,人脸信息和
                     Args:
                       image: 去重的image，可以是一个图片路径，也可以是 处理完的特征向量，
                     Returns:
                       void
        """
        

        if  isinstance(image, str):
            test_representation = self.get_represents(image)
        else:
            test_representation = image
        if test_representation  is  not None :
            reset = {}
            if  not self.features :
                self.features.append(test_representation)
                return False, 0
            else:
                pbar = tqdm(
                    range(0, len(self.features)),
                    desc="陌生人归类：👽👽👽 ",
                    mininterval=0.1, 
                    maxinterval=1.0, 
                    smoothing=0.01,                 
                    colour='red',
                    postfix="👽👽")
                
                for i in pbar:
                    instance = self.features[i]
                    ten = findCosineDistance_CPU(instance,test_representation)
                    reset[ten.item()]= instance 
                    # 如果得分大于阈值`0.3`个单位，则比较完成，跳出循环
                    if threshold + (threshold * 0.3)  < ten:
                        break
                      
                cosine_similarity =  max(reset.keys())      
                if cosine_similarity >= threshold :
                    return True, cosine_similarity   
                else:
                    self.features.append(test_representation)
                    return False, cosine_similarity 
                
        else:
            return False,-1
    
    
    
   


def multiplayer_re(ada,threshold=0.39,discard=False,stranger_discard_threshold=0.4,uni_threshold=0.5):
    """
    @Time    :   2023/06/18 06:19:17
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   多人识别
                 Args:
                   
                 Returns:
                   void
    """

    # 识别出的人脸数据 
    faces ={}
    while True:
        
        face_rs =  ada.rc.rpop(ada.adaface_model_name)
        if face_rs is None:
            print(f"😈😈 特征队列 {ada.adaface_model_name} 数据为空",time.time())
            continue
        else:
            reset = {}
            test_representation,i_img = pickle.loads(face_rs)
            for _, instance in ada.df.iterrows():
                source_representation = instance[f"{ada.adaface_model_name}_representation"]
                if source_representation is None:
                    continue
                ten = findCosineDistance_CPU(source_representation,test_representation)
                reset[ten.item()]= instance["identity"]        
                # 如果得分大于阈值`0.3`个单位，则跳出循环，不寻找最大得分
                if threshold + (uni_threshold * threshold)  < ten:
                    break
                cosine_similarity =  max(reset.keys()) #0.4
                # 是否抛弃误差数据
                if discard and (cosine_similarity < threshold) and ( (cosine_similarity <= 0 )or (threshold -  cosine_similarity <=  stranger_discard_threshold)) :
                    pass
                    continue
            # [是否识别到，最大相似度得分，识别到的人路径，人脸image对象，特征值]    
            b,c,r,i,t =(cosine_similarity > threshold ,cosine_similarity,reset[cosine_similarity],i_img,test_representation)
            print("🤢🤢 识别结果",os.path.basename(r)," 最大相似度得分：",cosine_similarity)
            # 识别成功
            if b:
                if r not in faces:
                    faces[r]=c
                    b64 = utils.get_marge_image_to_base64(r,i,"./")
                    ada.rc.hset("face_Y",r,b64)
                    print("👻👻 识别到相似特征",r,time.time())    
                    
                else:
                    if faces[r] < c: 
                        b64 = utils.get_marge_image_to_base64(r,i,"./")
                        ada.rc.hset("face_Y",r,b64)
                        print("👻👻 识别到相似特征",r,time.time())            
            else:
                # 陌生人处理
                if c < 0.2:   
                    bo,tt  = ada.stranger_weight_removals(t,0.35)
                    if bo:
                        # 陌生人存在直接跳过
                        continue
                    else:
                        numpy_image = np.array(i)
                        cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                        boo, img = face_yaw_pitc_roll.is_gesture(cv2_image,40)
                        if boo:
                            b64 = utils.get_Image_to_base64(i)
                            ada.rc.hset("face_N",str(c)+'.jpg',b64)
                            print("👽👽 陌生人",time.time())   
                




def findCosineDistance_CPU(source_representation, test_representation):
    """
    @Time    :   2023/06/16 12:19:27
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   计算两个向量之间的余弦相似度得分，CPU 版本
                 Args:
                 Returns:
                   void
    """
    import torch.nn.functional as F
    import torch
    return F.cosine_similarity(source_representation, test_representation)



def findCosineDistance_GPU(source_representation, test_representation):
    """
    @Time    :   2023/06/28 21:39:48
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   计算两个向量之间的余弦相似度得分，GPU 版本
                 Args:
                   
                 Returns:
                   void
    """
    from torch.nn import DataParallel
    import torch.nn.functional as F
    import torch
    if torch.cuda.device_count() > 1:
        model = DataParallel(F.cosine_similarity)
    else:
        model = F.cosine_similarity
    source_representation = source_representation.cuda()
    test_representation = test_representation.cuda()
    return model(source_representation, test_representation)


def comparison(db_path="face_alignment/test",adaface_model_name="adaface_model",threshold=0.5):
    ada =  AdafaceFaceIdentification(db_path,adaface_model_name)
    multiplayer_re(ada,threshold)



if __name__ == '__main__':


    comparison(db_path="face_alignment/emp",adaface_model_name="adaface_model",threshold=0.30)
    
               
        



    

