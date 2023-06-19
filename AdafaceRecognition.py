#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   AdafaceRecognition.py
@Time    :   2023/06/17 22:38:21
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   Adaface 人脸识别
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
import requests
import face_yaw_pitc_roll


class AdafaceRecognition:
    __instance = None
    adaface_models = {
        'ir_101': "pretrained/adaface_ir101_webface12m.ckpt",
    }

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, db_path,adaface_model_name='adaface_model',architecture='ir_101'):
        """
        @Time    :   2023/06/17 22:41:40
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   加载模型，特征文件加载
                     Args:
                       
                     Returns:
                       void
        """
        # 模型加载
        self.adaface_model_name = adaface_model_name
        self.db_path = db_path
        self.features = []
        self.load_pretrained_model(architecture)
        self.build_vector_pkl(self.db_path,self.adaface_model_name)
        self.read_vector_pkl(self.db_path, self.adaface_model_name)


    def load_pretrained_model(self,architecture='ir_101'):
        assert architecture in self.adaface_models.keys()
        self.model = net.build_model(architecture)
        statedict = torch.load(
            self.adaface_models[architecture], map_location=torch.device('cpu'))['state_dict']
        model_statedict = {key[6:]: val for key,
                           val in statedict.items() if key.startswith('model.')}
        self.model.load_state_dict(model_statedict)
        self.model.eval()
        print("😍😊😍😊😍😊😍😊😍😊 model 加载完成")


    def to_input(self,pil_rgb_image):
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
            #tensor = torch.tensor([brg_img.transpose(2, 0,1)]).float()
            tensor = torch.tensor(np.array([brg_img.transpose(2, 0,1)])).float()

        except Exception :
            #print("识别图片预处理异常,图片自动忽略")
            pass    
        return tensor    

    def read_vector_pkl(self,db_path, adaface_model_name):
        """
        @Time    :   2023/06/16 12:10:47
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   读取特征向量文件
                     Args:

                     Returns:
                       df
        """
        
        file_name = f"representations_adaface_{adaface_model_name}.pkl"
        file_name = file_name.replace("-", "_").lower()
        with open(f"{db_path}/{file_name}", "rb") as f:
                representations = pickle.load(f)
        self.df = pd.DataFrame(representations, columns=["identity", f"{adaface_model_name}_representation"])
        print("😍😊😍😊😍😊😍😊😍😊 representation  加载完成")


    def build_vector_pkl(self,db_path, adaface_model_name='adaface_model'):
        """
        @Time    :   2023/06/16 11:40:23
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   构建特征向量文件
                     Args:

                     Returns:
                       void
        """
        tic = time.time()

        if os.path.isdir(db_path) is not True:
            raise ValueError("Passed db_path does not exist!")

        file_name = f"representations_adaface_{adaface_model_name}.pkl"
        file_name = file_name.replace("-", "_").lower()
        if path.exists(db_path + "/" + file_name):
            pass
        else:
            employees = []
            for r, _, f in os.walk(db_path):
                for file in f:
                    if (
                        (".jpg" in file.lower())
                        or (".jpeg" in file.lower())
                        or (".png" in file.lower())
                    ):
                        exact_path = r + "/" + file
                        employees.append(exact_path)

            if len(employees) == 0:
                raise ValueError(
                    "没有任何图像在  ",
                    db_path,
                    "  文件夹! 验证此路径中是否存在.jpg或.png文件。",
                )
            representations = []

            pbar = tqdm(
                range(0, len(employees)),
                desc="生成向量特征文件中：⚒️⚒️⚒️",
                mininterval=0.1, 
                maxinterval=1.0, 
                smoothing=0.1,                 
                colour='green',
                postfix=" ⚒️"
            )
            for index in pbar:
                employee = employees[index]

                img_representation = self.get_represent(employee)[0]
                instance = []
                instance.append(employee)
                instance.append(img_representation)
                representations.append(instance)


            with open(f"{db_path}/{file_name}", "wb") as f:
                pickle.dump(representations, f)
            print("😍😊😍😊😍😊😍😊😍😊 representations  构建完成")    


    def find_face(self,test_image_path,threshold=0.5):
        """
        @Time    :   2023/06/16 14:02:52
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   根据图片在人脸库比对找人(单人)
                     Args:

                     Returns:
                       void
        """

        test_representation = self.get_represent(test_image_path)
        if test_representation  is  not None:
            reset = {}
            for index, instance in self.df.iterrows():
                source_representation = instance[f"{self.adaface_model_name}_representation"]
                ten = AdafaceRecognition.findCosineDistance(source_representation,test_representation)
                reset[ten.item()]= instance["identity"]        
            cosine_similarity =  max(reset.keys())        
            return cosine_similarity > threshold ,reset[cosine_similarity]
        else:
            return False,0

    def find_faces(self,test_image_path,threshold=0.5):
        """
        @Time    :   2023/06/18 06:16:19
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   根据图片在人脸库比对找人(多人)
                     Args:
                       
                     Returns:
                       返回比对结果集
        """

        test_representations = self.get_represents(test_image_path)
        res = []
        if test_representations  is  not []:
            pbar = tqdm(
                range(0, len(test_representations)),
                desc="人脸比对中：🔬🔬 ",               
                colour='#f7d8d8',
                postfix="🔬🔬")
            
            for i in pbar:
                test_representation,img =  test_representations[i]
                reset = {}
                for index, instance in self.df.iterrows():
                    source_representation = instance[f"{self.adaface_model_name}_representation"]
                    ten = AdafaceRecognition.findCosineDistance(source_representation,test_representation)
                    reset[ten.item()]= instance["identity"]        
                cosine_similarity =  max(reset.keys())
                res.append((cosine_similarity > threshold ,cosine_similarity,reset[cosine_similarity],img))         
            return res
        else:
            return res

    def get_represent(self,path):
        """
        @Time    :   2023/06/16 11:54:11
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   获取脸部特征向量
                     Args:

                     Returns:
                       返回特性向量
        """

        feature = None
        try:
            if isinstance(path, Image.Image):
               aligned_rgb_img =  path
            else:
                aligned_rgb_img = align.get_aligned_face(path)[0]
        except Exception:
            pass  
            #print(f"无法提取rgb 图像: {path}") 
            return  feature
        bgr_tensor_input = self.to_input(aligned_rgb_img)
        if bgr_tensor_input is not None:
            feature, _ = self.model(bgr_tensor_input)
        else:
            #print(f"无法提取脸部特征向量: {path}")  
            pass 
        return feature
    

    def get_represents(self,path):
        """
        @Time    :   2023/06/18 06:03:09
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   获取脸部特征向量
                     Args:
                       
                     Returns:
                       返回特征向量和对应人脸
        """
        features_t = []
        try:
            aligned_rgb_imgs = align.get_aligned_face(path)
        except Exception:
            pass  
            #print(f"无法提取rgb 图像: {path}") 
            return  features_t
        if aligned_rgb_imgs is not None:
            for aligned_rgb_img in aligned_rgb_imgs:
                bgr_tensor_input = self.to_input(aligned_rgb_img)
                if bgr_tensor_input is not None:
                    feature, _ = self.model(bgr_tensor_input)
                    features_t.append((feature,aligned_rgb_img))
                else:
                    #print(f"无法提取脸部特征向量: {path}")  
                    pass 
        return features_t
    

    
    @staticmethod
    def marge(m1,m2,path):
        """
        @Time    :   2023/06/17 23:00:32
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   图片合并
                     Args:

                     Returns:
                       void
        """
        if isinstance(m1, Image.Image):
            image1 = m1
        else:
            image1 = Image.open(m1)

        if isinstance(m2, Image.Image):
            image2 = m2
        else:
            image2 = Image.open(m2)
        
        # 获取第一张图片的大小
        width1, height1 = image1.size
        # 获取第二张图片的大小
        width2, height2 = image2.size
        # 创建一个新的画布，大小为两张图片的宽度之和和高度的最大值
        new_image = Image.new("RGB", (width1 + width2, max(height1, height2)))
        # 将第一张图片粘贴到画布的左侧
        new_image.paste(image1, (0, 0))
        # 将第二张图片粘贴到画布的右侧
        new_image.paste(image2, (width1, 0))
        # 保存拼接后的图片
        
        new_image.save(path+os.path.basename(m1))    

    @staticmethod
    def findCosineDistance(source_representation, test_representation):
        """
        @Time    :   2023/06/16 12:19:27
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   计算两个向量之间的余弦相似度得分
                     Args:

                     Returns:
                       void
        """
        import torch.nn.functional as F
        return F.cosine_similarity(source_representation, test_representation)
    
    @staticmethod
    def load_image(img):
        exact_image = False
        base64_img = False
        url_img = False

        if type(img).__module__ == np.__name__:
            exact_image = True

        elif img.startswith("data:image/"):
            base64_img = True

        elif img.startswith("http"):
            url_img = True

        # ---------------------------

        if base64_img is True:
            img = AdafaceRecognition.loadBase64Img(img)

        elif url_img is True:
            img = np.array(Image.open(requests.get(img, stream=True, timeout=60).raw).convert("RGB"))

        elif exact_image is not True:  # image path passed as input
            if os.path.isfile(img) is not True:
                raise ValueError(f"Confirm that {img} exists")

            img = cv2.imread(img)

        return img
        
    @staticmethod
    def loadBase64Img(uri):
        encoded_data = uri.split(",")[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img  

    
    def stranger_weight_removal(self,image_path,threshold=0.15):
        """
        @Time    :   2023/06/18 06:47:25
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   陌生人去重
                     Args:
                       
                     Returns:
                       void
        """
        
        test_representation = self.get_represent(image_path)
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
                    colour='#f6b26b',
                    postfix="👽👽")
                
                for i in pbar:
                    instance = self.features[i]
                    ten = AdafaceRecognition.findCosineDistance(instance,test_representation)
                    reset[ten.item()]= instance 
                      
                cosine_similarity =  max(reset.keys())      
                if cosine_similarity > threshold :
                    return True, cosine_similarity   
                else:
                    self.features.append(test_representation)
                    return False, cosine_similarity 
                
        else:
            return False,-1
              
    def stranger_weight_removals(self,image_path,threshold=0.15):
        test_representation = self.get_represent(image_path)
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
                    colour='#f6b26b',
                    postfix="👽👽")
                
                for i in pbar:
                    instance = self.features[i]
                    ten = AdafaceRecognition.findCosineDistance(instance,test_representation)
                    reset[ten.item()]= instance 
                      
                cosine_similarity =  max(reset.keys())      
                if cosine_similarity >= threshold :
                    return True, cosine_similarity   
                else:
                    self.features.append(test_representation)
                    return False, cosine_similarity 
                
        else:
            return False,-1
    
    
    def exec(self,img,threshold=0.5):
        return self.find_face(img,threshold)
    
    
    @staticmethod
    def single_re(ada,test_image_path):
        """
        @Time    :   2023/06/18 06:18:56
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   单人识别
                     Args:
                       
                     Returns:
                       void
        """
        
        f = set()
        while True:
            print("👻👻👻👻🧟‍♀️🧟‍♀️🧟‍♀️🧟‍♀️😀😀😀😀🥶😡🤢😈👽😹🙈🦝",time.time())
            file_paths = list(paths.list_images(test_image_path))
            pbar = tqdm(
                    range(0, len(file_paths)),
                    desc="人脸识别中：👻 ",
                    mininterval=0.1, 
                    maxinterval=1.0, 
                    smoothing=0.01,                 
                    colour='#b6d7a8',
                    postfix=" 👻") 

            for index in pbar:
                    path = file_paths[index]               
                    b, r = ada.find_face(path,0.25)
                    if b:
                        if    r not in f:
                            f.add(r)
                            AdafaceRecognition.marge(r,path,"./")
                    else:
                        img = cv2.imread(path)
                        boo, img = face_yaw_pitc_roll.is_gesture(img,10)
                        if boo:
                            bo,tt  = ada.stranger_weight_removal(path,0.17)
                            if bo:
                                os.remove(path) 
                                continue
                            else:
                                cv2.imwrite(str(tt) +".jpg", img)    
                    os.remove(path) 
            time.sleep(1)        

    @staticmethod
    def multiplayer_re(ada,test_image_path):
        """
        @Time    :   2023/06/18 06:19:17
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   多人识别
                     Args:
                       
                     Returns:
                       void
        """
        
        f ={}
        # 标记人脸识别
        t_  = 0

        while True:
            print("👻👻👻👻🧟‍♀️🧟‍♀️🧟‍♀️🧟‍♀️😀😀😀😀🥶😡🤢😈👽😹🙈🦝",time.time())
            file_paths = list(paths.list_images(test_image_path))
            pbar = tqdm(
                    range(0, len(file_paths)),
                    desc="人脸识别中：👻 ",
                    mininterval=0.1, 
                    maxinterval=1.0, 
                    smoothing=0.01,                 
                    colour='#00ff00',
                    postfix=" 👻") 

            for index in pbar:
                    path = file_paths[index]               
                    data_f_r = ada.find_faces(path,0.18)
                    for  b,c,r,i  in   data_f_r:
                        # 识别成功
                        if b:
                            if r not in f:
                                f[r]=c
                                w, h = i.size
                                i = i.resize((w * 3, h * 3))
                                AdafaceRecognition.marge(r,i,"./")
                            else:
                                if f[r] < c:
                                    w, h = i.size
                                    i = i.resize((w * 3, h * 3))
                                    AdafaceRecognition.marge(r,i,"./")    
                        else:
                            numpy_image = np.array(i)
                            cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                            boo, img = face_yaw_pitc_roll.is_gesture(cv2_image,10)
                            if boo:
                                bo,tt  = ada.stranger_weight_removals(i,0.15)
                                if bo:
                                    continue
                                else:
                                    cv2.imwrite(str(tt) +".jpg", img)    
                    os.remove(path) 
            time.sleep(1)
        
if __name__ == '__main__':


    ada =  AdafaceRecognition(db_path="face_alignment/test",adaface_model_name="adaface_model")
    test_image_path = 'W:\python_code\deepface\\temp\\temp'
    #AdafaceRecognition.single_re(ada,test_image_path)
    AdafaceRecognition.multiplayer_re(ada,test_image_path)
    
               
        



    

