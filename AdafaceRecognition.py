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
import hashlib
import glob

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
        @Version :   3.0
        @Desc    :   初始化处理，加载模型，特征文件加载
        """
        # 模型加载
        self.adaface_model_name = adaface_model_name
        self.db_path = db_path
        self.architecture =architecture
        self.features = []
        file_name = f"representations_adaface_{adaface_model_name}.pkl"
        self.file_name = file_name.replace("-", "_").lower()
        self.load_pretrained_model().build_vector_pkl().read_vector_pkl()


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
        statedict = torch.load(
            self.adaface_models[self.architecture], map_location=torch.device('cpu'))['state_dict']
        model_statedict = {key[6:]: val for key,
                           val in statedict.items() if key.startswith('model.')}
        self.model.load_state_dict(model_statedict)
        self.model.eval()
        print("😍😊😍😊😍😊😍😊😍😊 模型 加载完成")
        return self


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
        


    def build_vector_pkl(self):
        """
        @Time    :   2023/06/16 11:40:23
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   构建特征向量文件
                     构建前会做一个人脸文件数量和文件的 MD5 值匹配，自动更新
        """

        if os.path.isdir(self.db_path) is not True:
            raise ValueError("人脸库文件不存在!")

        if path.exists(f"{self.db_path}/{self.file_name}"):
            # 特征文件存在的情况
            il =  len(list(paths.list_images(self.db_path)))
            md5_file = f"{self.db_path}/{il}.md5"
    
            if path.exists(md5_file):
               # 判断 MD5
               md5_old_str = ''
               with open(md5_file,'r') as f:
                    md5_old_str = f.read()
               md5_new_str = AdafaceRecognition.get_dir_md5(self.db_path)
               if md5_old_str != md5_new_str:
                   self.build_vector_pkl_file() 
            else:
               # 校验文件不存在
               self.build_vector_pkl_file() 
        else:
            # 特征文件不存在
            self.build_vector_pkl_file()
        return self    
    

    def build_vector_pkl_file(self):
        """
        @Time    :   2023/06/19 23:02:16
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   特征文件生成
        """
        
        employees = list(paths.list_images(self.db_path))
        el = len(employees)
        if el == 0:
            raise ValueError("没有任何图像在  ", self.db_path,  "  文件夹! 验证此路径中是否存在.jpg或.png文件。", )
        representations = []
        pbar = tqdm(
            range(0, el),
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
        AdafaceRecognition.rm_suffix_file(self.db_path,"pkl")    
        # 保存特征文件 
        with open(f"{self.db_path}/{self.file_name}", "wb") as f:
            pickle.dump(representations, f)
        print(f"😍😊😍😊😍😊😍😊😍😊 特征文件 {self.db_path}/{self.file_name} 构建完成")    
        # 保存 人脸数和对应的 文件的 MD5 值

        md5_file = f"{self.db_path}/{el}.md5"
        md5 = str(self.get_dir_md5(self.db_path))
        AdafaceRecognition.rm_suffix_file(self.db_path,"md5")    
        with open(md5_file,'w') as f:
            f.write(md5)  
        print(f"😍😊😍😊😍😊😍😊😍😊 校验文件 {md5_file} 生成完成")
        return self


    def find_face(self,test_image_path,threshold=0.5,stranger_discard_threshold=0.1  ):
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
                        b -->   识别结果              cosine_similarity > threshold 
                        c -->   比对完的最大相似度得分 cosine_similarity
                        r -->   对应的人脸库人脸位置   reset[cosine_similarity]
                        i -->   比对的图片            img
                        t -->   对比图片的特征值       test_representation

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
                    # 如果得分大于阈值`0.3`个单位，则比较完成，跳出循环
                    if threshold + (0.3 * threshold)  < ten:
                        break
                cosine_similarity =  max(reset.keys())
                res.append((cosine_similarity > threshold ,cosine_similarity,reset[cosine_similarity],img,test_representation))         
            return res
        else:
            return res

    def get_represent(self,path):
        """
        @Time    :   2023/06/16 11:54:11
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   获取单个脸部特征向量
                     Args:
                        path: 可以是图片路径，获取第一个人脸的特征向量，也可以是 Image.Image 对象
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
                    colour='#f6b26b',
                    postfix="👽👽")
                
                for i in pbar:
                    instance = self.features[i]
                    ten = AdafaceRecognition.findCosineDistance(instance,test_representation)
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
    
    
    def load_memory_db(self):
        """
        @Time    :   2023/06/18 23:21:52
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   加载内存中存在的识别数据
        """
        m_db_f  = f"{self.db_path}/M_{self.file_name}"
        if path.exists(m_db_f):
            pass
            with open(m_db_f, "rb") as f:
                representations = pickle.load(f)
            
        else:
            print("内存特征文件未保存!")


    def save_memory_db(self):
        """
        @Time    :   2023/06/18 23:37:37
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   保存内存中存在的识别数据
        """
        m_db_f  = f"{self.db_path}/M_{self.file_name}"
        if path.exists(m_db_f): 
            os.remove(m_db_f)
             

        else:
            print("内存特征文件未保存!")
            pass
            with open(m_db_f, "rb") as f:
                representations = pickle.load(f)
    
    def exec(self,img,threshold=0.5):
        return self.find_face(img,threshold)
    
    
    
    @staticmethod
    def marge(m1,m2,path):
        """
        @Time    :   2023/06/17 23:00:32
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   图片合并
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

    
    @staticmethod
    def get_file_md5(file_path):
        """
        @Time    :   2023/06/19 21:48:31
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   获取文件 MD5
                     Args:
                       file_path：str 文件路径
                     Returns:
                       MD5 对象的十六进制表示形式
        """
        
        with open(file_path, 'rb') as f:
            md5_obj = hashlib.md5()
            while True:
                data = f.read(4096)
                if not data:
                    break
                md5_obj.update(data)
        return md5_obj.hexdigest()
    
    @staticmethod
    def get_dir_md5(dir_path):
        """
        @Time    :   2023/06/19 23:26:20
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   None
                     Args:
                       dir_path: 目录路径
                     Returns:
                       void
        """
        
        md5 = hashlib.md5()
        for img_path  in paths.list_images(dir_path):
            md5.update(AdafaceRecognition.get_file_md5(img_path).encode())
        return md5.hexdigest()
    
    @staticmethod
    def rm_suffix_file(dir_path,suffix): 
        file_paths = glob.glob(os.path.join(dir_path, f"*.{suffix}"))
        for file_path in file_paths:
           os.remove(file_path)


    @staticmethod
    def multiplayer_re(ada,test_image_path,is_memory_db=False):
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
                    # 0.18               
                    data_f_r = ada.find_faces(path,0.4)
                    pbar = tqdm(
                        range(0, len(data_f_r)),
                        desc="识别结果归类：🎉🎉🎉 ",
                        mininterval=0.1, 
                        maxinterval=1.0, 
                        smoothing=0.01,                 
                        colour='#f6b26b',
                        postfix="🎉🎉🎉")
                    for  index  in   pbar:
                        b,c,r,i,t = data_f_r[index]
                        # 识别成功
                        if b:
                            if r not in faces:
                                faces[r]=c
                                AdafaceRecognition.marge(r,i,"./")
                            else:
                                if faces[r] < c: 
                                    AdafaceRecognition.marge(r,i,"./")    
                        else:
                            numpy_image = np.array(i)
                            cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                            boo, img = face_yaw_pitc_roll.is_gesture(cv2_image,10)
                            if boo:
                                # 0.15
                                bo,tt  = ada.stranger_weight_removals(t,0.4)
                                if bo:
                                    continue
                                else:
                                    cv2.imwrite(str(tt) +".jpg", cv2_image)    
                    os.remove(path) 
            time.sleep(1)
        
if __name__ == '__main__':


    ada =  AdafaceRecognition(db_path="face_alignment/test",adaface_model_name="adaface_model")
    test_image_path = 'W:\python_code\deepface\\temp\\temp'

    #AdafaceRecognition.single_re(ada,test_image_path)
    AdafaceRecognition.multiplayer_re(ada,test_image_path)
    
               
        



    

