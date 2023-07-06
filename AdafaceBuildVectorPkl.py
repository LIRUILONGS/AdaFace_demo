#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   AdafaceBuildVectorPkl.py
@Time    :   2023/06/29 02:57:25
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   人脸库特征文件构建
"""



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

class AdafaceBuildVectorPkl:
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
        
        self.adaface_model_name = adaface_model_name
        self.db_path = db_path
        self.architecture =architecture
        self.features = []
        file_name = f"representations_adaface_{adaface_model_name}.pkl"
        self.file_name = file_name.replace("-", "_").lower()
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
        statedict = torch.load(
            self.adaface_models[self.architecture], map_location=torch.device('cpu'))['state_dict']
        model_statedict = {key[6:]: val for key,
                           val in statedict.items() if key.startswith('model.')}
        self.model.load_state_dict(model_statedict)
        self.model.eval()
        print("😍😊 模型 加载完成")
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
                md5_new_str = AdafaceBuildVectorPkl.get_dir_md5(self.db_path)
                if md5_old_str != md5_new_str:
                   self.build_vector_pkl_file()   
                else:
                    print(f"😍😊人脸库数据未发生变化，特征文件已生成！{self.db_path}/{self.file_name}")    
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
            img_representation = self.get_represent(employee)
            if img_representation is None:
                raise ValueError(f"图像没有任何人脸在 {employee} ! 验证此路径中是否存文件是否有人脸。", )
            instance = []
            instance.append(employee)
            instance.append(img_representation)
            representations.append(instance)
        AdafaceBuildVectorPkl.rm_suffix_file(self.db_path,"pkl")    
        # 保存特征文件 
        with open(f"{self.db_path}/{self.file_name}", "wb") as f:
            pickle.dump(representations, f)
        print(f"😍😊😍 特征文件 {self.db_path}/{self.file_name} 构建完成")    
        # 保存 人脸数和对应的 文件的 MD5 值

        md5_file = f"{self.db_path}/{el}.md5"
        md5 = str(self.get_dir_md5(self.db_path))
        AdafaceBuildVectorPkl.rm_suffix_file(self.db_path,"md5")    
        with open(md5_file,'w') as f:
            f.write(md5)  
        print(f"😍😊😍 校验文件 {md5_file} 生成完成")
        return self

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
            with torch.no_grad():
                feature, _ = self.model(bgr_tensor_input)
        else:
            #print(f"无法提取脸部特征向量: {path}")  
            pass 
        return feature
    
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
            md5.update(AdafaceBuildVectorPkl.get_file_md5(img_path).encode())
        return md5.hexdigest()
    
    @staticmethod
    def rm_suffix_file(dir_path,suffix): 
        file_paths = glob.glob(os.path.join(dir_path, f"*.{suffix}"))
        for file_path in file_paths:
           os.remove(file_path)

def build_vector_Pkl_file(db_path="face_alignment/test",adaface_model_name="adaface_model"):
    """
    @Time    :   2023/07/03 21:55:16
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   人脸库构建
                 Args:
                 db_path：人脸库位置
                 adaface_model_name：  特征文件名字
                 Returns:
                   void
    """
    ada =  AdafaceBuildVectorPkl(db_path,adaface_model_name)
    ada.build_vector_pkl()
    # 特征值检查
     
    for _, instance in ada.read_vector_pkl().df.iterrows():
        source_representation = instance[f"{ada.adaface_model_name}_representation"]
        if source_representation is None:
            print(f"{instance}特征数据提取失败，请排查照片")



def check_buildVectorPkl():
    """
    @Time    :   2023/07/05 04:39:11
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   对特征文件进行校验
                 Args:
                   
                 Returns:
                   void
    """
    
    import shutil
    ada =  AdafaceBuildVectorPkl(db_path="face_alignment/emp",adaface_model_name="adaface_model")
    for _, instance in ada.read_vector_pkl().df.iterrows():
        source_representation = instance[f"{ada.adaface_model_name}_representation"]
        if source_representation is None:
            print(f"{instance}特征数据提取失败，请排查照片")
            file_path = instance['identity']
            if os.path.exists(file_path):
                dir_path = os.path.dirname(file_path)
                shutil.rmtree(dir_path)

def face_slicing(im_path,ou_path='./temp/'):
    """
    @Time    :   2023/07/05 01:51:38
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   对人脸照片进行人脸切片
                 Args:
                   
                 Returns:
                   void
    """
    for path in  list(paths.list_images(im_path)):
        faces = align.get_aligned_face(path,None,1)
        print(f"人脸切片： {path}")
        for _,face in enumerate(faces):
            face.save(ou_path+ os.path.basename(path)) 






if __name__ == '__main__':


    build_vector_Pkl_file(db_path="face_alignment/emp",adaface_model_name="adaface_model")
    #face_slicing("C:\\Users\\liruilong\\Documents\\GitHub\\AdaFace_demo\\face_alignment\\emp")

    #AdafaceRecognition.single_re(ada,test_image_path)
    
    
               
        



    

