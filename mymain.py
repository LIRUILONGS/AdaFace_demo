#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   mymain.py
@Time    :   2023/07/05 04:32:45
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   人脸识别Demo 入口
"""

# here put the import lib



import AdafaceBuildVectorPkl as BuildVector
import AdafaceFaceCollection as CollectingFaces
import AdafaceFaceIdentification as Identification
import AdafaceWriteResults as WriteResults


def complete():
    import multiprocessing


    db_path="face_alignment/emp"
    adaface_model_name="adaface_model"
    face_image = "img_dir"
    image_path = "W:\jw"
    
    
    # 构建特征文件
    BuildVector.build_vector_Pkl_file(db_path,adaface_model_name)
    # 进行人脸提取，特征提取
    #CollectingFaces.get_face_detector(image_path,adaface_model_name, face_image)
    process2 = multiprocessing.Process(target=CollectingFaces.get_face_detector, args=(image_path,adaface_model_name))
    process3 = multiprocessing.Process(target=Identification.comparison, args=(db_path,adaface_model_name))
    #Identification.comparison(db_path,adaface_model_name)
    #WriteResults.to_file()

    # 启动进程
    process2.start()
    process3.start()

    # 等待进程结束
    process2.join()
    process3.join()





if __name__ == '__main__':
    complete()
    




