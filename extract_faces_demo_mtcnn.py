# !/usr/bin/env python
# -*- encoding: utf-8 -*-

# pip install deepface==0.0.79
# pip install imutils==0.5.4

"""
@File    :   extract_faces_demo_mtcnn.py
@Time    :   2023/05/20 03:11:14
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   deepface 人脸提取 extract_faces  demo 
"""

# here put the import lib

from deepface import DeepFace
import cv2
import imutils
from decimal import Decimal
from imutils import paths
import os
import face_yaw_pitc_roll
import uuid

import inference



def extract_faces_all(img_path):
    """
    @Time    :   2023/05/20 03:50:07
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   extract_faces 用于对图像进行特征分析，提取头像坐标，
                    在实际使用中，如果对精度有要求，可以通过 `confidence` 来对提取的人脸进行过滤，
                 Args:
                 extract_faces方法接受以下参数：
                    - img_path：要从中提取人脸的图像路径、numpy数组（BGR）或base64编码的图像。
                    - target_size：人脸图像的最终形状。将添加黑色像素以调整图像大小。
                    - detector_backend：人脸检测后端可以是retinaface、mtcnn、opencv、ssd或dlib。
                    - enforce_detection：如果在提供的图像中无法检测到人脸，则该函数会引发异常。如果不想得到异常并仍要运行该函数，则将其设置为False。
                    - align：根据眼睛位置对齐。
                    - grayscale：以RGB或灰度提取人脸。

                 Returns:
                   返回一个包含人脸图像、人脸区域和置信度的字典列表。其中，
                   - face 键对应的值是提取的人脸图像
                   - facial_area 键对应的值是人脸在原始图像中的位置和大小
                   - confidence 键对应的值是人脸检测的置信度

    """
    # img_path = "huge_1.jpg"

    # 读取原始图像
    image = cv2.imread(img_path)
    rst = None
    try:
        rst = DeepFace.extract_faces(
            img_path=image,
            target_size=(224, 224),
            detector_backend="dlib",
            enforce_detection=True,
            align=True,
            grayscale=False)
    except Exception as e:
        print(e)
        # print("解析错误的图片：",img_path)
        # file_name = os.path.basename(img_path)
        # cv2.imwrite("C:\putot\\"+file_name, image)
        # print("保存解析错误的图：","C:\putot\\"+file_name)
        return

    # print(rst)
    # if face_detector_obj in None:
    #     print("数据为空：",face_detector_obj)

    for i, f in enumerate(rst):
        print('😊'.rjust(i * 2, '😊'))
        #print("编号：", i, '\n', " 检测人脸位置:", f['facial_area'], '\n', " 置信度:", f['confidence'])
        x, y, w, h = f['facial_area'].values()
        x1, y1, x2, y2 = x, y, x + w, y + h
        # 根据不同的置信度做不同标记
        confidence = Decimal(str(f['confidence']))
        best = Decimal('1')
        color = (0, 255, 0)

        abs = best - confidence
        if 0.001 <= abs < 0.05:
            color = (255, 0, 255)
        elif 0.08 > abs >= 0.05:
            color = (0, 165, 255)
        elif abs >= 0.08:
            color = (255, 255, 255)
        else:
            # 这个精度的考虑用于识别,切片保存

            cropped_img = image[y:y + h, x:x + w]
            # 对切片进行等比放大
            cropped_img = imutils.resize(cropped_img, width=224)
            boo, img = face_yaw_pitc_roll.is_gesture(cropped_img,15)
            if boo:
                cv2.imwrite('temp\\cf_' + str(i) + str(uuid.uuid4()).replace('-', '')+".jpg", img)
            else:
                continue
            #cv2.imwrite('new_' + str(i) + img_path, cropped_img)
            pass
        # 根据坐标标记图片,标记框的左上角和右下角的坐标,
        #cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        # 添加 置信度标签
        #cv2.putText(image, format(f['confidence'], '0.4f'), (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1,cv2.LINE_4)
        #file_name = os.path.basename(img_path)

        #cv2.imwrite('C:\putot\\new_\\' + file_name, image)
        #print("保存图片位置："+'C:\putot\\new_\\__' + file_name)

# if __name__ == '__main__':
#     for path in paths.list_images("W:\photo\\b1205"):
#         print(path)
#         extract_faces_all(path)

def face_recognition(file_path):
    """
    @Time    :   2023/05/26 00:02:22
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   并行处理
                 Args:
                   file_path
                 Returns:
                   void
    """
    files =  paths.list_images(file_path)
    from concurrent import futures
    with futures.ProcessPoolExecutor(3) as pool:
        while pool.map(extract_faces_all, files):
            pass
        print("💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀")


def for_m():
    for  path  in paths.list_images("W:\\back_20230522\\"):
        print(path)
        extract_faces_all(path)

if __name__ == '__main__':
    
    face_recognition("W:\\back_20230522\\")
    

