#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2023/06/28 04:32:11
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   None
"""

# here put the import lib

import os
import pickle
import torch
import torch.distributed as dist
from urllib.parse import urlparse
import uuid
import requests
import base64
import numpy as np
import cv2 
from io import BytesIO
import zipfile
import hashlib
from imutils import paths
import glob
from PIL import Image



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm

def fuse_features_with_norm(stacked_embeddings, stacked_norms):

    assert stacked_embeddings.ndim == 3 # (n_features_to_fuse, batch_size, channel)
    assert stacked_norms.ndim == 3 # (n_features_to_fuse, batch_size, 1)
    
    pre_norm_embeddings = stacked_embeddings * stacked_norms
    fused = pre_norm_embeddings.sum(dim=0)
    fused, fused_norm = l2_norm(fused, axis=1)

    return fused, fused_norm 


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    local_rank = get_local_rank()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(local_rank)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=torch.device("cuda", local_rank))
    size_list = [torch.tensor([0], device=torch.device("cuda", local_rank)) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8,
                            device=torch.device("cuda", local_rank)))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8,
                            device=torch.device("cuda", local_rank))
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def get_num_class(hparams):
    # getting number of subjects in the dataset
    if hparams.custom_num_class != -1:
        return hparams.custom_num_class

    if 'faces_emore' in hparams.train_data_path.lower():
        # MS1MV2
        class_num = 70722 if hparams.train_data_subset else 85742
    elif 'ms1m-retinaface-t1' in hparams.train_data_path.lower():
        # MS1MV3
        assert not hparams.train_data_subset
        class_num = 93431
    elif 'faces_vgg_112x112' in hparams.train_data_path.lower():
        # VGGFace2
        assert not hparams.train_data_subset
        class_num = 9131
    elif 'faces_webface_112x112' in hparams.train_data_path.lower():
        # CASIA-WebFace
        assert not hparams.train_data_subset
        class_num = 10572
    elif 'webface4m' in hparams.train_data_path.lower():
        assert not hparams.train_data_subset
        class_num = 205990
    elif 'webface12m' in hparams.train_data_path.lower():
        assert not hparams.train_data_subset
        class_num = 617970
    elif 'webface42m' in hparams.train_data_path.lower():
        assert not hparams.train_data_subset
        class_num = 2059906
    else:
        raise ValueError('Check your train_data_path', hparams.train_data_path)

    return class_num


def is_valid_url(url):
    """
    @Time    :   2023/05/29 21:49:19
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   url  字符串 校验
                 Args:
                   url
                 Returns:
                   booler
    """
    
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_uuid():
    """
    @Time    :   2023/05/29 21:50:16
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   生成 UUID
                 Args:
                   
                 Returns:
                   string
    """
    
    return str(uuid.uuid4()).replace('-', '')


def get_img_url_base64(url):
    """
    @Time    :   2023/05/29 21:50:42
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   图片 url 解析为 base64 编码
                 Args:
                   url
                 Returns:
                   base64_bytes
    """
    response = requests.get(url)
    image_bytes = response.content
    base64_bytes = base64.b64encode(image_bytes)
    return base64_bytes.decode('utf-8')

def get_base64_to_img(base64_str):
    """
    @Time    :   2023/05/29 21:51:23
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   base64 编码转化为 opencv img 对象
                 Args:
                   
                 Returns:
                   void
    """
    
    # 从 base64 编码的字符串中解码图像数据
    img_data = base64.b64decode(base64_str)
    # 将图像数据转换为 NumPy 数组
    nparr = np.frombuffer(img_data, np.uint8)
    # 解码图像数组
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_img_to_base64(img):
    """
    @Time    :   2023/05/29 21:54:26
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   opencv img 对象转化为 base64 编码
                 Args:
                   
                 Returns:
                   void
    """
    img_b64 = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')

    return img_b64

def get_b64s_and_make_to_zip(b64s_mark,img_id):
    # 创建一个名为 'images.zip' 的 zip 文件
    with zipfile.ZipFile(img_id+'_images.zip', 'w') as zip_file:
    # 遍历字典中的每个图像
        for img_b64,img_name in b64s_mark:
            # 将 base64 编码的数据解码为二进制数据
            img_data_binary = base64.b64decode(img_b64)
            # 将图像数据写入 zip 文件中
            zip_file.writestr(img_name+"_" +get_uuid()+ '.jpg', img_data_binary)

    #return send_file("..\\"+img_id+'_images.zip', as_attachment=True)


def  build_img_text_marge(img_,text):
    """
    @Time    :   2023/06/01 05:29:09
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   生成文字图片拼接到 img 对象
                 Args:
                   
                 Returns:
                   void
    """
    # 创建一个空白的图片
    img = np.zeros((500, 500, 3), dtype=np.uint8)

    # 设置字体和字号
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2

    # 在图片上写入文字
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness=2)
    text_x = (500 - text_size[0]) // 2
    text_y = (500 + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness=2)
    montage_size = (300, 400)
    montages = cv2.build_montages([img_,img], montage_size, (1, 2))



    # 保存图片
    return montages

    
def image_to_base64(image):
    """
    @Time    :   2023/06/28 02:47:28
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   Image.image 对象转化为 base64 编码
                 Args:
                   
                 Returns:
                   void
    """
    
    # 将图片转换为 base64 编码的字符串
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_str


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

def get_dir_md5(dir_path):
    """
    @Time    :   2023/06/19 23:26:20
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   获取目录图片文件的MD5
                 Args:
                   dir_path: 目录路径
                 Returns:
                   void
    """
    md5 = hashlib.md5()
    for img_path  in paths.list_images(dir_path):
        md5.update(get_file_md5(img_path).encode())
    return md5.hexdigest()

def rm_suffix_file(dir_path,suffix="jpg"): 
    """
    @Time    :   2023/06/27 23:18:51
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   删除指定后缀的文件
                 Args:
                   
                 Returns:
                   void
    """
    if isinstance(dir_path, str): 
        file_paths = glob.glob(os.path.join(dir_path, f"*.{suffix}"))
    else:
        file_paths = dir_path
    for file_path in file_paths:
        os.remove(file_path)



def get_marge_image_to_base64(m1,m2,path,is_bash64=True):
    """
    @Time    :   2023/06/17 23:00:32
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   Image.Image 图片合并并转化为 Base64
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
    if is_bash64:
        return get_Image_to_base64(new_image)
    else:
        new_image.save(path+os.path.basename(m1)) 

def get_Image_to_base64(image_):
    """
    @Time    :   2023/07/03 06:24:15
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   Image.Image 图片对象转化为 Base64
                 Args:
                   
                 Returns:
                   void
    """
    if isinstance(image_, Image.Image):
        image = image_
    else:
        image = Image.open(image_)

    image_stream = BytesIO()
    image.save(image_stream,format='PNG')
    image_stream.seek(0)
    base64_data = base64.b64encode(image_stream.read()).decode('utf-8')
    return base64_data


def save_image_from_base64(b64_str, op,fn):
    """
    @Time    :   2023/07/04 00:01:36
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   b64 保存为图片
                 Args:
                   
                 Returns:
                   void
    """
    print(op,fn)
    image_data = base64.b64decode(b64_str)
    image = Image.open(BytesIO(image_data))
    
    image.save(os.path.join(op, os.path.basename(fn)))