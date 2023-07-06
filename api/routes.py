#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   routes.py
@Time    :   2023/07/03 04:55:41
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   人脸识别路由层
"""

# here put the import lib


from flask import Blueprint, request, Flask, jsonify, request, render_template,send_file
import service
import base64
import numpy as np
import cv2
import imutils

import os
from io import BytesIO
from redis_uits import RedisClient 
import json

#  pip  installed redis-4.5.5

blueprint = Blueprint("routes", __name__)
rc = RedisClient()

MEMORY = True



@blueprint.route("/")
def home():
    return "<h1>Welcome to DeepFace API!</h1>"

@blueprint.route("/initdate",methods=["GET"])
def initdate():
    """
    @Time    :   2023/05/29 05:16:21
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   初始化检测的数据
                 Args:
                   
                 Returns:
                   void
    """
    lists = []
    y =  list(rc.hgetall("face_Y"))
    data = [json.loads(value.decode('utf-8')) for _ , value in rc.hgetall("face_Y")]
    print(y)
    n = rc.hgetall("face_N")
    lists.append( )
    
    if lists :
        return jsonify({"status":200,"data": data})
    








@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img1_path = input_args.get("img1_path")
    img2_path = input_args.get("img2_path")

    if img1_path is None:
        return {"message": "you must pass img1_path input"}

    if img2_path is None:
        return {"message": "you must pass img2_path input"}

 
    return 0


