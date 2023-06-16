#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   demo.py
@Time    :   2023/06/16 08:34:52
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   None
"""

# here put the import lib


from face_alignment import align
from inference import load_pretrained_model, to_input

model = load_pretrained_model('adaface_ir101_webface12m.ckpt')
path = '.new_find.jpg'
aligned_rgb_img = align.get_aligned_face(path)
bgr_input = to_input(aligned_rgb_img)
feature, _ = model(bgr_input)

