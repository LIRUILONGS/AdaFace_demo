import sys
import os

from face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))
#mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(224, 224))
def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face(image_path, rgb_pil_image=None):
    try:

        if rgb_pil_image is None:
            img = Image.open(image_path).convert('RGB')
        else:
            assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
            img = rgb_pil_image
    except Exception:
        return  None         
    # find face
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1000)
        #pbar = tqdm(
        #            range(0, len(faces)),
        #            desc="检测到人脸：😈😈 ",
        #            mininterval=0.1, 
        #            maxinterval=1.0, 
        #            smoothing=0.01,                 
        #            colour='green',
        #            postfix="😈😈")
        #        
        #for i in pbar:
        #    import uuid
        #    faces[i].save(str(i)+ str(uuid.uuid4()).replace('-', '')[0:10]+'.jpg')
        #    print("检测到人脸：",str(i),"😈😈😈😈😈😈😈",faces[i].size)
        face = faces
    except Exception as e:
        #print('Face detection Failed due to error.')
        #print(e)
        face = None

    return face


