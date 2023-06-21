import net
import torch
import os
from face_alignment import align
import numpy as np
import pandas as pd


adaface_models = {
    'ir_101': "pretrained/adaface_ir101_webface12m.ckpt",
}


def load_pretrained_model(architecture='ir_101'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(
        adaface_models[architecture], map_location=torch.device('cpu'))['state_dict']
    model_statedict = {key[6:]: val for key,
                       val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    tensor = None
    try:
        np_img = np.array(pil_rgb_image)
        brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
        tensor = torch.tensor([brg_img.transpose(2, 0,1)]).float()
    except Exception :
        return tensor    
    return tensor


def read_vector_pkl(db_path, adaface_model_name):
    """
    @Time    :   2023/06/16 12:10:47
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   读取特征向量文件
                 Args:
                   
                 Returns:
                   df
    """
    import pickle


    file_name = f"representations_adaface_{adaface_model_name}.pkl"
    file_name = file_name.replace("-", "_").lower()
    with open(f"{db_path}/{file_name}", "rb") as f:
            representations = pickle.load(f)
    df = pd.DataFrame(representations, columns=["identity", f"{adaface_model_name}_representation"])
    return df


def build_vector_pkl(db_path, adaface_model_name='adaface_model'):
    """
    @Time    :   2023/06/16 11:40:23
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   构建特征向量文件
                 Args:

                 Returns:
                   void
    """
    import time
    from os import path
    from tqdm import tqdm
    import pickle

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
            desc="生成向量特征文件中：😍😊🔬🔬🔬⚒️⚒️⚒️🎢🎢🎢🎢🎢",
        )
        for index in pbar:
            employee = employees[index]

            img_representation = get_represent(employee)
            instance = []
            instance.append(employee)
            instance.append(img_representation)
            representations.append(instance)

        # -------------------------------

        with open(f"{db_path}/{file_name}", "wb") as f:
            pickle.dump(representations, f)



def get_represent(path):
    """
    @Time    :   2023/06/16 11:54:11
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   获取脸部特征向量
                 Args:
                   
                 Returns:
                   void
    """
    feature = None
    aligned_rgb_img = align.get_aligned_face(path)
    bgr_tensor_input = to_input(aligned_rgb_img)
    if bgr_tensor_input is not None:
        with torch.no_grad():
            encoding, _ = model(bgr_tensor_input)
    else:
       print(f"无法提取脸部特征向量{path}")     
    return feature

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


def demo1():
    model_name = "test"
    build_vector_pkl(test_image_path,model_name)
    df = read_vector_pkl(test_image_path, model_name)
    for index, instance in df.iterrows():
        source_representation = instance[f"{model_name}_representation"]
        #distance = findCosineDistance(source_representation, target_representation)
        print(source_representation)
        features.append(source_representation)
    similarity_scores = torch.cat(features) @ torch.cat(features).T   
    print(similarity_scores)

def find(test_image_path,threshold=0.5):
    """
    @Time    :   2023/06/16 14:02:52
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   根据图片在人脸库比对找人
                 Args:
                   
                 Returns:
                   void
    """
        
    test_representation = get_represent(test_image_path)
    if test_representation  is  not None:
        reset = {}
        for index, instance in df.iterrows():
            source_representation = instance[f"{model_name}_representation"]
            ten = findCosineDistance(source_representation,test_representation)
            reset[ten.item()]= instance["identity"]        
        cosine_similarity =  max(reset.keys())        
        print(f"💝💝💝💝💝💝💝💝💝💝 {cosine_similarity} 💝💝💝💝💝{threshold}")
        return cosine_similarity > threshold ,reset[cosine_similarity]
    else:
        return False,0

def marge(m1,m2):
    from PIL import Image
    import uuid
    # 打开第一张图片
    image1 = Image.open(m1)
    # 打开第二张图片
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
    new_image.save(str(uuid.uuid4()).replace('-', '')+"new_image.jpg")





if __name__ == '__main__':
   
    import imutils 
    from imutils import paths
    import cv2
    import uuid
    model = load_pretrained_model('ir_101')
    #feature, norm = model(torch.randn(2, 3, 112, 112))
    
    test_image_path = 'face_alignment/ser'
    
    features = set()
    model_name = "test_img"
    build_vector_pkl("face_alignment/test",model_name)
    df = read_vector_pkl("face_alignment/test", model_name)

    
    for path in paths.list_images(test_image_path):
        b, r = find(path,0.25)
        if b:
            if r not in features:
                features.add(r)
                marge(r,path)
        else:
            img = cv2.imread(path)
            cv2.imwrite('__not'  + str(uuid.uuid4()).replace('-', '')+".jpg", img)
        


    #similarity_scores = torch.cat(features) @ torch.cat(features).T
    #print(similarity_scores)
     

     

