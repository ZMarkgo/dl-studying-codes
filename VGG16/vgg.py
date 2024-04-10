# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:33:17 2024

@author: PC
"""

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions  
from tensorflow.keras.preprocessing import image  
  
# 加载预训练的VGG16模型（不包括顶部的全连接层）  
model = VGG16(weights='imagenet', include_top=False)  
  
# 加载要分类的图像  
img_path = "dog.jpg"  # 替换为你的图像路径  
img = image.load_img(img_path, target_size=(224, 224))  # VGG16的输入大小为224x224  
x = image.img_to_array(img)  # 将图像转换为NumPy数组  
x = np.expand_dims(x, axis=0)  # 添加一个维度以匹配模型的输入形状  
x = preprocess_input(x)  # 对图像进行预处理，以满足VGG16的输入要求  
  
# 使用模型进行预测  
features = model.predict(x)  
# 如果你想要对特征进行进一步的处理或分类（例如，使用顶部的全连接层进行分类），  
# 你可以加载完整的VGG16模型（include_top=True），并使用decode_predictions来解码预测结果。  
complete_model = VGG16(weights='imagenet', include_top=True)  
predictions = complete_model.predict(x)  
decoded_predictions = decode_predictions(predictions, top=1)  # 获取前3个最可能的类别 
print(decoded_predictions)