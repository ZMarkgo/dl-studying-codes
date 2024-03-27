# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 20:13:16 2020

@author: admin

"""
from matplotlib import pyplot as plt
from PIL import Image

#读入图像
photo = Image.open('莲花.jpg')

#改变大小
print(photo.size)
photo=photo.resize([128,128])
print(photo.size)
print(photo)

#矩阵转换
import numpy as np
Im=np.array(photo)
print(Im.shape)
print(Im[:,:,1])

#尺度变化
Im=Im/255
print(Im[:,:,0])

#图像展示
plt.imshow(Im)

#图像的代数运算
Im1=Im+0.5
Im2=1-Im
Im3=0.5*Im
Im4=Im/0.5
plt.figure()
fig,ax=plt.subplots(1,4)
fig.set_figwidth(15)
ax[0].imshow(Im1)
ax[1].imshow(Im2)
ax[2].imshow(Im3)
ax[3].imshow(Im4)

