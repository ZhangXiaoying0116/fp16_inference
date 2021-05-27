import numpy as np
import scipy.misc
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
from PIL import Image
import os
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"  
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1" 
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"] = "1" 
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LOG_PATH'] ="./amp_log/"

# 加载模型
model_architecture = 'cifar10_architecture.json'
model_weights = 'cifar_weights.h5'
model = model_from_json(open(model_architecture).read())		# 加载模型结构
model.load_weights(model_weights)		# 加载模型权重

# 加载图片
img_names = ['cat.jpg', 'deer.jpg', 'dog.jpg']
imgs_ = [np.transpose(Image.open(img_names[0]).resize((32, 32)), (1, 0, 2)).astype('float32') for img_name in img_names]
imgs = np.array(imgs_) / 255	# 归一化

# 训练
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])		# 编译模型

# 预测样本属于每个类别的概率
print(model.predict(imgs))		# 打印概率
print(np.argmax(model.predict(imgs), axis=1))		# 打印最大概率对应的标签
# 原文链接：https://blog.csdn.net/tszupup/article/details/85275111
