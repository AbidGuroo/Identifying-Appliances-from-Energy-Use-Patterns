## 
import os
import numpy as np
from PIL import Image 


data = []
labels = []


path = "./data/traindata/" 

files= os.listdir(path)

for i in files:
  path = "./data/traindata/{}".format(i)

  images= os.listdir(path)
  for imgfile in images:

    img = Image.open(path+"/"+imgfile)

    img = np.array(img)

    data.append(img)

    labels.append([i])


x_train = np.array(data)
y_train = np.array(labels)
print(x_train.shape)
print(y_train.shape)

## Save Numpy Arrays to be used for Training
np.save('x_train.npy',x_train)
np.save('y_train.npy',y_train)

