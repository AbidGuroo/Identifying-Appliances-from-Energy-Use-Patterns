## Vertical Merge of Input Voltage and Current Images
import cv2
import numpy as np
import os
import csv

## Training Data

router = './data/train/'
partitioneddir = './data/traindata/'


with open('./data/train_labels.csv') as labelfile:

    if not os.path.exists(partitioneddir):
        os.mkdir(partitioneddir)


    rows = csv.reader(labelfile)

    for row in rows:
        try:
            files = '{}/{}'.format(router,row[0]) 
            labels  = partitioneddir + '{}'.format(row[1])
            if row[0] == 'id':
                continue
            if not os.path.exists(labels):
                os.mkdir(labels)
            img1 = cv2.imread(files+'_c.png')
            img2 = cv2.imread(files+'_v.png')

            image = np.concatenate((img1, img2))
            cv2.imwrite('{}/{}.png'.format(labels,row[0]), image)

        except:
            continue


## Test Data

router = './data/test/'
partitioneddir = './data/testdata/'


with open('./data/submission_format.csv') as labelfile:

    if not os.path.exists(partitioneddir):
        os.mkdir(partitioneddir)

    rows = csv.reader(labelfile)

    for row in rows:
        try:
            files = '{}/{}'.format(router,row[0])
            img1 = cv2.imread(files+'_c.png')
            img2 = cv2.imread(files+'_v.png')
            image = np.concatenate((img1, img2))
            cv2.imwrite('{}/{}.png'.format(partitioneddir,row[0]), image)

        except:
            continue
