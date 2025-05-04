
# import os
# import sys
# import pandas as pd
# import pickle as pkl
# import numpy
# from scipy.misc import imread, imresize, imsave

# image_path='C:\\Users\\msraspeech\\PycharmProjects\\v-honwa\\off_image_test\\off_image_test\\'
# outFile='offline-test.pkl'
# oupFp_feature=open(outFile,'wb')

# features={}

# channels=1

# sentNum=0

# scpFile=open('C:\\Users\\msraspeech\\PycharmProjects\\v-honwa\\test_caption.txt')
# while 1:
#     line=scpFile.readline().strip() # remove the '\r\n'
#     if not line:
#         break
#     else:
#         key = line.split('\t')[0]
#         image_file = image_path + key + '_' + str(0) + '.bmp'
#         im = imread(image_file)
#         mat = numpy.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')
#         for channel in range(channels):
#             image_file = image_path + key + '_' + str(channel) + '.bmp'
#             im = imread(image_file)
#             mat[channel,:,:] = im
#         sentNum = sentNum + 1
#         features[key] = mat
#         if sentNum / 500 == sentNum * 1.0 / 500:
#             print('process sentences ', sentNum)

# print('load images done. sentence number ',sentNum)

# pkl.dump(features,oupFp_feature)
# print('save file done')
# oupFp_feature.close()
import os
import sys
import pandas as pd
import pickle as pkl
import numpy
import imageio
from PIL import Image
"C:\Users\Pragya Pandey\Downloads\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\off_image_test\off_image_test"
# Define paths and initialize variables
image_path = 'C:\\Users\\Pragya Pandey\Downloads\\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\\v-honwa\\off_image_test\\off_image_test\\'
outFile = 'offline-test.pkl'
oupFp_feature = open(outFile, 'wb')

features = {}
channels = 1
sentNum = 0

# Open the caption file
scpFile = open('C:\\Users\\msraspeech\\PycharmProjects\\v-honwa\\test_caption.txt')
while True:
    line = scpFile.readline().strip()  # remove the '\r\n'
    if not line:
        break
    else:
        key = line.split('\t')[0]
        image_file = image_path + key + '_' + str(0) + '.bmp'

        # Replace imread with imageio.imread
        im = imageio.imread(image_file)
        mat = numpy.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')

        # Read image for each channel
        for channel in range(channels):
            image_file = image_path + key + '_' + str(channel) + '.bmp'
            im = imageio.imread(image_file)
            mat[channel, :, :] = im
        
        sentNum = sentNum + 1
        features[key] = mat
        
        # Print progress every 500 sentences
        if sentNum % 500 == 0:
            print('Process sentences ', sentNum)

print('Load images done. Sentence number ', sentNum)

# Save the features into a pickle file
pkl.dump(features, oupFp_feature)
print('Save file done')

# Close the pickle file
oupFp_feature.close()
