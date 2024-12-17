import numpy as np
import torch
import torch.utils.data
import random


# class SlidingWindowDataset(torch.utils.data.Dataset):
#     # def __init__(self, data, window=1, horizon=1, transform=None, dtype=torch.float):
#     def __init__(self, data, window, horizon, transform=None, dtype=torch.float):
#         super().__init__()
#         self._data = data
#         self._window = window
#         self._horizon = horizon
#         self._dtype = dtype
#         self._transform = transform

#     # def __getitem__(self, index):
#     #     x = self._data[index : index + self._window]
#     #     y = self._data[index + self._window : index + self._window + self._horizon]
#     #     # y=[]
#     #     # chance=0.5
#     #     # if chance <=0.5:
#     #     #     y.append(True)
#     #     # else:
#     #     #     y.append(False)

#     #     # switching to PyTorch format C,D,H,W
#     #     x = np.swapaxes(x, 0, 1)
#     #     y = np.swapaxes(y, 0, 1)

#     #     if self._transform:
#     #         x = self._transform(x)
#     #         y = self._transform(y)

#     #     return (
#     #         torch.from_numpy(x).type(self._dtype),
#     #         torch.from_numpy(y).type(self._dtype),
#     #     )
#     def __getitem__(self, index):
#         x = self._data[index : index + self._window]
#         # y = self._data[index + self._window : index + self._window + self._horizon]
#         y=[]

#         chance=random.uniform(0, 1)
#         if chance <=0.5:
#             y.append(True)
#         else:
#             y.append(False)

#         # switching to PyTorch format C,D,H,W
#         x = np.swapaxes(x, 0, 1)
#         # y = np.swapaxes(y, 0, 1)

#         if self._transform:
#             x = self._transform(x)
#             # y = self._transform(y)

#         return (
#             torch.from_numpy(x).type(self._dtype),
#             np.array(y),
#         )

#     def __len__(self):
#         return self._data.shape[0] - self._window - self._horizon + 1
import torchvision.transforms as transforms
from glob import glob
import random
from PIL import Image
import os
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor()
])
patientFolderList = glob("C:/Users/Cyanb/OneDrive/Desktop/E3D-LSTM-master/data/images/NegativeIMG/*/", recursive = True)
# print(patientFolderList)
# l=[]
nameList=[name for name in os.listdir("C:/Users/Cyanb/OneDrive/Desktop/E3D-LSTM-master/data/images/NegativeIMG/")]
nameCount=0
All_bag=[]
for i in patientFolderList:
    dateList=glob(i+"*/", recursive = True)
    TimeBag=torch.empty((0,1,1, 64, 64, 64))
    for j in dateList:
        dcmPaths=[]
        imgFolderPath=""+j+"/V/*"
        # print(imgFolderPath)
        liOfCTLayer=glob(imgFolderPath)
        # print(liOfCTLayer)
        image0 = Image.open(liOfCTLayer[0])
        image0=image0.resize((64,64),Image.ANTIALIAS)
        A_bag = transform(image0)
        # print(A_bag.shape)
        A_bag=A_bag.unsqueeze(0)
        # print(A_bag.shape)
        for i in liOfCTLayer[2:128:2]:
            image = Image.open(i)
            image=image.resize((64,64),Image.ANTIALIAS)
            #print(np.array(image).shape )
            img_tensor = transform(image)
            #print (tf.shape(img_tensor))
            #print (torch.sum(img_tensor))
            img_tensor=img_tensor.unsqueeze(0)
            A_bag = torch.cat((A_bag, img_tensor), 0 )
            # print (A_bag.shape)
        A_bag=A_bag.unsqueeze(0)
        A_bag=A_bag.unsqueeze(0)
        A_bag=np.swapaxes(A_bag,2,3)
        # print(TimeBag.shape)
        # print(A_bag.shape)
        TimeBag = torch.cat((TimeBag, A_bag), 0 )
        # print(TimeBag.shape)
    All_bag.append((torch.tensor([False]),TimeBag))
    
    nameCount+=1

patientFolderList = glob("C:/Users/Cyanb/OneDrive/Desktop/E3D-LSTM-master/data/images/PositiveIMG/*/", recursive = True)
# print(patientFolderList)
# l=[]
nameList=[name for name in os.listdir("C:/Users/Cyanb/OneDrive/Desktop/E3D-LSTM-master/data/images/PositiveIMG/")]
nameCount=0
for i in patientFolderList:
    dateList=glob(i+"*/", recursive = True)
    TimeBag=torch.empty((0,1,1, 64, 64, 64))
    for j in dateList:
        dcmPaths=[]
        imgFolderPath=""+j+"/V/*"
        # print(imgFolderPath)
        liOfCTLayer=glob(imgFolderPath)
        # print(liOfCTLayer)
        image0 = Image.open(liOfCTLayer[0])
        image0=image0.resize((64,64),Image.ANTIALIAS)
        A_bag = transform(image0)
        # print(A_bag.shape)
        A_bag=A_bag.unsqueeze(0)
        # print(A_bag.shape)
        for i in liOfCTLayer[2:128:2]:
            image = Image.open(i)
            image=image.resize((64,64),Image.ANTIALIAS)
            #print(np.array(image).shape )
            img_tensor = transform(image)
            #print (tf.shape(img_tensor))
            #print (torch.sum(img_tensor))
            img_tensor=img_tensor.unsqueeze(0)
            A_bag = torch.cat((A_bag, img_tensor), 0 )
            # print (A_bag.shape)
        A_bag=A_bag.unsqueeze(0)
        A_bag=A_bag.unsqueeze(0)
        A_bag=np.swapaxes(A_bag,2,3)
        TimeBag = torch.cat((TimeBag, A_bag), 0 )
        # print(TimeBag.shape)
    All_bag.append((torch.tensor([True]),TimeBag))
    
    nameCount+=1



random.shuffle(All_bag)
# print(All_bag[0][1].shape)

#print("ONLY 1 DS!!!!")
TestSet=All_bag[:int(len(TestSet)*0.15)]
TrainSet=All_bag[int(len(TestSet)*0.15):]