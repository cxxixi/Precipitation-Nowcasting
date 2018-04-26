# -*- coding: utf-8 -*-
#! /usr/bin/python3
"""
@author: cx,pqxu
"""

from PIL import Image
import torch.utils.data as data
import arrow
import sys
import os
import os.path
import time
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as func
from torch.utils.data import DataLoader
import argparse



args = argparse.ArgumentParser()

args.add_argument('--data_dir',
                  type = str,
                  default = './data/',
                  help = "dir where training is conducted")
args.add_argument('--logs_train_dir',
                  type = str,
                  default = './logs/loss_record/',
                  help = "dir where summary is saved")                 
args.add_argument('--img_dir',
                  type = str,
                  default = './logs/test/',
                  help = "dir where output images are saved")
args.add_argument('--model_dir',
                  type = str,
                  default = './logs/models/',
                  help = "dir where models are saved")
args.add_argument('--trainset_name',
                  type = str,
                  default = 'trainset',
                  help = "the training set where the training is conducted")
args.add_argument('--testset_name',
                  type = str,
                  default = 'testset',
                  help = "the test set where the test is conducted")
args.add_argument('--seq_length',
                  type = int,
                  default = 15,
                  help = "length of the sequence")
args.add_argument('--img_size',
                  type = int,
                  default = 256,
                  help = "length of the sequence")
args.add_argument('--seq_start',
                  type = int,
                  default = 5,
                  help = """start of the sequence generation""")
args.add_argument('--epoches',
                  type = int,
                  default = 50,
                  help = """number of epoches""")
args.add_argument('--lr',
                  type = float,
                  default = .001,
                  help = """learning rate""")
args.add_argument('--wd',
                  type = float,
                  default = 1e-4,
                  help = """learning rate""")
args.add_argument('--factor',
                  type = float,
                  default = .0005,
                  help = """factor of regularization""")
args.add_argument('--batch_size',
                   type = int,
                   default = 8,
                   help = """batch size for training""")
args.add_argument('--weight_init',
                  type = float,
                  default = .1,
                  help = """weight init for FC layers""")
args.add_argument('--threshold',
                  type = float,
                  default = 48.0,
                  help = """the threshold pass which is identified as hit""")

args = args.parse_args()


def initcircle(shape,r,batch,num):          
    shape = shape[1]
    center = (shape - 1)/2.0
    x = np.array(shape*[ np.arange(shape) ])        ## ncol * [ mrow ]      = 2d array    
    y = np.transpose(x,(1,0))
    x = x - center
    y = y - center
    
    output = np.sqrt(x**2 + y**2)   
    output = np.float32(output<r)                   ## >=r 数值变成0，<r 数值变成1
    
    output1 = []
    for i in range(num):        ##输出通道数量
        output1.append(output)
        
    output2 = []
    for i in range(batch):      ##每次迭代， batch==16
        output2.append(output1)
    output2 = np.asarray(output2)
    
    return output2



class Dataloader0():
    def __init__(self,dir,give,predict,rot= True):
        
        self.rot = rot
        self.data_list = []
        for root, dir, files in os.walk(dir):
            for file in files:
                if file.find('.png')!= -1:
                    self.data_list.append(os.path.join(root, file))
        self.data_list.sort()

        self.seq_length = give + predict
        self.give = give
        self.predict = predict

        self.all_list = []
        for i in range(len(self.data_list) - (self.seq_length - 1)):
            pre = self.data_list[i]
            nex = self.data_list[i+(self.seq_length - 1)]

            a = pre.split('/')
            b = nex.split('/')

            a_UTC = a[-1][:-4]
            b_UTC = b[-1][:-4]

            a_UTC = arrow.get(a_UTC,'MMDDHHmm')
            a_unix = int(a_UTC.timestamp)
            b_UTC = arrow.get(b_UTC,'MMDDHHmm')
            b_unix = int(b_UTC.timestamp)

            if b_unix - a_unix > (self.seq_length-2)*6*60 and b_unix - a_unix < self.seq_length*6*60:
                self.all_list.append(self.data_list[i:i+self.seq_length])

        print(len(self.all_list))

    def __getitem__(self, index):
        
        path = self.all_list[index]
        data = []
        label = []
        for i in range(len(path)):
            datai = Image.open(path[i]).resize((args.img_size,args.img_size),Image.BILINEAR)
            datai = np.asarray(datai)
            if i < self.give:
                data.append(datai)
            else:
                label.append(datai)

        data = np.asarray(data)
        label = np.asarray(label)
        if self.rot:
            random = int(3.999999*np.random.random())
            data = np.float32(np.rot90(data, random,axes = (1,2)))
            label = np.float32(np.rot90(label, random,axes = (1,2)))
        else:
            data = np.float32(data)
            label = np.float32(label)

        return data, label

    def __len__(self):
        return len(self.all_list)

if __name__ == '__main__':

    dataloader0 = Dataloader0('trainset',5,15)
    data,label = dataloader0.__getitem__(1)


