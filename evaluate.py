# -*- coding: utf-8 -*-
#! /usr/bin/python3
"""
@author: pqxu
 edited by: cx
"""


import numpy as np
import os
from PIL import Image
from util import *
import re

P_num = args.seq_length-args.seq_start
Size = (args.img_size,args.img_size)
Detect_r = 180*256/461
Threshold = args.threshold

Hit = np.zeros((P_num),dtype = np.int64)
Miss = np.zeros((P_num),dtype = np.int64)
Falsealarm = np.zeros((P_num),dtype = np.int64)

def initcircle(shape,r):
    shape = shape[1]
    center = (shape - 1)/2.0
    x = np.array(shape*[np.arange(shape)])
    y = np.transpose(x,(1,0))
    x = x - center
    y = y - center
    output = np.sqrt(x**2 + y**2)
    output = np.uint8(output<r)
    return output,x,y

Circle,x,y = initcircle(Size,Detect_r)


def main(dir):
    for root, dir, files in os.walk(dir):
        # print root
        for file in files:
            if file.find('_') == -1:
                filename = os.path.join(root, file)
                truth = np.asarray(Image.open(filename))*Circle
                truth_is = (truth>=Threshold)
                truth_not = (truth<Threshold)
                for i in range(P_num):
                    p_namei = filename[:-4] + '_' + str(i) + '.png'
                    if os.path.isfile(p_namei):
                        p_i = np.asarray(Image.open(p_namei))*Circle
                        p_i_is = (p_i>=Threshold*0.93 - i)
                        p_i_not = (p_i<Threshold*0.93 - i)

                        hit_i = np.sum(np.logical_and(truth_is,p_i_is))
                        miss_i = np.sum(np.logical_and(truth_is,p_i_not))
                        falsealarm_i = np.sum(np.logical_and(truth_not,p_i_is))
                        Hit[i] += hit_i
                        Miss[i] += miss_i
                        Falsealarm[i] += falsealarm_i

if __name__ == '__main__':
    main(args.img_dir)
    print('csi:',Hit/np.float64(Hit+Falsealarm+Miss))
    print('pod:',Hit/np.float64(Hit+Miss))
    print('far:',Falsealarm/np.float64(Hit + Falsealarm))

