import numpy as np
import torch 

import matplotlib.pyplot as plt 

import pandas as pd

from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader, random_split
from einops.layers.torch import Rearrange

def one_hot_TDK(labels_far,labels_near): # order far -> near , fT = 0 , fA = 1, nT = 2, nA = 3 
    y = []
    for case in labels_far: 
        if case[0] == 'Transport':
            if case[3] in ['Bag_lige','For_lige','Ml_lige-bred','Ml_lige-midt']:
                y.append(0)
            elif case[3] in ['Mellem_midt','For_midt']:
                y.append(1)
            elif case[3] in ['Bag_ulige','For_ulige','Ml_ulige-bred','Ml_ulige-midt']:
                y.append(2)
        elif case[0] == 'Angreb':
            if case[3] in ['Bag_lige','For_lige','Ml_lige-bred','Ml_lige-midt']:
                y.append(3)
            elif case[3] in ['Mellem_midt','For_midt']:
                y.append(4)
            elif case[3] in ['Bag_ulige','For_ulige','Ml_ulige-bred','Ml_ulige-midt']:
                y.append(5)
    for case in labels_near:
        if case[0] == 'Transport':
            if case[3] in ['Bag_lige','For_lige','Ml_lige-bred','Ml_lige-midt']:
                y.append(6)
            elif case[3] in ['Mellem_midt','For_midt']:
                y.append(7)
            elif case[3] in ['Bag_ulige','For_ulige','Ml_ulige-bred','Ml_ulige-midt']:
                y.append(8)
        elif case[0] == 'Angreb':
            if case[3] in ['Bag_lige','For_lige','Ml_lige-bred','Ml_lige-midt']:
                y.append(9)
            elif case[3] in ['Mellem_midt','For_midt']:
                y.append(10)
            elif case[3] in ['Bag_ulige','For_ulige','Ml_ulige-bred','Ml_ulige-midt']:
                y.append(11)
    return np.array(y)


def one_hot_ol(labels_arr):
    y = []
    for case in labels_arr:
        if case[0] == ' n' or case[0] == 'n' or case[0] == 'n ':
            y.append(0)
        elif case[0] == 'rtpt' or case[0] == 'rtpt ':
            y.append(1)
        elif case[0] == 'rtpb' or case[0] == 'rptb' or case[0] == 'rtpb rtpb' or case[0] == 'rtpb ':
            y.append(2)
        elif case[0] == 'spt':
            y.append(3)
        elif case[0] == 'spb':
            y.append(4)
        elif case[0] == 'smpt':
            y.append(5)
        elif case[0] == 'smpb':
            y.append(6)
        elif case[0] == 'lbpt' or case[0] == 'lpbt':
            y.append(7)
        elif case[0] == 'lbpb' or case[0] == 'lbpb ':
            y.append(8)
        elif case[0] == 'fhpt':
            y.append(9)
        elif case[0] == 'fhpb':
            y.append(10)
        elif case[0] == 'bhpt' or case[0] == 'bhpt ' :
            y.append(11)
        elif case[0] == 'bhpb':
            y.append(12)
        else:
            print(case[0])
    return np.array(y)

import math
def euclid_dist(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)
    

def normalization_3(inp): # should specify for confidence level / 3D aswell
    x = inp.copy().reshape(-1,inp.shape[2],inp.shape[3])
    for batch in x:
        for t in batch:
            min_x = 2000
            min_y = 2000
            for v in t[0::3]:
                if v !=0 and v<min_x:
                    min_x = v
            for v in t[1::3]:
                if v !=0 and v<min_y:
                    min_y = v
            max_x = np.amax(t[0::3])
            max_y = np.amax(t[1::3])
            dist = euclid_dist((min_x,min_y),(max_x,max_y))
            if dist > 0:
                for i in range(0,len(t),3): #(if t_i >0.0) yes
                    if t[i]>0.:
                        t[i] = (t[i]-min_x)/dist
                    if t[i+1]>0.:
                        t[i+1] = (t[i+1]-min_y)/dist 
    return x.reshape(inp.shape[0],inp.shape[1],inp.shape[2],inp.shape[3])

def normalization(inp):
    x = inp.copy().reshape(-1,inp.shape[2],inp.shape[3]) 
    for batch in x:
        for t in batch:
            min_x = 2000
            min_y = 2000
            for v in t[0::2]:
                if v !=0 and v<min_x:
                    min_x = v
            for v in t[1::2]:
                if v !=0 and v<min_y:
                    min_y = v
            max_x = np.amax(t[0::2])
            max_y = np.amax(t[1::2])
            dist = euclid_dist((min_x,min_y),(max_x,max_y))
            
            if dist > 0:
                for i in range(0,len(t),2): #(if t_i >0.0) yes
                    if t[i]>0.:
                        t[i] = (t[i]-min_x)/dist
                    if t[i+1]>0.:
                        t[i+1] = (t[i+1]-min_y)/dist 
    return x.reshape(inp.shape[0],inp.shape[1],inp.shape[2],inp.shape[3])


def create_limbs_robust(keypoints, pairs):
    num_steps = 3
    limbs = []
    unique_points = torch.tensor(list({value for pair in pairs for value in pair})).long()
    for pose in keypoints:
        pose_limbs = []
        useful_points  = pose[unique_points,:2] 
        for start, end in pairs:
            start_point = pose[start]
            end_point = pose[end]
            if start_point[0] <= 0.01 or start_point[1] <= 0.01 or end_point[0] <= 0.01 or end_point[1] <= 0.01:
                limb = torch.zeros((num_steps-2, 2))
            else:
                x_linspace = torch.linspace(start_point[0], end_point[0], steps=num_steps)[1:-1]
                y_linspace = torch.linspace(start_point[1], end_point[1], steps=num_steps)[1:-1]
                limb = torch.stack([x_linspace, y_linspace], dim=1)
            pose_limbs.append(limb)
        limbs.append(torch.cat([useful_points,rearrange(torch.stack(pose_limbs),'n p d -> (n p) d')],dim=0))
    return torch.stack(limbs)

def create_bones_robust(keypoints, pairs):
    limbs = []
    for pose in keypoints:
        pose_limbs = []
        for start, end in pairs:
            start_point = pose[start]
            end_point = pose[end]
            if start_point[0] <= 0.01 or start_point[1] <= 0.01 or end_point[0] <= 0.01 or end_point[1] <= 0.01:
                limb = torch.zeros((2))
            else:
                x_bone = end_point[0]-start_point[0]
                y_bone = end_point[1]-start_point[1]
                limb = torch.tensor([x_bone, y_bone])
            pose_limbs.append(limb)
        limbs.append(torch.stack(pose_limbs))
    return torch.stack(limbs)

class PoseData_OL(Dataset): ### for the 
    def __init__(self,dataset,labels,position,shuttle,normalize=True,len_max = 50,concat=False,interpolation=False, transform=None): ## org 75
        super().__init__()
        self.transform = transform
        self.concat = concat
        self.pairs_b25 = [
            (1,2),(2,3),(3,4),
            (1,5),(5,6),(6,7),
            (1,8),(2,9),(5,12),
            (8,9),(9,10),(10,11),
            (8,12),(12,13),(13,14),
            ]

        ## head , arms, torso, legs
        self.pairs_coco = [
             (0,1),(0,2),(1,3),(2,4),
             (5,7),(7,9),(6,8),(8,10),
             (5,6),(5,11),(6,12),(11,12),
             (11,13),(13,15),(12,14),(14,16)
             ]
        self.clip_len = len_max
        self.n_max  = 2
        temporal = []
        persons = []
        poses = []
        pos_pad = []
        shuttle_pad = []
        for m,i in enumerate(dataset):
            if len(i[0])<=self.clip_len:
                t_s = np.zeros((self.clip_len,2))
                t_p = np.zeros((self.clip_len,4))
                temp = np.zeros((self.n_max,self.clip_len,75))
                t_s[:len(i[0])] = np.array(shuttle[m])[:,:2]
                t_p[:len(i[0])] = np.array(position[m])
                temp[:len(i),:len(i[0])] = i[:self.n_max]
                temporal.append(len(i[0]))
                persons.append(len(i))
                pos_pad.append(t_p)
                shuttle_pad.append(t_s)
                poses.append(rearrange(temp,'n t p -> n t p')) # identity now()

            elif len(i[0])>self.clip_len:
                frames_len = len(i[0])
                snip_loc = frames_len//self.clip_len#np.random.randint(0,frames_len-self.clip_len)
                snip_stop = frames_len-frames_len%self.clip_len
                temp = np.zeros((self.n_max,self.clip_len,75))
                t_s = np.zeros((self.clip_len,2))
                t_p = np.zeros((self.clip_len,4))
                temp[:len(i)] = i[:self.n_max, 0:snip_stop:snip_loc]
                t_s = np.array(shuttle[m])[0:snip_stop:snip_loc,:2]
                t_p = np.array(position[m])[0:snip_stop:snip_loc]
                temporal.append(len_max)
                persons.append(len(i))
                pos_pad.append(t_p)
                shuttle_pad.append(t_s)
                poses.append(rearrange(temp,'n t p -> n t p'))
        self.persons = torch.tensor(persons).type(torch.LongTensor)
        self.temporal = torch.tensor(temporal).type(torch.LongTensor)
        self.data = np.array(poses)
        self.position = torch.from_numpy(np.array(pos_pad)).type(torch.FloatTensor)/670 
        self.shuttle = torch.from_numpy(np.array(shuttle_pad)).type(torch.FloatTensor)/((1270+720)/2)
        self.sp = torch.cat((self.position,self.shuttle[:,:,:2]),dim=2)

        self.data =  torch.from_numpy(normalization_3(self.data))

        data_bones = create_bones_robust(rearrange(self.data.reshape(len(self.data),len(self.data[0]),len(self.data[0,0]),25,3),'b n t p d -> (b n t) p d'),self.pairs_coco)
        self.bones = rearrange(data_bones,'(b n t) p d -> b n t p d',n=len(self.data[0]),t= len(self.data[0,0]))
        if interpolation:
            data_limbs = create_limbs_robust(rearrange(self.data.reshape(len(self.data),len(self.data[0]),len(self.data[0,0]),25,3),'b n t p d -> (b n t) p d'),self.pairs_coco)
            self.data = rearrange(data_limbs,'(b n t) p d -> b n t p d',n=len(self.data[0]),t= len(self.data[0,0]))
        else:
            self.data = self.data.reshape(len(self.data),len(self.data[0]),len(self.data[0,0]),25,3)[:,:,:,:,:2]
        self.label = torch.LongTensor(labels)
        
        print(self.data.shape)
        print(self.label.shape)
        

    def __len__(self):
        ### Method to return number of data points
        return len(self.data)
    def __getitem__(self,index):
        x_key = self.data[index]
        x_bones = self.bones[index]
        x_sp = self.sp[index]
        #y = self.labels[index]
        if self.transform:
            # Apply data augmentation
            x_key = self.transform(x_key)
        if self.concat:
            return torch.cat((rearrange(x_key,'n t p d -> n t (p d)').type(torch.FloatTensor),rearrange(x_bones,'n t p d -> n t (p d)').type(torch.FloatTensor),repeat(x_sp.unsqueeze(0), '() t d -> n t d', n=2)),dim=2),self.label[index],x_sp,self.temporal[index]    
        else:
            return torch.cat((rearrange(x_key,'n t p d -> n t (p d)').type(torch.FloatTensor),rearrange(x_bones,'n t p d -> n t (p d)').type(torch.FloatTensor)),dim=2),self.label[index],x_sp,self.temporal[index]


class RandomScaling(object):
    def __init__(self, scale_range=(0.9, 1.1),prob=0.3):
        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, x):
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        if np.random.uniform(0, 1)<self.prob:
            x = x * scale
        return x

class RandomRotation(object):
    def __init__(self, angle_range=(-10, 10)):
        self.angle_range = angle_range

    def __call__(self, x):
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        c, s = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return x.dot(rotation_matrix)

class RandomTranslation(object):
    def __init__(self, translation_range=(-0.3, 0.3),prob=0.3):
        self.translation_range = translation_range
        self.prob = prob

    def __call__(self, x):
        translation = np.random.uniform(self.translation_range[0], self.translation_range[1], size=(2,))
        if np.random.uniform(0, 1)<self.prob:
            x = x + translation
        return x 
    
class RandomFlip(object):
    def __init__(self, flip_prob=1.0,prob=0.3):
        self.flip_prob = flip_prob
        self.prob = prob

    def __call__(self, x):
        if np.random.uniform(0, 1) < self.prob:
            if np.random.uniform(0, 1) < self.flip_prob:
                # Randomly flip horizontally
                x[:,:,:, 0] = 1 - x[:,:,:, 0]  # Flip x-coordinates
                #x[:,:,:, 0] = x[:,:,:, 0] - 0.5  # Adjust x-coordinates to new coordinate system

            if np.random.uniform(0, 1) < self.flip_prob:
                # Randomly flip vertically
                x[:,:,:, 1] = 1 - x[:,:,:, 1]  # Flip y-coordinates
                #x[:,:,:, 1] = x[:,:,:, 1] - 0.5  # Adjust y-coordinates to new coordinate system

        return x

def select_trainingtest(data,lab,position,shuttle,number):
    if number is None:
        test_label = []
    else:
        test_label = np.unique(lab[:,1])[number]
    if len(test_label) == 1:
        test_label = [test_label]
    Key_arr_te = []
    Labels_te = []
    pos_te = []
    shut_te = []
    Key_arr_tr = []
    Labels_tr = []
    pos_tr = []
    shut_tr = []
    for j,x in enumerate(lab[:,1]):
        if x in test_label:
            Key_arr_te.append(data[j])
            Labels_te.append(lab[j])
            pos_te.append(position[j])
            shut_te.append(shuttle[j])
        else:
            Key_arr_tr.append(data[j])
            Labels_tr.append(lab[j])
            pos_tr.append(position[j])
            shut_tr.append(shuttle[j])
    return (Key_arr_tr, Labels_tr,pos_tr,shut_tr),(Key_arr_te, Labels_te,pos_te,shut_te)

def filterOL(data,lab,position,shuttle): ### lose single frame sequences
    Key_arr = []
    Labels = []
    pos = []
    shut = []
    for j,x in enumerate(data):
        if len(x[0])>=int(2) and len(x[0])<=int(1000):
            Key_arr.append(x)
            Labels.append(lab[j])
            pos.append(position[j])
            shut.append(shuttle[j])
    return Key_arr, Labels,pos,shut
