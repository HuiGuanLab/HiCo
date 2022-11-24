import time
import torch

import numpy as np
np.set_printoptions(threshold=np.inf)
import random

try:
    from feeder import augmentations
except:
    import augmentations


class Feeder(torch.utils.data.Dataset):
    """ 
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 input_representation,
                 mmap=True):

        self.data_path = data_path
        self.num_frame_path= num_frame_path
        self.input_size=input_size
        self.input_representation=input_representation
        self.crop_resize =True
        self.l_ratio = l_ratio


        self.load_data(mmap)

        self.N, self.C, self.T, self.V, self.M = self.data.shape
        self.S = self.V
        self.B = self.V
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
        
        
        print(self.data.shape,len(self.number_of_frames))
        print("l_ratio",self.l_ratio)

    def load_data(self, mmap):
        # data: N C T V M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # load num of valid frame length
        if self.num_frame_path != None:
            self.number_of_frames= np.load(self.num_frame_path)
        else:
            self.number_of_frames= np.ones(self.data.shape[0],dtype=np.int32)*50

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):
  
        # get raw input

        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        
        number_of_frames = self.number_of_frames[index]

        # apply spatio-temporal augmentations to generate  view 1 

        # temporal crop-resize
        data_numpy_v1_crop = augmentations.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio, self.input_size)


        # randomly select  one of the spatial augmentations 
        flip_prob  = random.random()
        if flip_prob < 0.5:
                 data_numpy_v1 = augmentations.joint_courruption(data_numpy_v1_crop)
        else:
                 data_numpy_v1 = augmentations.pose_augmentation(data_numpy_v1_crop)


        # apply spatio-temporal augmentations to generate  view 2

        # temporal crop-resize
        data_numpy_v2_crop = augmentations.temporal_cropresize(data_numpy,number_of_frames, self.l_ratio, self.input_size)

        # randomly select  one of the spatial augmentations 
        flip_prob  = random.random()
        if flip_prob < 0.5:
                 data_numpy_v2 = augmentations.joint_courruption(data_numpy_v2_crop)
        else:
                 data_numpy_v2 = augmentations.pose_augmentation(data_numpy_v2_crop)

        if self.input_representation == "joint":
            # joint
            # query
            # CTVM --> TMVC
            # time-majored 
            qc_joint = data_numpy_v1.transpose(1,3,2,0)
            qc_joint = qc_joint.reshape(self.input_size, self.M*self.V*self.C).astype('float32')
            # CTVM --> MVTC
            # space-majored
            qp_joint = data_numpy_v1.transpose(3,2,1,0)
            qp_joint = qp_joint.reshape(self.M*self.V, self.input_size*self.C).astype('float32')
            
            # key
            kc_joint = data_numpy_v2.transpose(1,3,2,0)
            kc_joint = kc_joint.reshape(self.input_size, self.M*self.V*self.C).astype('float32')
            kp_joint = data_numpy_v2.transpose(3,2,1,0)
            kp_joint = kp_joint.reshape(self.M*self.V, self.input_size*self.C).astype('float32')

            return qc_joint, qp_joint, kc_joint, kp_joint
        
        elif self.input_representation == "motion":
            # motion
            motion_v1 = np.zeros_like(data_numpy_v1) 
            motion_v1[:,:-1,:,:] = data_numpy_v1[:,1:,:,:] - data_numpy_v1[:,:-1,:,:]  
            
            # query
            # CTSM --> TMSC
            qc_motion = motion_v1.transpose(1,3,2,0)
            qc_motion = qc_motion.reshape(self.input_size, self.M*self.S*self.C).astype('float32')
            # CTSM --> MSTC
            qp_motion = motion_v1.transpose(3,2,1,0)
            qp_motion = qp_motion.reshape(self.M*self.S, self.input_size*self.C).astype('float32')
            
            motion_v2 = np.zeros_like(data_numpy_v2)
            motion_v2[:,:-1,:,:] = data_numpy_v2[:,1:,:,:] - data_numpy_v2[:,:-1,:,:]
            # key
            kc_motion = motion_v2.transpose(1,3,2,0)
            kc_motion = kc_motion.reshape(self.input_size, self.M*self.S*self.C).astype('float32')
            kp_motion = motion_v2.transpose(3,2,1,0)
            kp_motion = kp_motion.reshape(self.M*self.S, self.input_size*self.C).astype('float32')

            return qc_motion, qp_motion, kc_motion, kp_motion
        
        elif self.input_representation == "bone":
            # bone
            bone_v1 = np.zeros_like(data_numpy_v1)
            for v1,v2 in self.Bone:
                bone_v1[:,:,v1-1,:] = data_numpy_v1[:,:,v1-1,:] - data_numpy_v1[:,:,v2-1,:]
                
            # CTBM --> TMBC
            qc_bone = bone_v1.transpose(1,3,2,0)
            qc_bone = qc_bone.reshape(self.input_size, self.M*self.B*self.C).astype('float32')
            # CTBM --> MBTC
            qp_bone = bone_v1.transpose(3,2,1,0)
            qp_bone = qp_bone.reshape(self.M*self.B, self.input_size*self.C).astype('float32')
            
            bone_v2 = np.zeros_like(data_numpy_v2)
            for v1,v2 in self.Bone:
                bone_v2[:,:,v1-1,:] = data_numpy_v2[:,:,v1-1,:] - data_numpy_v2[:,:,v2-1,:]
            # CTBM --> TMBC
            kc_bone = bone_v2.transpose(1,3,2,0)
            kc_bone = kc_bone.reshape(self.input_size, self.M*self.B*self.C).astype('float32')
            kp_bone = bone_v2.transpose(3,2,1,0)
            kp_bone = kp_bone.reshape(self.M*self.B, self.input_size*self.C).astype('float32')

            return qc_bone, qp_bone, kc_bone, kp_bone

        
