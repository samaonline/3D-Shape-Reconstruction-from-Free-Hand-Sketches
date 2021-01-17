import sys
import numpy as np
import cv2
import random
import math
import os
import time
import zlib
import socket
import threading
import queue
import sys
import pickle
import show3d
from plyfile import PlyData, PlyElement
import glob
import skimage as sk
import skimage.io as skio
import cv2
import skimage
import imageio

from img_utils import (mls_affine_deformation, mls_affine_deformation_inv,
                       mls_similarity_deformation, mls_similarity_deformation_inv,
                       mls_rigid_deformation, mls_rigid_deformation_inv)
import os

from pdb import set_trace as st

FETCH_BATCH_SIZE=15 #39
NUM_PCLASS = 3
BATCH_SIZE=FETCH_BATCH_SIZE#32
HEIGHT=192
WIDTH=256
POINTCLOUDSIZE=2048 #16384
OUTPUTPOINTS=1024
REEBSIZE=1024

def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def binary(im):
    return (im>180)*255

def affine_im(face, transpix=[10, 5], angle=10):
    im1 = skimage.transform.rotate(255-binary(face), angle)
    translation_matrix = np.float32([ [1, 0, transpix[0]], [0, 1, transpix[1]] ])
    im1 = cv2.warpAffine(im1, translation_matrix, (face.shape[0], face.shape[1]))
    
    im1 = im1/np.max(im1)
    return (1-im1)*255

def transform_im(image, NUM_MVP = 20):
    rowid, colid = np.where( image <100)
    id2cs = np.random.choice(range(len(rowid)), NUM_MVP, replace=False)
    
    p = np.stack((rowid, colid)).transpose()
    p = p[id2cs]
    q = p + (20*(np.random.rand(p.shape[0], p.shape[1]) -0.5 ) ).astype(int)
    
    transformed_image = 255*mls_rigid_deformation_inv(image, p, q, alpha=1, density=1)

    if np.random.uniform()>0.5:
        transformed_image = affine_im(transformed_image, transpix=0.1*image.shape[0]*(np.random.uniform(size=[2])-0.5), angle=10*(np.random.uniform()-0.5))
    return transformed_image

class BatchFetcher(threading.Thread):
    def __init__(self, dataname, flag):
        super(BatchFetcher,self).__init__()
        self.queue=queue.Queue(64)
        self.stopped=False
        self.datadir = dataname
        self.bno=0
        self.flag=flag
        self.ids = ["02691156", "02933112", "03001627", "03636649", "04090263", "04379243", "04530566", "02828884",
               "02958343", "03211117", "03691459", "04256520", "04401088"]
        
        ###
        train_files = []
        test_files = []
        
        for id_ in self.ids:
            train_files.append(glob.glob("/ssd/peterwg/shapenet_pc/train/"+id_+"/*"))
            test_files.append(glob.glob("/ssd/peterwg/shapenet_pc/test/"+id_+"/*"))
        
        #train_files = glob.glob("/home/peterwg/dataset/shapenet_pc/train/*/*")
        #test_files = glob.glob("/home/peterwg/dataset/shapenet_pc/test/*/*")
        
        for i, train_file in enumerate(train_files):
            temp = []
            for str_ in train_file:
                temp.append((str_.split('/')[-2],str_.split('/')[-1][:-4]))
            train_files[i] = temp
        
        for i, test_file in enumerate(test_files):
            temp = []
            for str_ in test_file:
                temp.append((str_.split('/')[-2],str_.split('/')[-1][:-4]))
            test_files[i] = temp
        
        """for str in test_files:
            self.test_keys.append((str.split('/')[-3],str.split('/')[-1][:-4]))"""
        self.val_cnt=0
        
        self.train_keys = train_files
        self.test_keys = test_files
        
        ####
        sub_dir = "/ssd/peterwg/ShapeNetSK/04379243/34bdbfbe94a760aba396ce8e67d44089/rendering/"#"/ssd/peterwg/ShapeNetSK/03001627/406561a447d3f7787f4096327f1fb3a7/rendering/"#"/home/peterwg/dataset/collection/03001627_5"
        files_sk = [os.path.join(sub_dir, file) for file in os.listdir(sub_dir) if file.endswith('.png')]
        self.test_keys1 = []
        for str_ in files_sk:
            self.test_keys1.append(str_)#str.split('/')[-1].split('.')[0])
        ####
        
        files = glob.glob("/home/peterwg/dataset/shapenet_pc/train/03001627/*")
        #files = glob.glob("/home/peterwg/temp/comp_vox/*")
        #files_sk = glob.glob("/home/peterwg/dataset/sketchy/256x256/sketch/tx_000100000000/car_(sedan)/*")
        files_sk = [file for file in os.listdir("/home/peterwg/dataset/collection") if file.endswith('.png')]
        #files_sk = glob.glob("/home/peterwg/temp/chair_sk/*")
        self.test_keys = []
        for str_ in files:
            self.test_keys.append(str_.split('/')[-1].split('.')[0])
            
        ###
        for str_ in files_sk:
            self.test_keys1.append(str_.split('/')[-1].split('.')[0])
        
        ####
        
    def work(self,bno):
        data = []
        ptclouds = []
        labels = []
        if(self.flag):
            for label, train_key in enumerate(self.train_keys):
                keys_indices = np.random.choice(len(train_key), NUM_PCLASS, replace=False)
                for i in range(NUM_PCLASS):
                    ptcloud=[]
                    idx = np.random.randint(0,24)
                    if(idx<10):
                        idx = '0'+str(idx)
                    else:
                        idx = str(idx)
                    try:
                        sketch_name = glob.glob("/ssd/peterwg/ShapeNetSK/" + train_key[keys_indices[i]][0] + "/" + train_key[keys_indices[i]][1] + "/rendering/" + idx + ".png")
                        im = skio.imread(sketch_name[0])
                    except:
                        sketch_name = glob.glob("/ssd/peterwg/ShapeNetSK/" + train_key[keys_indices[i]][0] + "/" + train_key[keys_indices[i]][1] + "/rendering/" + idx + ".png")
                        im = skio.imread(sketch_name[0])
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    if np.random.uniform() > 0.8:
                        try:
                            im = transform_im(im)
                        except:
                            pass
                    im = cv2.resize(im, (256,192), interpolation = cv2.INTER_AREA)
                    im = sk.img_as_float(im)
                    print(np.unique(im))
                    data.append(im)                

                    plydata = PlyData.read("/ssd/peterwg/shapenet_pc/train/" + train_key[keys_indices[i]][0] + "/" + train_key[keys_indices[i]][1] + ".ply")
                    for j in range(len(plydata.elements[0].data)):
                        temp = []
                        temp.append(plydata.elements[0].data[j][0])
                        temp.append(plydata.elements[0].data[j][1])
                        temp.append(plydata.elements[0].data[j][2])
                        ptcloud.append(temp)
                    ptclouds.append(ptcloud)
                    label_ = np.zeros(13)
                    label_[label] = 1
                    labels.append(label_)
            
            """keys_indices = np.random.choice(len(self.train_keys), FETCH_BATCH_SIZE, replace=False)
            i = 0
            while(i < FETCH_BATCH_SIZE):
                ptcloud=[]
                idx = np.random.randint(0,24)
                if(idx<10):
                    idx = '0'+str(idx)
                else:
                    idx = str(idx)

                sketch_name = glob.glob("/home/peterwg/dataset/shapenet_pix2vox/ShapeNetSK/" + self.train_keys[keys_indices[i]][0] + "/" + self.train_keys[keys_indices[i]][1] + "/rendering/" + idx + ".png")
                im = skio.imread(sketch_name[0])
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im = cv2.resize(im, (256,192), interpolation = cv2.INTER_AREA)
                im = sk.img_as_float(im) / 255.
                data.append(im)

                plydata = PlyData.read("/home/jlin/data/output/train/" + self.train_keys[keys_indices[i]][0] + "/" + self.train_keys[keys_indices[i]][1] + ".ply")
                for j in range(len(plydata.elements[0].data)):
                    temp = []
                    temp.append(plydata.elements[0].data[j][0])
                    temp.append(plydata.elements[0].data[j][1])
                    temp.append(plydata.elements[0].data[j][2])
                    ptcloud.append(temp)
                ptclouds.append(ptcloud)
                i+=1"""
        else:
            print(self.val_cnt)
            keys_indices = self.val_cnt * FETCH_BATCH_SIZE + np.arange(FETCH_BATCH_SIZE)

            i = 0
            while (i < FETCH_BATCH_SIZE):
                ptcloud = []
                # fix a view
                idx = 0
                if (idx < 10):
                    idx = '0' + str(idx)
                else:
                    idx = str(idx)
                sketch_name = glob.glob(self.test_keys1[keys_indices[i]])
                #sketch_name = glob.glob("/home/jlin/data/ShapeNetSK/" + self.test_keys[keys_indices[i]][0] + "/" + self.test_keys[keys_indices[i]][1] + "/rendering/" + idx + ".png")
                #im = skio.imread(sketch_name[0])
                #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im = imageio.imread(sketch_name[0])
                if im.shape[-1] ==4:
                    im = rgb2gray( rgba2rgb(im) )
                else:
                    im = rgb2gray( im)
                im = cv2.resize(im, (256, 192), interpolation=cv2.INTER_AREA)
                im = sk.img_as_float(im) / 255.
                data.append(im)
                
                plydata = PlyData.read("/ssd/peterwg/shapenet_pc/test/" + self.test_keys[keys_indices[0]][i][0] + "/" + self.test_keys[keys_indices[0]][i][1] + ".ply") #PlyData.read("/home/jlin/data/output/" + self.test_keys[keys_indices[i]][0] + "/test/" + self.test_keys[keys_indices[i]][1] + ".ply")
                for j in range(len(plydata.elements[0].data)):
                    temp = []
                    temp.append(plydata.elements[0].data[j][0])
                    temp.append(plydata.elements[0].data[j][1])
                    temp.append(plydata.elements[0].data[j][2])
                    ptcloud.append(temp)
                ptclouds.append(ptcloud)
                i += 1

        validating=np.zeros(FETCH_BATCH_SIZE,dtype='float32')
        data = np.array(data)
        ptclouds = np.array(ptclouds)
        return (np.expand_dims(data,3),ptclouds,validating, np.array(label))

    def run(self):
        # change 300000 to 64
        while self.bno<300000 and not self.stopped:
            data,ptclouds,validating=self.work(self.bno % 64)
            self.queue.put((data,ptclouds,validating))
            self.bno+=1
            self.val_cnt+=1

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()
    def shutdown(self):
        self.stopped=True
        while not self.queue.empty():
            self.queue.get()

if __name__=='__main__':
    dataname = "YTTRBtraindump_220k"
    fetchworker = BatchFetcher(dataname)
    fetchworker.bno=0
    fetchworker.start()
    for cnt in range(100):
        data,ptcloud,validating = fetchworker.fetch()
        validating = validating[0]!=0

        assert len(data)==FETCH_BATCH_SIZE
        for i in range(len(data)):
            cv2.imshow('data',data[i])
            while True:
                cmd=show3d.showpoints(ptcloud[i])
                if cmd==ord(' '):
                    break
                elif cmd==ord('q'):
                    break
            if cmd==ord('q'):
                break


