import sys
import tensorflow as tf
import numpy as np
import cv2
import tflearn
import random
import math
import os
#os.system("chmod +w ../data")
import time
import zlib
import socket
import threading
import queue
import sys
import tf_nndistance
import pickle
import glob
from BatchFetcher import *
import skimage as sk
import skimage.io as skio
import cv2
import show3d
from plyfile import PlyData, PlyElement
from transform_nets import input_transform_net

RESTORE_PATH=False#"../dump2/train_nn_trans.ckpt" #False
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import utils.tf_util as tf_util


lastbatch=None
lastconsumed=FETCH_BATCH_SIZE


def fetch_batch():
    global lastbatch,lastconsumed
    if lastbatch is None or lastconsumed+BATCH_SIZE>FETCH_BATCH_SIZE:
        lastbatch=fetchworker.fetch()
        lastconsumed=0
    ret=[i[lastconsumed:lastconsumed+BATCH_SIZE] for i in lastbatch]
    lastconsumed+=BATCH_SIZE
    return ret

def stop_fetcher():
    fetchworker.shutdown()

def build_graph(resourceid, reg_weight=0.001):
    with tf.device('/gpu:%d'%resourceid):
        tflearn.init_graph(seed=1029,num_cores=2,gpu_memory_fraction=0.9,soft_placement=True)
        img_inp=tf.placeholder(tf.float32,shape=(BATCH_SIZE,HEIGHT,WIDTH,1),name='img_inp')
        pt_gt=tf.placeholder(tf.float32,shape=(BATCH_SIZE,POINTCLOUDSIZE,3),name='pt_gt')
        label=tf.placeholder(tf.float32,shape=(BATCH_SIZE,13),name='label')

        x=img_inp
#192 256
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x0=x
        with tf.variable_scope('transform_net1') as sc:
            x_d = tflearn.layers.conv.max_pool_2d(x, 8, strides=8)
            transform = input_transform_net(tf.reshape(x_d, (BATCH_SIZE,4096,3)), is_training=tf.constant(True, dtype=tf.bool), bn_decay=0.5, K=3)
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#96 128
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x1=x
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#48 64
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x2=x
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#24 32
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x3=x
        x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#12 16
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x4=x
        x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#6 8
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x5=x
        x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#3 4
        x_additional=tflearn.layers.core.fully_connected(x,2048,activation='relu',weight_decay=1e-3,regularizer='L2')
        x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
        x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x5))
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x5=x
        x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
        x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x4))
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x4=x
        x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
        x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x3))
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x3=x
        x=tflearn.layers.conv.conv_2d_transpose(x,32,[5,5],[48,64],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#48 64
        x2=tflearn.layers.conv.conv_2d(x2,32,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x2))
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x2=x
        x=tflearn.layers.conv.conv_2d_transpose(x,16,[5,5],[96,128],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#96 128
        x1=tflearn.layers.conv.conv_2d(x1,16,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x1))
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#48 64
        x2=tflearn.layers.conv.conv_2d(x2,32,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x2))
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x2=x
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
        x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x3))
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x3=x
        x=tflearn.layers.conv.conv_2d(x,128,(5,5),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
        x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x4))
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x4=x
        x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
        x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x5))
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x5=x
        x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#3 4
        x_additional=tflearn.layers.core.fully_connected(x_additional,2048,activation='linear',weight_decay=1e-4,regularizer='L2')
        x_additional=tf.nn.relu(tf.add(x_additional,tflearn.layers.core.fully_connected(x,2048,activation='linear',weight_decay=1e-3,regularizer='L2')))
        x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
        x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x5))
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x5=x
        x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
        x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x4))
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x4=x
        x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
        x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x3))
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')

        x_additional=tflearn.layers.core.fully_connected(x_additional,1024,activation='relu',weight_decay=1e-3,regularizer='L2')
        
        x_additional=tflearn.layers.core.fully_connected(x_additional,256*3,activation='linear',weight_decay=1e-3,regularizer='L2')
            
        x_additional=tf.reshape(x_additional,(BATCH_SIZE,256,3))
        x=tflearn.layers.conv.conv_2d(x,3,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.reshape(x,(BATCH_SIZE,32*24,3))
        x=tf.concat([x_additional,x],1)
        x=tf.reshape(x,(BATCH_SIZE,OUTPUTPOINTS,3))
        x = tf.matmul(x, transform)
        K = transform.get_shape()[1].value
        mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
        mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
        mat_diff_loss = tf.nn.l2_loss(mat_diff) 
        
        dists_forward,_,dists_backward,_=tf_nndistance.nn_distance(pt_gt,x)
        mindist=dists_forward
        dist0=mindist[0,:]
        dists_forward=tf.reduce_mean(dists_forward)
        dists_backward=tf.reduce_mean(dists_backward)
        loss_nodecay=(dists_forward+dists_backward/2.0)*10000
        loss=loss_nodecay+tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.1 + mat_diff_loss*reg_weight
        batchno = tf.Variable(0, dtype=tf.int32)
        optimizer = tf.train.AdamOptimizer(3e-5*BATCH_SIZE/FETCH_BATCH_SIZE).minimize(loss,global_step=batchno)
        batchnoinc=batchno.assign(batchno+1)
    return img_inp,x,pt_gt,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0


def main(resourceid,keyname):
    if not os.path.exists(dumpdir):
        os.system("mkdir -p %s"%dumpdir)
    #is_training_pl = tf.placeholder(tf.bool, shape=())
    #print(is_training_pl)

    img_inp,x,pt_gt,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0=build_graph(resourceid)
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    saver=tf.train.Saver()
    with tf.Session(config=config) as sess,\
                open('%s/%s.log'%(dumpdir,keyname),'a') as fout:
        sess.run(tf.global_variables_initializer())
        if RESTORE_PATH:
            saver.restore(sess, RESTORE_PATH) #'%s/' % dumpdir + keyname + ".ckpt")
        trainloss_accs=[0,0,0]
        trainloss_acc0=1e-9
        validloss_accs=[0,0,0]
        validloss_acc0=1e-9
        lastsave=time.time()
        bno=sess.run(batchno)

        fetchworker.bno=bno//(FETCH_BATCH_SIZE/BATCH_SIZE)
        fetchworker.start()
        # change from 300000 to 64

        while bno<300000:
            t0=time.time()
            data,ptcloud,validating,class_=fetch_batch()
            st()
            t1=time.time()
            validating=validating[0]!=0
            if not validating:
                _,pred,total_loss,trainloss,trainloss1,trainloss2,distmap_0=sess.run([optimizer,x,loss,loss_nodecay,dists_forward,dists_backward,dist0],
                    feed_dict={img_inp:data,pt_gt:ptcloud,label:class_})
                trainloss_accs[0]=trainloss_accs[0]*0.99+trainloss
                trainloss_accs[1]=trainloss_accs[1]*0.99+trainloss1
                trainloss_accs[2]=trainloss_accs[2]*0.99+trainloss2
                trainloss_acc0=trainloss_acc0*0.99+1
            else:
                _,pred,total_loss,validloss,validloss1,validloss2,distmap_0=sess.run([batchnoinc,x,loss,loss_nodecay,dists_forward,dists_backward,dist0],
                    feed_dict={img_inp:data,pt_gt:ptcloud})
                validloss_accs[0]=validloss_accs[0]*0.997+validloss
                validloss_accs[1]=validloss_accs[1]*0.997+validloss1
                validloss_accs[2]=validloss_accs[2]*0.997+validloss2
                validloss_acc0=validloss_acc0*0.997+1
            t2=time.time()
            down=2

            bno=sess.run(batchno)
            if not validating:
                showloss=trainloss
                showloss1=trainloss1
                showloss2=trainloss2
            else:
                showloss=validloss
                showloss1=validloss1
                showloss2=validloss2
            print(fout,bno,trainloss_accs[0]/trainloss_acc0,trainloss_accs[1]/trainloss_acc0,trainloss_accs[2]/trainloss_acc0,showloss,showloss1,showloss2,validloss_accs[0]/validloss_acc0,validloss_accs[1]/validloss_acc0,validloss_accs[2]/validloss_acc0,total_loss-showloss)
            if bno%128==0:
                fout.flush()
            if time.time()-lastsave>900:
                saver.save(sess,'%s/'%dumpdir+keyname+".ckpt")
                lastsave=time.time()

            print (bno,'t',trainloss_accs[0]/trainloss_acc0,trainloss_accs[1]/trainloss_acc0,trainloss_accs[2]/trainloss_acc0,'v',validloss_accs[0]/validloss_acc0,validloss_accs[1]/validloss_acc0,validloss_accs[2]/validloss_acc0,total_loss-showloss,t1-t0,t2-t1,time.time()-t0,fetchworker.queue.qsize())
        saver.save(sess,'%s/'%dumpdir+keyname+".ckpt") 

def dumppredictions(resourceid,keyname,valnum):
    is_training_pl = tf.placeholder(tf.bool, shape=())
    img_inp,x,pt_gt,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0=build_graph(resourceid)
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    saver=tf.train.Saver()
    fout = open("%s/%s.v.pkl"%(dumpdir,keyname),'wb')

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "%s/%s.ckpt" % (dumpdir, keyname))
        fetchworker.bno = 0
        fetchworker.start()
        cnt = 0

        validloss_accs = [0, 0, 0]
        validloss_acc0 = 1e-9

        for i in range(1):#range(0, 7934 // 32):
            t0 = time.time()
            data, ptcloud, validating = fetch_batch()
            validating = validating[0] != 0
            cnt += 1
            _, total_loss, validloss, validloss1, validloss2, distmap_0, pred, distmap = sess.run(
                [batchnoinc, loss, loss_nodecay, dists_forward, dists_backward, dist0, x, mindist],
                feed_dict={img_inp: data, pt_gt: ptcloud})

            validloss_accs[0] = validloss_accs[0] * 0.997 + validloss
            validloss_accs[1] = validloss_accs[1] * 0.997 + validloss1
            validloss_accs[2] = validloss_accs[2] * 0.997 + validloss2
            validloss_acc0 = validloss_acc0 * 0.997 + 1
            
            pickle.dump((i, data, ptcloud, pred, distmap), fout, protocol=-1)

            print('v', validloss_accs[0] / validloss_acc0, validloss_accs[1] / validloss_acc0,
                  validloss_accs[2] / validloss_acc0)

    fout.close()

if __name__=='__main__':
    resourceid = 0
    datadir,dumpdir,cmd,valnum="data","../dump2","predict",3
    for pt in sys.argv[1:]:
        if pt[:5]=="data=":
            datadir = pt[5:]
        elif pt[:5]=="dump=":
            dumpdir = "../dump2"
        elif pt[:4]=="num=":
            valnum = int(pt[4:])
        else:
            cmd = pt
    if datadir[-1]=='/':
        datadir = datadir[:-1]
    if dumpdir[-1]=='/':
        dumpdir = dumpdir[:-1]
    assert os.path.exists(datadir),"data dir not exists"
    os.system("mkdir -p %s"%dumpdir)
    # set flag to 1 for train and 0 for test
    flag=1
    fetchworker=BatchFetcher(datadir,flag)
    print("datadir=%s dumpdir=%s num=%d cmd=%s started"%(datadir,dumpdir,valnum,cmd))


    keyname=os.path.basename(__file__).rstrip('.py')

    try:
        if cmd=="train":
            main(resourceid,keyname)
        elif cmd=="predict":
            dumppredictions(resourceid,keyname,valnum)
        else:
            assert False,"format wrong"
    finally:
        stop_fetcher()


