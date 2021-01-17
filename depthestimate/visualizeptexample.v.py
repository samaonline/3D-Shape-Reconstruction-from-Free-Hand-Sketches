import show3d_balls as show3d
# import show3d
import cv2
import numpy as np
import ctypes as ct
import cv2
import pickle
import matplotlib.pyplot as plt
import sys
import point_cloud_utils as pcu

colorflag = 0
showz = 0


def heatmap(x, normalize=True):
    if normalize:
        x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    r = (x * 3 - 1).clip(0, 1)
    g = (1.5 - np.abs(x - 0.5) * 3).clip(0, 1)
    b = (2 - x * 3).clip(0, 1)
    return cv2.merge([b, g, r])


dists = []

if __name__ == "__main__":
    fin = open(sys.argv[1],'rb')
    while True:
        try:
            bno, data, ptcloud, pred, distmap = pickle.load(fin)

            print(bno)
        except EOFError:
            break

        for i in range(len(data)):
            def updatecolor():
                global c0, c1, c2, showpoints
                if colorflag:
                    showpoints = np.vstack([ptcloud[i][::1], pred[i]])

                    value = distmap[i] / distmap[i].max()
                    rgb = np.zeros((len(value), 3), dtype='float32')
                    rgb[:, 2] = (value * 2).clip(0, 1) * 255
                    rgb[:, 1] = (2 - value * 2).clip(0, 1) * 255
                    c0 = np.hstack([rgb[:, 1][::1], np.zeros(len(pred[i]))])
                    c1 = np.hstack([rgb[:, 2][::1], np.zeros(len(pred[i]))])
                    c2 = np.hstack([rgb[:, 0][::1], np.ones(len(pred[i]))])

                else:
                    showpoints = np.vstack([pred[i]])
                    
                    a= pred[i]
                    b = ptcloud[i]
                    hausdorff_a_to_b = pcu.hausdorff(a, b)
                    hausdorff_b_to_a = pcu.hausdorff(b, a)
                    hausdorff_dist = max(hausdorff_a_to_b, hausdorff_b_to_a)
                    print("hausdorff_dist",hausdorff_dist)

                    M = pcu.pairwise_distances(a, b)
                    w_a = np.ones(a.shape[0],dtype='float32')
                    w_b = np.ones(b.shape[0],dtype='float32')
                    P = pcu.sinkhorn(w_a, w_b, M, eps=1e-3)
                    sinkhorn_dist = (M * P).sum()
                    print("sinkhorn_dist",sinkhorn_dist)

                    '''
                    if showz:
                        value = (np.linspace(0, 1, len(pred[i])) < 0.25) * 1.0
                        rgb = np.zeros((len(value), 3), dtype='float32')
                        rgb[:, 2] = (value * 3 - 1).clip(0, 1) * 255
                        rgb[:, 1] = (1.5 - np.abs(value - 0.5) * 3).clip(0, 1) * 255
                        rgb[:, 0] = (2 - value * 3).clip(0, 1) * 255
                        c0 = np.hstack([rgb[:, 1][::1]])
                        c1 = np.hstack([rgb[:, 2][::1], ])
                        c2 = np.hstack([rgb[:, 0][::1]])
                    else:
                        c0 = None
                        c1 = None
                        c2 = None
                    '''

            updatecolor()

            '''
            def big(x):
                return cv2.resize(x, (0, 0), fx=4, fy=4, interpolation=0)

            
            rsz = int(pred[i].shape[0] ** 0.5 + 0.5)
            cv2.imshow('x', big(heatmap(pred[i].reshape((rsz, rsz, 3))[:, :, 0])))
            cv2.imshow('y', big(heatmap(pred[i].reshape((rsz, rsz, 3))[:, :, 1])))
            cv2.imshow('z', big(heatmap(pred[i].reshape((rsz, rsz, 3))[:, :, 2])))
            cv2.imshow('data', data[i])
            
            while True:
                cmd = show3d.showpoints(showpoints, c0=c0, c1=c1, c2=c2, waittime=100,
                                        magnifyBlue=(0 if colorflag == 1 else 0),
                                        background=((128, 128, 128) if colorflag == 1 else (0, 0, 0)),
                                        ballradius=(2 if colorflag == 1 else 12)) % 256
                if cmd == ord('c'):
                    colorflag = 1 - colorflag
                    updatecolor()
                if cmd == ord('z'):
                    showz = 1 - showz
                    updatecolor()
                if cmd == ord(' '):
                    break
                if cmd == ord('q'):
                    break
            if cmd == ord('q'):
                break
            '''