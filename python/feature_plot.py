# -*- coding: utf-8 -*-
"""
Junior

Plotting local features as clusters

===================================
Demo of HDBSCAN clustering algorithm
===================================
Finds a clustering that has the greatest stability over a range
of epsilon values for standard DBSCAN. This allows clusterings
of different densities unlike DBSCAN.
"""
print(__doc__)

import cv2

import numpy as np

from hdbscan import HDBSCAN
from sklearn import metrics

import time

def getClusterIdx(labels, label):
    lst = []
    
    for i in range(len(labels)):
        if labels[i] == label:
            lst.append(i)
            
    return lst

def getKeypoints(keypoints, idx):
    kps = []
    for i in idx:
        kps.append(keypoints[i])
    return kps

def decomposeDescriptors(desc):
    shape = desc.shape
    dec = [[], []]
    for i in range(shape[0]):
        x, y = 0.0, 0.0
        row = desc[i]
        for j in range(0, shape[1], 2):
            x = x + row[j]
            y = y + row[j+1]
        dec[0].append(x)
        dec[1].append(y)
    return dec
    
def distance(desc):
    ground = desc[1]
    dist = []
    for row in desc:
        d = np.linalg.norm(row - ground)
        dist.append(d)
    return dist

def showCluster(img, label):
    idx = getClusterIdx(labels1, label)
    kps = getKeypoints(keypoints, idx)
    img2 = cv2.drawKeypoints(image,kps,None,(255,0,0), cv2.DRAW_MATCHES_FLAGS_)
    plt.imshow(img2)
    plt.show()
    
image = cv2.imread('/home/ojmakh/programming/phd/data/vocount/test/1 frame.jpg')
surf = cv2.xfeatures2d.SURF_create(500)
keypoints, descriptors = surf.detectAndCompute(image, None)

##############################################################################
# Compute DBSCAN
hdb_t1 = time.time()
hdb = HDBSCAN(min_cluster_size=4).fit(descriptors.astype('double'))
labels1 = hdb.labels_
hdb_elapsed_time = time.time() - hdb_t1

lset1 = set(labels1)
