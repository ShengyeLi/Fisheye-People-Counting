# -*- coding: utf-8 -*-
"""
Activity-blind application of YOLO v3 on people counting using overhead fisheye camera.
Copyright: Shengye Li, 2019
"""

# # import necessary files of yolo
from google.colab import files 
util = files.upload()       # select util.py for upload
darknet = files.upload()    # select darknet.py for upload

# these two lines are to mount google drive for writing and reading directly
from google.colab import drive      
drive.mount('/content/gdrive')

from __future__ import division
import cv2 as cv
import numpy as np
import sys
import os.path
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle as pkl
import pandas as pd
import random
import time
from util import *
from darknet import Darknet
from scipy import ndimage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import csv

import warnings
warnings.filterwarnings('ignore')

print(cv.__version__)
print(torch.cuda.is_available())

# set the path for model, weights, coco.names files
path = "/content/gdrive/My Drive/Colab Notebooks/"
clasess = load_classes(path + "coco.names")
model = Darknet(path + "yolov3.cfg")
model.load_weights(path + "yolov3.weights")
CUDA = torch.cuda.is_available()    # if gpu is avaliable
readpath = "/content/gdrive/My Drive/framesfor4/"    #path to read images
savepath = "/content/gdrive/My Drive/BF_KMEANS_446_421/"    #path to write results
numberofframe = 0

def padding(image,psize):
  # ****************************************************************************
  # function to zero-padding image on its right and bottom side to a square of psize*psize. 
  # input: 1. image: image for zero-padding
  #        2. psize: size of zero-padded image
  # output: image padded to shape of psize*psize*3
  # output is processed for gpu
  # ****************************************************************************
  
  
  xsize = image.shape[0]
  ysize = image.shape[1]
  pd_img = np.zeros((psize,psize,3),dtype = np.uint8)
  pd_img[0:xsize,0:ysize,:] = image
  pd_img = pd_img[:,:,::-1].transpose((2,0,1)).copy()   #reverse order of channels to RGB
  pd_img = torch.from_numpy(pd_img).float().div(255.0).unsqueeze(0)   
  return pd_img

def reverseMap(box,shape):
  # ****************************************************************************
  # function to reverse map detections(BBs) from each focus window to complete fisheye image
  # only center of BB is mapped reversely
  # input: 1. box: detection(BB) from focus window, 
  #           8-dimensional vector, [x,y,w,h,objectness, classId, confidence, anlge]
  #        2. shape: shape of destination fisheye image
  # output: reverse mapped to complete fisheye image
  #         only x,y,angle will be revised in this function
  #         x,y will be changed from coordinates in focus window to coordinates in fisheye image
  #         angle will be changed from focus window's angle position to angle position of center of mapped BB
  # ****************************************************************************
  
  
  angle = box[7]      # the angle of focus window
  zx = int(shape[0]/2)  # shape of fisheye image
  zy = int(shape[1]/2)
  ox = 624     # (left,top) of a focus window(1300*800) from top-center of a fisheye image(2048*2048)
  oy = 0
  # calculate new x,y for reverse-mapped BB
  alpha = math.radians(-angle)
  rotationMatrix = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]])
  xy = np.array([[box[0]+ox-zx],[zy-box[1]-oy]])
  xy_ = np.matmul(rotationMatrix, xy)
  xy_ = np.transpose(xy_)
  xy_[0][0] = xy_[0][0] + zx
  xy_[0][1] = zy - xy_[0][1]
  xy_ = xy_.tolist()
  cx = int(xy_[0][0])
  cy = int(xy_[0][1])
  # calculate angle for reverse-mapped BB
  dx = cx - zx
  dy = zy - cy
  if dx <= 0:
    theta = math.atan(dy/min(-0.01,dx)) - math.pi # avoid a/0
  else:
    theta = math.atan(dy/dx)
  theta = math.degrees(theta)
  box[0] = cx
  box[1] = cy
  box[7] = theta
  return box

def topoints(shape,box):
  # ****************************************************************************
  # Get the coordinates of four vertices of BBs in fisheye image
  # input: 1. shape: shape of fisheye image
  #        2. box: BBs to interpret 
  # output: list of cooperates of four vertices of the BB
  # ****************************************************************************
  
  
  # set center of image as origin.
  ox = int(shape[0]/2)
  oy = int(shape[0]/2)
  # get coordinate of center of BB in the new coordinate system
  cx = box[0] - ox
  cy = oy - box[1]
  # calculate four vertices based on a series of tiangular calculations
  theta = math.radians(box[7])
  wc = box[2]*math.cos(theta)/2
  hc = box[3]*math.cos(theta)/2
  ws = box[2]*math.sin(theta)/2
  hs = box[3]*math.sin(theta)/2
  sign = [[-1,1,1,1,],[1,1,-1,1],[1,-1,-1,-1],[-1,-1,1,-1]]
  # output
  pts = []
  for i in sign:
    pts.append((int(cx+i[0]*ws+i[1]*hc)+ox,oy-int(cy+i[2]*wc+i[3]*hs)))
  return(pts)

def drawBB(image,boxes,color):
  # ****************************************************************************
  # function to draw BBs 
  # input: 1. image: image to draw BBs on
  #        2. boxes: Bounding boxes to draw
  #        3. color: color for BB
  # ****************************************************************************
  
  
  for box in boxes:
    pts = topoints(image.shape,box) # get the vertices of BB
    for i in range(0,5):
      cv.line(image,pts[i%4],pts[(i-1)%4],color,3) # link vertices in order

def randomize(image,num):
  # ****************************************************************************
  # function to add randomisty to small image candidate for verification
  # input: 1. image: small image
  #        2. num: number of small image, used to select random seed for gaussian noise
  # output: horizontally flipped small image with noise
  
  # *********************NOT USED IN THESIS*************************************
  # ****************************************************************************
  
  
  image = cv.flip(image, 1) # flip horizontally
  np.random.seed(num)  # select a random seed
  noise = np.random.normal(20,10,image.shape) # generate 2D gaussian noise
  noise[noise<0]=0 # truncate the noise into range of o to 255
  noise[noise>255] = 255
  for i in range(0,3):
    image[:,:,i] = image[:,:,i] #+ noise[:,:,i]    #add the noise to image
    cv.imwrite(savepath+str(num)+"_"+str(i)+".jpg",image)
  return image

def NMS(boxes, shape ,th):
  # ****************************************************************************
  # function to merge reverse-mapped boxes using NMS-based method
  # detail can be found in 3.6.1 of Shengye's thesis
  # pseudo code avaibliable in Algorithm 1 of my thesis
  # input: 1. boxes: all BBs after reverse mapping
  #        2. shape: shape of fisheye image
  #        3. th: threshold for IOU. if IOU > th, two BBs will be merged.
  # output: a group of BBs of which any pair's IOU < th
  # ****************************************************************************
  
  
  xsize = shape[0]
  ysize = shape[1]
  confidences = [] #list to store confidence score 
  result = [] # list of output, initialized here
  # extract the confidence score from input BBs, and sort it in descending order.
  for box in boxes:
    confidences.append(box[6])
  idxs = sorted(range(len(confidences)), key=lambda k: confidences[k],reverse=True)
  # NMS 
  for ii in range(len(idxs)):
    # if output list is empty, append the BB with highest conf. directly
    if ii == 0:
      result.append(boxes[idxs[ii]]) 
    else:
      
      i = idxs[ii]
      # get the area of rotated rectangle by count pixels
      # In the later version of opencv, NMS for rotated rectangles is available,
      # Strongly suggest to change to that function provided by opencv which calculate IOU anatically.
      refi = np.zeros((xsize,ysize))
      pts = topoints((xsize,ysize),boxes[i])
      contours = np.array([pts[0],pts[1],pts[2],pts[3]])
      cv.fillPoly(refi, pts =[contours], color=(1,1,1)) 
      si = boxes[i][2]*boxes[i][3]
      addable = 1 # flag: if it is able to add the BB into output list
      for j in range(len(result)):
        refj = np.zeros((xsize,ysize))
        pts = topoints((xsize,ysize),result[j])
        contours = np.array([pts[0],pts[1],pts[2],pts[3]])
        cv.fillPoly(refj, pts =[contours], color=(1,1,1))
        sj = si = result[j][2]*result[j][3]
        ref = refj + refi
        so = len(np.argwhere(ref == 2))
        iou = so / (sj + si - so) # calculate IOU
        # if IOU of this BBs and any of outputs(whose confidence socre is higher) > TH, 
        # suppress the BB.
        if iou > th: 
          addable = 0
          break
      if addable == 1:
        # otherwise append BB to the final output.
        result.append(boxes[i])
  return result

def KMEANS(im, bbs):
  # ****************************************************************************
  # function to merge all BBs after reverse mapping using clustering-based method
  # detail can be found in 3.6.2 of shengye's thesis
  # input: 1. im: fisheye images
  #        2. bbs: BBs after reverse mapping
  # output: a list of BBs after merging. 
  # ****************************************************************************
  
  # compose feature vectors
  bbs = np.asarray(bbs)
  features = np.zeros((bbs.shape[0],12)) # compose a 12-dimensional feature vector
  features[:,0] = bbs[:,0]/2048 # normalize x
  features[:,1] = bbs[:,1]/2048 # normalize y
  T = bbs[:,7]
  X = bbs[:,0]
  Y = bbs[:,1]
  W = bbs[:,2]
  H = bbs[:,3]
  # start compose the grayscale histogram for feature vector
  for i, delta in enumerate(T):
    # first rotate the BB to upright position
    M = cv.getRotationMatrix2D((cf.shape[0]/2,cf.shape[1]/2),90-delta,1)
    image = cv.warpAffine(cf,M,(cf.shape[0],cf.shape[1]))
    dx = X[i] - 1024
    dy = 1024 - Y[i]
    alpha = math.radians(90-delta)
    rotationMatrix = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]])
    xy = np.array([[dx],[dy]])
    xy_ = np.matmul(rotationMatrix, xy)
    xy_ = np.transpose(xy_)
    xy_ = xy_.tolist()
    cx = int(xy_[0][0]) + 1024
    cy = 1024 - int(xy_[0][1])
    margin = 0 # get area exactly same as BB
    # calculate the four vertices of rectangular area to extract grayscale histogram
    top = max(cy - margin - int(H[i] / 2),0)
    left = max(cx - margin - int(W[i] / 2),0)
    right = min(cx + margin + int(W[i] / 2),2048)
    bottom = min(cy + margin + int(H[i] / 2),2048)  
    roi = image[top:bottom,left:right,:] # extract area inside of BB from fisheye image
    roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY) # convert to grayscale
    roi.reshape(1,roi.shape[0]*roi.shape[1]) # reshape to 1*n vector
    hist = np.histogram(roi,bins=10,range=(0,260),density=False) # 10-bin historgram
    features[i,2:12] = 0.8 * hist[0] / (roi.shape[0]*roi.shape[1])
  inertia = np.zeros((features.shape[0]))
  costf = np.zeros((features.shape[0]))
  y = range(1,features.shape[0]+1)
  for i in range(1, features.shape[0]+1): 
    kmeans = KMeans(n_clusters = i, random_state=0) # do kmeans with fixed random seed
    kmeans.fit(features) # input features
    inertia[i-1] = kmeans.inertia_  # cost function without regularization
    costf[i-1] = kmeans.inertia_ + 0.0025*i**2 # cost function with regularization
# # plot and save the cost function if needed
#   plt.plot(inertia)
#   plt.savefig(savepath + "Kmeans_" + str(numberofframe) + ".jpg")
#   plt.clf()
#   plt.plot(costf)
#   plt.savefig(savepath + "Cost_" + str(numberofframe) + ".jpg")
#   plt.clf()
#   numofcluster = elbow(costf)
#   print(costf)
#   print(inertia)
  numofcluster = np.argmin(costf) + 1 # find the K with minimum cost
  kmeans = KMeans(n_clusters = numofcluster, random_state=0) # do Kmeans again to cluster
  kmeans.fit(features)
  result = []
  # generate a representeive BB for each cluster(person)
  for i in range(0,numofcluster):
    cx = int(np.mean(features[kmeans.labels_ == i,0])*2048)
    cy = int(np.mean(features[kmeans.labels_ == i,1])*2048)
    width = int(np.mean(bbs[kmeans.labels_ == i,2]))
    height = int(np.mean(bbs[kmeans.labels_ == i,3]))
    dx = cx - 1024
    dy = 1024 - cy
    if dx <= 0:
      theta = math.atan(dy/min(-0.01,dx)) - math.pi
    else:
      theta = math.atan(dy/dx)
    result.append([cx,cy,width,height,0,0,0,math.degrees(theta)])
  return result

# list to count time
time_detection = []
time_counting = []
time_verification = []

# main steps begin here
for numberofframe in range(1,1772,30): 
  cf = cv.imread(readpath + str(numberofframe) + ".jpg")
  begin = time.time() # time counting begins
  loaded_ims = [] #images feed into YOLO
  psize = 0 # find the size for reshae 
  
  # get roi---------------------------------------------------------------------
  for angle in range(0, 360, 15):
    # rotate image by 15 degree, and extract a window of 800*1300 from its top-center 
    M = cv.getRotationMatrix2D((cf.shape[0]/2,cf.shape[1]/2),angle,1)
    rotation = cv.warpAffine(cf,M,(cf.shape[0],cf.shape[1]))
    roi = rotation[0:1300,624:1424,:]
    loaded_ims.append(roi)
    psize = max(roi.shape[0],roi.shape[1],psize)
    
  # prepart batch---------------------------------------------------------------
  batch_size = 2 # divide 24 images into bathes, and feed into YOLO
  psize = int(np.ceil(psize/32)*32) # image feed into YOLO must have a shape of n*32
  model.net_info["height"] = int(psize) # set input size for darknet 
  model.cuda() 
  im_batches = list(map(padding, loaded_ims, [psize for x in range(len(loaded_ims))])) # zero-padding images
  im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
  im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
  leftover = 0
  if (len(im_dim_list) % batch_size):
    leftover = 1
  if batch_size != 1:
    num_batches = len(loaded_ims) // batch_size + leftover            
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size, 
                                                           len(im_batches))]))  for i in range(num_batches)]  
  im_dim_list = im_dim_list.cuda()
  
  # detection-------------------------------------------------------------------
  persons = [] # list to store people detection result
  for i, batch in enumerate(im_batches):
    batch = batch.cuda()
    with torch.no_grad():
      prediction = model(Variable(batch), CUDA)
    # parameters for write_result(prediction, objectness, number of classes, TH of IOU for NMS)  
    prediction = write_results(prediction, 0.3, 80, nms_conf = 0.4)
    if type(prediction) != int:
      for pred in prediction:
        detection = pred.cpu().numpy()
        ind = int(pred[0])
        factor = 1312 / psize # zoom in/zoom out factor, if image feed into yolo is down-sampled, this is necessary
        cx = int(factor * (detection[1] + detection[3])/2)
        cy = int(factor * (detection[2] + detection[4])/2)
        width = int(factor * (detection[3] - detection[1]))
        height = int(factor * (detection[4] - detection[2]))
        objectness = detection[5]
        confidence = detection[6]
        classId = int(detection[7])
        top = cy - height / 2
        left = cx - width / 2
        right = cx + width / 2
        bottom = cy + height / 2
        box = [cx, cy, width, height, objectness, classId, confidence, (i*batch_size+ind)*15]
        #      0    1    2       3         4         5         6               7
        if classId == 0:  # only people detection is accepted here
          if left > 50 and right < 750 and bottom < 1250: # spatial outlier rejection with delta = 50
          persons.append(box)
#             print(box)
  
  # Reverse Map-----------------------------------------------------------------
  im = cf.copy()
  BBs = list(map(reverseMap, persons, [(2048,2048) for x in persons])) # reverse map
  drawBB(im, BBs, (255,0, 0)) # draw result of reverse map in blue
  
  stop = time.time()
  time_detection.append(stop-begin)
#   print(stop-begin)
  
  # People counting ------------------------------------------------------------
  BBs = NMS(BBs, (2048, 2048), 0.4)   # choose between NMS and KMEANS
#   BBs = KMEANS(cf, BBs)
  drawBB(im, BBs, (0, 255,0))
#   drawBB(im, BBs, (0,0, 255))
  stop1 = time.time()
#   print(stop1-stop)
  time_counting.append(stop1-stop)
  

# varification------------------------------------------------------------------
  verified_result = [] # list to store verified result
  for numberofbox, box in enumerate(BBs):
    # prepare batch
    positive = 0    # number of "Yes" vote
    psize = 0       # size for zero-padding
    loaded_ims = [] 
    candidates = []
    # rotate BB to up-right position
    M = cv.getRotationMatrix2D((cf.shape[0]/2,cf.shape[1]/2),90-box[7],1)
    image = cv.warpAffine(cf,M,(cf.shape[0],cf.shape[1]))
    dx = box[0] - 1024
    dy = 1024 - box[1]
    alpha = math.radians(90-box[7])
    rotationMatrix = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]])
    xy = np.array([[dx],[dy]])
    xy_ = np.matmul(rotationMatrix, xy)
    xy_ = np.transpose(xy_)
    xy_ = xy_.tolist()
    cx = int(xy_[0][0]) + 1024
    cy = 1024 - int(xy_[0][1])
    pts = topoints((2048,2048),[cx,cy,box[2],box[3],0,0,0,90-box[7]])   
    # extract margin slightly larger than BB
    margin = 30
    top = max(cy - margin - int(box[3] / 2),0)
    left = max(cx - margin - int(box[2] / 2),0)
    right = min(cx + margin + int(box[2] / 2),2048)
    bottom = min(cy + margin + int(box[3] / 2),2048)    
    roi_0 = image[top:bottom,left:right,:] # extract the upright image 
    roi_n10 = ndimage.rotate(roi_0, -15) # further rotate -15 degrees
    roi_p10 = ndimage.rotate(roi_0, 15) # future rotate 15 degrees
    
    loaded_ims.append(randomize(roi_n10,0))
    loaded_ims.append(randomize(roi_0,1))
    loaded_ims.append(randomize(roi_p10,2))
                                 
    for mmm, roi in enumerate(loaded_ims):
      psize = max(roi.shape[0],roi.shape[1],psize) 
#       cv.imwrite(savepath + "verification_"+str(numberofbox)+"_"+str(mmm)+".jpg", roi)
    batch_size = 3
    psize = int(np.ceil(psize/32)*32) # repeat steps in detection part
    model.net_info["height"] = int(psize)
    model.cuda()
    im_batches = list(map(padding, loaded_ims, [psize for x in range(len(loaded_ims))]))
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    leftover = 0
    if (len(im_dim_list) % batch_size):
      leftover = 1
    if batch_size != 1:
      num_batches = len(loaded_ims) // batch_size + leftover            
      im_batches = [torch.cat((im_batches[i*batch_size : min((i + 1)*batch_size, 
                                                             len(im_batches))]))  for i in range(num_batches)]  
    im_dim_list = im_dim_list.cuda()
    
    # detection
    for i, batch in enumerate(im_batches):
      batch = batch.cuda()
      with torch.no_grad():
        prediction = model(Variable(batch), CUDA)
      prediction = write_results(prediction, 0.3, 80, nms_conf = 0) # set TH of IOU for NMS as 0
      if type(prediction) != int:
        for pred in prediction:
          detection = pred.cpu().numpy()
          ind = int(pred[0])
          factor = 1
          cx = int(factor * (detection[1] + detection[3])/2)
          cy = int(factor * (detection[2] + detection[4])/2)
          width = int(factor * (detection[3] - detection[1]))
          height = int(factor * (detection[4] - detection[2]))
          objectness = detection[5]
          confidence = detection[6]
          classId = int(detection[7])
          veri = [cx, cy, width, height, objectness, classId, confidence, (i*batch_size+ind)*15-15]
          #      0    1    2       3         4         5         6               7
          if classId == 0 : # if person is detected again 
            candidates.append(veri) 
#             print("numberofbox: ", str(numberofbox))
#             print(veri)
    # vote     
    if len(candidates) >= 2: # majority vote
      verified_result.append(box)
      
  #output number and location
  stop2 = time.time()
  print(numberofframe,end=": ")
  print(stop-begin)
  drawBB(im, verified_result, (0,0,255)) # draw verified detections in red
  time_verification.append(stop2-stop1)
  cv.imwrite(savepath + "result_" + str(numberofframe) + ".jpg", im)
  torch.cuda.empty_cache()

torch.cuda.empty_cache()

# output time counting of each steps
with open(savepath+"time_detection.csv", mode='w') as box_file:
  box_writer = csv.writer(box_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  for i in time_detection:
    box_writer.writerow([i])
with open(savepath+"time_counting.csv", mode='w') as box_file:
  box_writer = csv.writer(box_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  for i in time_counting:
    box_writer.writerow([i])  
with open(savepath+"time_verification.csv", mode='w') as box_file:
  box_writer = csv.writer(box_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  for i in time_verification:
    box_writer.writerow([i])







