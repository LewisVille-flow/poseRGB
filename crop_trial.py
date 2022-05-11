import cv2
import jetson.inference
import jetson.utils
import numpy as np

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.gstCamera(640, 480, "/dev/video0")

while True:
   img, width, height = camera.CaptureRGBA(zeroCopy = True)
   jetson.utils.cudaDeviceSynchronize()
   
   aimg = jetson.utils.cudaToNumpy(img, width, height, 4)
   aimg1 = cv2.cvtColor(aimg.astype(np.uint8), cv2.COLOR_RGBA2BGR)
   
   crop_frame = aimg1[100:200,100:200]
   
   cv2.imshow('test', aimg1)
   cv2.imshow('crop test', crop_frame)
   
   key = cv2.waitKey(1)
   if key == ord('q'):
       break
