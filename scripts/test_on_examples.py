import os, sys
import cv2
import numpy as np
import caffe
from glob import glob
def transform_input(img, transpose=True, dtype=np.float32):
    """Return transformed input image 'img' for CNN input
    transpose: if True, channels in dim 0, else channels in dim 2
    dtype: type of returned array (sometimes important)    
    """
    inpt = cv2.resize(img, (300,300))
    inpt = inpt - 127.5
    inpt = inpt / 127.5
    inpt = inpt.astype(dtype)
    if transpose:
        inpt = inpt.transpose((2, 0, 1))
    return inpt
    
def transform_output(img, output):
    """Extract bbox info from NN output
    img: original image
    output: NN output
    returns: boxes(list of array[4]), classes(list of int), confidence(list of float)"""
    h,w = img.shape[:2] 
    boxes = (output['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])).astype(int)
    classes = output['detection_out'][0,0,:,1].astype(int)
    confidence = output['detection_out'][0,0,:,2]
    if (len(confidence)==1) and (confidence[0]<0):
        return [],[],[]
    return boxes, classes, confidence
classes_dict = {1: 'mask', 2: 'nomask'}
if __name__ == "__main__":
    # Test Net on several images from images/input, draw bboxes, save to images/output
    path = '/mnt/Storage/Polsec/datasets/Model_full/Mafa/test-images/jpg/' 
    #path = '../datasets/lfw_syntetic_mask/jpg/' 
    save_path = '../datasets/Model_full/Wider_original/WIDER_train/result/'
    save_path = 'result/'
    # proto = sys.argv[1]    
    
    #net = caffe.Net(proto, 'models/tmp/test.caffemodel', caffe.TEST)
    net = caffe.Net('models/deploy/ssd-mask.prototxt', 'models/deploy/ssd-mask.caffemodel', caffe.TEST)    
    
    
    images = glob(path+"*.jpg")
    margin=2
    for p in images:
        
        im = cv2.imread(p)
        img_size = np.asarray(im.shape)[0:2]
        inpt = transform_input(im)
        net.blobs['data'].data[...] = inpt
        output = net.forward() 
        boxes, classes, confidence = transform_output(im, output)
        
        
        for box, cls, conf in zip(boxes, classes, confidence):
            box[0] = np.maximum(box[0]-margin/2, 0)
            box[1] = np.maximum(box[1]-margin/2, 0)
            box[2] = np.minimum(box[2]+margin/2, img_size[1])
            box[3] = np.minimum(box[3]+margin/2, img_size[0])
            col = float(conf)
            col = (col*np.array([0,255,0]) + (1-col)*np.array([0,0,255])).astype(int)
            col = [int(c) for c in col]
            cropped = im[box[1]:box[3],box[0]:box[2],:]
            aligned_ = cv2.resize(cropped, (160, 160),interpolation=cv2.INTER_CUBIC)

            if int(cls)	 == 1:
                color = (0, 255, 0)
            else:
                color = (0, 0 , 255)
            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(im, '{0}: {1:.2f}'.format(classes_dict[int(cls)], conf * 100), (int(box[0]), int(box[1]-.01*box[2])), cv2.FONT_HERSHEY_SIMPLEX, (.004*box[3]), color)

        cv2.imwrite(save_path+p.split("/")[-1], im)
        
