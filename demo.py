# -*- coding:utf-8 -*-
import cv2
import time
import caffe
import argparse
import numpy as np
from PIL import Image
import os
import sys
np.set_printoptions(threshold=sys.maxsize)
id2class = {1: 'Mask', 2: 'UnMask'}
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"


def load_caffe_model(prototxt_path, caffemodel_path):
    model = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
    return model

def preprocess( img,target_shape):
    img = cv2.resize(img, target_shape, interpolation=cv2.INTER_CUBIC)

    img = img - 127.5
    img = img * 0.007843
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    return img
def box_union(a, b, i):
    areaA = (a[1] - a[0] + 1.) * (a[3] - a[2] + 1.)
    areaB = (b[1] - b[0] + 1.) * (b[3] - b[2] + 1.)
    return areaA + areaB - i
def box_intersection(a, b):
    xA = max(a[0], b[0])
    yA = max(a[2], b[2])
    xB = min(a[1], b[1])
    yB = min(a[3], b[3])
    return max(0., xB - xA + 1.) * max(0., yB - yA + 1.)
def box_iou( a, b):
        i = box_intersection(a, b)
        return i / box_union(a, b, i)

def transform_output( output, input_resolution, nms_threshold, threshold):
    """Extract bbox info from NN output
    img: original image
    output: NN output
    returns: boxes(list of array[4]), classes(list of int), confidence(list of float)"""
    boxes = output['detection_out'][0,0,:,3:7]
    classes = output['detection_out'][0,0,:,1].astype(int)
    confidence = output['detection_out'][0,0,:,2]
    confidence_result = []
    classes_result = []
    boxes_result = []
    
    for idx0 in range(len(boxes)):
        detection0 = boxes[idx0]
        xmin0 = detection0[0] * input_resolution[1]
        ymin0 = detection0[1] * input_resolution[0]
        xmax0 = detection0[2] * input_resolution[1]
        ymax0 = detection0[3] * input_resolution[0]
        for idx1 in range(idx0 + 1, len(boxes)):
            detection1 = boxes[idx1]
            xmin1 = detection1[0] * input_resolution[1]
            ymin1 = detection1[1] * input_resolution[0]
            xmax1 = detection1[2] * input_resolution[1]
            ymax1 = detection1[3] * input_resolution[0]
            iou = box_iou([xmin0, xmax0, ymin0, ymax0], [xmin1, xmax1, ymin1, ymax1])
            
            if iou > nms_threshold or confidence[idx0] <= threshold:
                continue

            confidence_result.append(confidence[idx0])
            classes_result.append(classes[idx0])
            boxes_result.append(boxes[idx0])
    
    if confidence[len(boxes)-1] > threshold:
        confidence_result.append(confidence[len(boxes)-1])
        classes_result.append(classes[len(boxes)-1])
        boxes_result.append(boxes[len(boxes)-1])
    return boxes_result, classes_result, confidence_result

def inference(image, model,
              conf_thresh=0.05,
              iou_thresh=0.5,
              target_shape=(300, 300),
              draw_result=True,
              show_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''

    output_info = []
    height, width, _ = image.shape


    image_transposed = preprocess(image, target_shape)

    model.blobs['data'].data[...] = image_transposed
    output = model.forward()

    boxes, classes, confidence = transform_output(output, target_shape, iou_thresh, conf_thresh )

    for bb,cls_,conf in zip (boxes, classes, confidence):
        class_id = int(cls_)
        left = max(0, int(bb[0] * width))
        top = max(0, int(bb[1] * height))
        right = min(width, int(
            bb[2] * width))
        bottom = min(height, int(
            bb[3] * height))


        if draw_result:
            if class_id == 1:
                color = (0, 255, 0)
            else:
                color = (0, 0 , 255)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (left + 2, top-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, left, right, top, bottom])

    if show_result:
        cv2.imshow('POLSECAI',image)
        cv2.waitKey(1000)
    return output_info


def run_on_video(video_path, model, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        status, img_raw = cap.read()
        read_frame_stamp = time.time()
        if (status):
            inference(img_raw, model,
                             conf_thresh,
                             iou_thresh=0.5,
                             target_shape=(300, 300), 
                             draw_result=True,
                             show_result=False)
            cv2.imshow('image', img_raw)
            cv2.waitKey(1)
            inference_stamp = time.time()
            # writer.write(img_raw)
            write_frame_stamp = time.time()
            idx += 1
            print("%d of %d" % (idx, total_frames))
            print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                                                                   inference_stamp - read_frame_stamp,
                                                                   write_frame_stamp - inference_stamp))
    # writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument('--img-mode', type=int, default=1, help='set 1 to run on image, 0 to run on video.')
    parser.add_argument('--img-path', type=str, help='path to your image.')
    parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')
    args = parser.parse_args()
    model = load_caffe_model('models/deploy/ssd-mask.prototxt','models/deploy/ssd-mask.caffemodel');

    if args.img_mode:
        inference_stamp = time.time()
        imgPath = args.img_path
        img = cv2.imread(imgPath)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inference(img, model, show_result=True, target_shape=(300, 300),)
        print(time.time()-inference_stamp)
    else:
        video_path = args.video_path
        if args.video_path == '0':
            video_path = 0
        run_on_video(video_path, model, '', conf_thresh=0.05)
