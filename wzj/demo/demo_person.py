#coding=utf-8
#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import sys
sys.path.append('/home/xiaosa/install/py-faster-rcnn-master/tools')
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse



CLASSES = ('__background__',
            'person')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def vis_detections_cv(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),3)
        cv2.putText(im, str(score), (int((bbox[0]+bbox[2]))/2,bbox[1]),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)



def demo_peroson(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im = cv2.imread(image_name)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    #scores, boxes = im_detect(net, im)
    scores, boxes = im_detect(net, image_name)
    print ("scores: ",scores.shape)
    print ("boxes: ",boxes.shape)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]   #截取所有框 对应类别的 boxes 信息
        cls_scores = scores[:, cls_ind]                   #截取所有框 对应类别的 score 信息       
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        print (dets.shape)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print (dets.shape)
        #vis_detections(image_name, cls, dets, thresh=CONF_THRESH)
        vis_detections_cv(image_name, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = '../train1/test.prototxt'
    caffemodel = '/home/xiaosa/install/py-faster-rcnn-master/output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_iter_70000.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    # test for img
    # im_path="/home/xiaosa/install/py-faster-rcnn-master/data/VOCdevkit2007/VOC2007/JPEGImages/000021.jpg"
    # im=cv2.imread(im_path)
    # if im is not None:
    #     demo_peroson(net, im)
    #     #plt.show()
    #     cv2.imshow("detect",im)
    #     cv2.waitKey(0)
    # else:
    #     print 'read image failed! '
        
    

    # test for video
    # video_path="/home/xiaosa/install/py-faster-rcnn-master/data/demo/000456.jpg"
    # cameraCapture = cv2.VideoCapture(video_path)
    # while True:
    #     sucess,frame=cameraCapture.read()
    #     if sucess:
    #         demo_peroson(net, im_path)
    #         cv2.imshow("detect",frame)
    #         cv2.waitKey(1)
    #     else:
    #         print 'video over'
    #         break

    # test for camera
    cameraCapture = cv2.VideoCapture(0)
    while True:
        sucess,frame=cameraCapture.read()
        if sucess:
            demo_peroson(net, frame)
            cv2.imshow("detect",frame)
            cv2.waitKey(1)
        else:
            print 'read image failed'
            break
