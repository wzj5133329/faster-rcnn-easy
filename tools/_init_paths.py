#coding=utf-8
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

# 将需要的自定义模块加入到sys中 后面就可以直接调用

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'caffe-fast-rcnn', 'python')
#print (caffe_path)
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
#print (lib_path)
add_path(lib_path)
