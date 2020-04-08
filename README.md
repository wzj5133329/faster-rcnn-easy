# 安装

## caffe 安装
需要使用faster rcnn的caffe版本  但是该版较低可能使用cudnn有问题
[fast rcnn](https://github.com/rbgirshick/caffe-fast-rcnn/tree/0dcd397b29507b8314e252e850518c5695efbb83)      需要pycaffe



## faster rcnn 安装 (该caffe路径下不能有中文)
 
1. Clone the Faster R-CNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git
  ```

2. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```

# 采用end2end方法训练

## 训练步骤

### step1 数据集制作

需要将数据集放在 data/VOCdevkit2007/VOC2007 文件夹下  
├── ./Annotations  #存放xmls文件
├── ./JPEGImages   #存放图片文件  
├── ./gen_txt.py   #生成ImageSets文件夹下各个txt脚本
├── ./ImageSets  
│   └── ./ImageSets/Main  
│       ├── ./ImageSets/Main/test.txt   
│       ├── ./ImageSets/Main/train.txt  
│       ├── ./ImageSets/Main/trainval.txt  
│       └── ./ImageSets/Main/val.txt  


### step2 模型文件修改

#### 模型修改（主要与检测类别有关）
train.prototxt  修改四处  
层(input-data,roi-data，cls_score，bbox_pred)

test.prototxt   修改两处  
层(cls_score，bbox_pred)

#### 源码修改

1. lib/datasets/pascal_voc.py

self._classes()改成自己的类别

2. 

### step3 开始训练

可以将以下文件拷贝到同一个文件夹中  
train.prototxt  
test.prototxt  
solver.prototxt  
experimentsscripts  
faster_rcnn_end2end.sh  
faster_rcnn_end2end.yml   #可以用来修改config.py中的设置值
test_net.py/train_net.py 

修改 faster_rcnn_end2end.sh中各个文件路径
修改 lib/fast_rcnn/config.py文件中保存caffemodel文件间隔次数，也可以在yml文件中设置

在训练时训练ZF前部分网络的batch_size是1


### step4 模型测试

faster_rcnn_end2end.sh  中设置了模型训练完成后测试最后保存的模型文件，也可以单独测试某个文件  
python2 ./test_net.py --gpu 0 --def test.prototxt --net your.caffemodel --imdb voc_2007_test --cfg faster_rcnn_end2end.yml

### step5 demo

cd wzj/demo  
python2 demo_person.py

## 常见错误

1.
Traceback (most recent call last):
  File "./train_net.py", line 113, in <module>
    max_iters=args.max_iters)
  File "/home/xiaosa/install/py-faster-rcnn-master/tools/../lib/fast_rcnn/train.py", line 160, in train_net
    model_paths = sw.train_model(max_iters)
  File "/home/xiaosa/install/py-faster-rcnn-master/tools/../lib/fast_rcnn/train.py", line 101, in train_model
    self.solver.step(1)
  File "/home/xiaosa/install/py-faster-rcnn-master/tools/../lib/rpn/proposal_target_layer.py", line 66, in forward
    rois_per_image, self._num_classes)
  File "/home/xiaosa/install/py-faster-rcnn-master/tools/../lib/rpn/proposal_target_layer.py", line 191, in _sample_rois
    _get_bbox_regression_labels(bbox_target_data, num_classes)
  File "/home/xiaosa/install/py-faster-rcnn-master/tools/../lib/rpn/proposal_target_layer.py", line 127, in _get_bbox_regression_labels
    bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
TypeError: slice indices must be integers or None or have an __index__ method

原因是似乎最新版本的numpy（1.12.0）不支持将浮点数用作索引 

解决方法： 可以降低numpy版本 
也可以一个更简单的解决方案是lib/proposal_target_layer.py
将以下行添加到126 行之后，
start=int(start)
end=int(end)
在第166行之后，
fg_rois_per_this_image=int(fg_rois_per_this_image)


2. 运行faster rcnn时，发生“protobuf'module' object has no attribute 'text_format'”，是因为protobuf的版本发生了变化  
解决方法：  
在文件./lib/fast_rcnn/train.py增加一行import google.protobuf.text_format 即可解决问题

3. 'NoneType' object has no attribute 'text'  
一般是xml没有相应的支点

4. difficult = np.array([x['difficult'] for x in R]).astype(np.bool)  测试时候  
可能是之前将difficult注释掉了，需要重新生成


# 参考(改写了一点点，翻译)

[faster rcnn](https://github.com/rbgirshick/py-faster-rcnn.git)
