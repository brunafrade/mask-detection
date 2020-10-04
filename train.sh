#!/bin/sh



/home/caffe-ssd/build/tools/caffe train -solver="train_files/solver_train_full_tuning.prototxt"
-weights snapshot/a2_aug_tuning_iter_15000.caffemodel
-gpu 0

