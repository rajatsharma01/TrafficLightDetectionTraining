#!/bin/bash

python ${TENSORFLOW_RESEARCH}/object_detection/legacy/train.py \
    --train_dir=./models/rfcn_resnet101_coco_2018_01_28_sim/train/ \
    --pipeline_config_path=./configs/rfcn_resnet101_coco_sim.config \
    --logtostderr
