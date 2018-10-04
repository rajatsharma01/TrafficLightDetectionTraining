#!/bin/bash

MODEL_DIR="./models/rfcn_resnet101_coco_2018_01_28_real"
CONFIG_FILE="./configs/rfcn_resnet101_coco_real.config"
#MODEL_DIR="./models/ssd_mobilenet_v1_coco_2018_01_28_real"
#CONFIG_FILE="./configs/ssd_mobilenet_v1_coco_real.config"

python ${TENSORFLOW_RESEARCH}/object_detection/train.py \
    --train_dir="${MODEL_DIR}/train/" \
    --pipeline_config_path=${CONFIG_FILE} \
    --logtostderr
