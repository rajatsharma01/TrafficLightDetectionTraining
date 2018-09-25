#!/bin/bash

MODEL_DIR="./models/rfcn_resnet101_coco_2018_01_28_sim"
TRAIN_DIR=${MODEL_DIR}/train
CHECKPOINT_PREFIX=`ls ${TRAIN_DIR}/model.ckpt-*.index | tail -1 | awk -F.index '{print $1}'`

python ${TENSORFLOW_RESEARCH}/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=./configs/rfcn_resnet101_coco_sim.config \
    --trained_checkpoint_prefix=${CHECKPOINT_PREFIX} \
    --output_directory=${MODEL_DIR}/fine_tuned_model
