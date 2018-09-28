#!/bin/bash

MODEL_DIR="./models/rfcn_resnet101_coco_2018_01_28_real"
TRAIN_DIR=${MODEL_DIR}/train
CHECKPOINT_PREFIX=`ls ${TRAIN_DIR}/model.ckpt-*.index | tail -1 | awk -F.index '{print $1}'`
OUTPUT_DIR=${MODEL_DIR}/fine_tuned_model
rm -rf ${OUTPUT_DIR}

python ${TENSORFLOW_RESEARCH}/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=./configs/rfcn_resnet101_coco_real.config \
    --trained_checkpoint_prefix=${CHECKPOINT_PREFIX} \
    --output_directory=${OUTPUT_DIR}
