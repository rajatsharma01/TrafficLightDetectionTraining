#!/bin/bash

MODEL_DIR=$1
MODEL_CONFIG=$2
TRAIN_DIR=${MODEL_DIR}/train
FINAL_DIR=${MODEL_DIR}/final
CHECKPOINT_PREFIX=`ls ${TRAIN_DIR}/model.ckpt-*.index | tail -1 | awk -F.index '{print $1}'`
OUTPUT_DIR=${MODEL_DIR}/fine_tuned_model
MAX_CHUNK_SIZE=`echo 25*1024*1024 | bc`

rm -rf ${OUTPUT_DIR}/*
rm -rf ${FINAL_DIR}/* 

python ${TENSORFLOW_RESEARCH}/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=${MODEL_CONFIG} \
    --trained_checkpoint_prefix=${CHECKPOINT_PREFIX} \
    --output_directory=${OUTPUT_DIR}

split -d --verbose --bytes=${MAX_CHUNK_SIZE} ${OUTPUT_DIR}/frozen_inference_graph.pb ${FINAL_DIR}/model_chunk_
