#!/bin/bash

OUTPUT_RECORD="./data/real_data.record"
rm -f ${OUTPUT_RECORD}

python tf_record_generator.py \
    --input_yaml_path="./data/real_training_data/real_data_annotations.yaml" \
    --output_record_path=${OUTPUT_RECORD} \
    --img_height=1096 \
    --img_width=1368
