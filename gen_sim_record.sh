#!/bin/bash

OUTPUT_RECORD="./data/sim_data.record"
rm -f ${OUTPUT_RECORD}

python tf_record_generator.py \
    --input_yaml_path="./data/sim_training_data/sim_data_annotations.yaml" \
    --output_record_path=${OUTPUT_RECORD} \
    --img_height=600 \
    --img_width=800
