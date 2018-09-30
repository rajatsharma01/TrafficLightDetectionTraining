#!/bin/bash

if [ ! -d ./ros_images ]; then
    mkdir ./ros_images
fi

cd ./ros_images
rm -f left*.jpg
rosrun image_view image_saver _sec_per_frame:=0.01 image:=/image_raw
cd -
