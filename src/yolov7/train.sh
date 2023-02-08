#!/usr/bin/zsh
python train.py --workers 8 --device 0 --batch-size 1 --data data/picam_al.yaml --img 1280 1280 --cfg cfg/training/yolov7x-picam.yaml --weights weights/yolov7x_training.pt --name yolov7x_picam_320 --hyp data/hyp.scratch.custom.yaml --epochs 100
