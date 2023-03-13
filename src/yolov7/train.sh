#!/usr/bin/zsh
python train.py --workers 8 --device 0 --batch-size 2 --data data/picam_al.yaml --img 1280 1280 --cfg cfg/training/yolov7-tiny-picam.yaml --weights weights/yolov7-tiny.pt --name yolov7_tiny_full --hyp data/hyp.scratch.custom.yaml --epochs 50
