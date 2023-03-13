#!/usr/bin/bash
python concatenate_datasets.py --datasets ../datasets/picam_data/extracted_data/Tierankatu_nodrone_low_fp \
  ../datasets/picam_data/extracted_data/Tierankatu_orangedrone_low_fps \
  ../datasets/picam_data/extracted_data/Tierankatu_tello_low_fps \
  ../datasets/picam_data/extracted_data/Hevolinna1 \
  ../datasets/picam_data/extracted_data/Hevolinna2 \
  ../datasets/picam_data/extracted_data/Narnia \
  --copy
