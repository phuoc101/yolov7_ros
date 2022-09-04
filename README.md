# YOLOv7 ROS
This is a ROS package for object detection with [the official Yolov7](https://github.com/WongKinYiu/yolov7)

## Installation
Built and tested on Python3.8, Ubuntu 20.04, ROS Noetic.

* Clone the package and install dependencies for Yolov7 (follow the guide from the official repository, might be better to set up a virtual environment)
  ```bash
  virtualenv --system-site-packages -p python3.8 ~/ros_torch_env
  ```

  ```bash
  cd <YOUR_ROS_WORKSPACE>/src
  git clone https://github.com/phuoc101/yolov7_ros
  cd src/yolov7
  pip install -r requirements.txt
  ```
  * Build the package and source it
  ```bash
  cd <YOUR_ROS_WORKSPACE>
  catkin build yolov7_ros
  source devel/setup.bash #or setup.zsh, depends on what shell you're using
  ```
## Testing Yolov7 with webcam
  ```bash
  cd scripts
  python3 webcam.demo --visualize --show_perf
  ```

## Usage
  ```bash
# default input image topic is /image_raw, but can be modified by specifying input_img_topic:=<YOUR_TOPIC>
  roslaunch yolov7_ros yolov7.launch
# to test with webcam, need to install usb_cam ros package first
  roslaunch yolov7_ros yolov7_webcam.launch
  ```

## Custom weights
  Put the weights you want to use in `yolov7_ros/src/yolov7/weights` and specifying it during launching
  ```bash
  roslaunch yolov7_ros yolov7.launch weights:=weights/<YOUR_WEIGHTS>.pt
# or
  roslaunch yolov7_ros yolov7_webcam.launch weights:=weights/<YOUR_WEIGHTS>.pt
  ```

