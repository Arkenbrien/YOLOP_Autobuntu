# OPTIONS:
# --weights
# --source          : Which file to analyze
#                   : Can be used to connect to a camera - set as 0
# --img-size        : Must be multiple of 32; vvv helpful list
#                   : 0 320 640 960 1280 1600 1920
# --conf-thres      : Default 0.25
# --iou-thres
# --device
# --save-dir        : Where to save the final product
# --augment
# --update

# Default test file found in the original repo
# python3 tools/demo.py --device gpu --source /home/autobuntu/Documents/YOLOP/inference/videos/1.mp4

# Using the video collected with the Van (Autobuntu)
python tools/demo_ROS.py --img-size 640 --device gpu --source 0 --save-dir /home/autobuntu/Videos/YOLOP_OUTPUT


