# OPTIONS:
# --weights
# --source          : Which file to analyze
# --img-size        : Must be multiple of 32
# --conf-thres      
# --iou-thres
# --device
# --save-dir
# --augment
# --update

# Default test file found in the original repo
# python3 tools/demo.py --device gpu --source /home/autobuntu/Documents/YOLOP/inference/videos/1.mp4

# Using the video collected with the Van (Autobuntu)
python3 tools/demo.py --img-size 1920 --device gpu --source /home/autobuntu/Videos/Van_Exported/2022-06-17-10-49-09.bag_right_camera_RAW.mp4


