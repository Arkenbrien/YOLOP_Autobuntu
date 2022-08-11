import glob
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import math
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils import letterbox_for_img, clean_str

import cv2
import numpy as np
import roslib
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import warnings

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

class LoadROSStream:

    print("LoadROSStream class reached")

    rospy.init_node('yolop')

    def __init__(self, opt):

        self.mode   = 'ros_stream'

        print("class __init__ def reached")

        print(opt.source)

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(opt.source, Image, self.callback, )

        self.image_pub_opencv = rospy.Publisher(opt.source+'_bgr_opencv',Image, queue_size=1)
        self.image_pub= rospy.Publisher(opt.source+'_rgb',Image, queue_size=1)

    def callback(self,data):

        print("class_callback def reached")

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data,desired_encoding= "bgr8")
            self.yolo_img = cv2.cvtColor(self.cv_image,cv2.COLOR_BGR2RGB)

            self.image_pub_opencv.publish(self.bridge.cv2_to_imgmsg(self.yolo_img, "bgr8"))
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.cv_image, "bgr8"))

        except CvBridgeError as e:
            print(e)

        # return self.sources, self.image_pub

    """ def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years """



"""

class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, auto=True):
        self.mode = 'stream'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print('')  # newline

        # check for common shapes

        s = np.stack([letterbox_for_img(x, self.img_size, auto=self.auto)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()

        h0, w0 = img0[0].shape[:2]
        img, _, pad = letterbox_for_img(img0[0], self.img_size, auto=self.rect and self.auto)

        # Stack
        h, w = img.shape[:2]
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        # Convert
        #img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0[0], None, shapes

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years

"""