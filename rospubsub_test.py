# !/usr/bin/env python
from __future__ import print_function
from cmath import e

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import warnings

class image_converter(object):
    def __init__(self, topic):
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(topic, Image, self.callback, )

        self.image_pub_opencv = rospy.Publisher(topic+'_bgr_opencv',Image, queue_size=1)
        self.image_pub= rospy.Publisher(topic+'_rgb',Image, queue_size=1)

    def callback(self,data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data,desired_encoding= "bgr8")
            self.yolo_img = cv2.cvtColor(self.cv_image,cv2.COLOR_BGR2RGB)

            self.image_pub_opencv.publish(self.bridge.cv2_to_imgmsg(self.yolo_img, "bgr8"))
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.cv_image, "bgr8"))

        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    rospy.init_node('image_converter', anonymous=True)

    # frame_rate = rospy.get_param('/frame_rate')
    # topic_list = rospy.get_param('/topic_list')

    frame_rate = 24
    topic_list = ["/camera_fr/image_raw"]

    image_converter_list = []
    for im_topic in topic_list:
        image_converter_list.append(image_converter(im_topic))

    time.sleep(1)
    count = 0
    while not rospy.is_shutdown():
        for ic in image_converter_list:
            img = ic.cv_image
            count += 1
        time.sleep(1/frame_rate)
        # rospy.spin()

    # sys.exit(1)
