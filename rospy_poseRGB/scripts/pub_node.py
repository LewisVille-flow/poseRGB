#!/usr/bin/env python
import rospy
from std_msgs.msg import String

import cv2
from PIL import Image
import jetson.inference
import jetson.utils
import sys
import numpy as np
import colorgram as colorgram

###################################################
## Define classes

class opt:
    network = "resnet18-body"
    threshold = 0.15
    overlay = "links,keypoints"
    input_URI = "/dev/video0"
    output_URI = ""

""" Class for converting RGB value to Color Name  """
class ColorOrg:
    """
    input Example:
        _red = [120, 0, 0]
        _black = [35, 35, 35]

	color_name_list = ["Red", "black"]
        color_rgb_list = np.array([_red, _black, _blue, _brown, _gray])
        color_original = ColorOrg(color_list=color_name_list, rgb_list=color_rgb_list)
    """

    def __init__(self, color_list, rgb_list):
        self.name_list = color_list
        self.rgb_list = rgb_list
        self.length = len(self.name_list)

    def GetColorName(self, color_list):
        _color_list = color_list
        rgb_distance_list = []
        for _i in range(self.length):
            _r = self.rgb_list[_i][0]
            _g = self.rgb_list[_i][1]
            _b = self.rgb_list[_i][2]

            rgb_distance = (abs(_color_list[0] - _r))**2 + (abs(_color_list[1] - _g))**2 + (abs(_color_list[2] - _b))**2
            rgb_distance_list.append(rgb_distance)

        min_val = min(rgb_distance_list)
        min_index = rgb_distance_list.index(min_val)

        target_name = self.name_list[min_index]
        
        return target_name

    def PrintColorRGB(self):
        print("entered rgb value is \n", self.rgb_list)

###################################################


def talker(data, _iter):
    RGB_name = data
    _iter_num = _iter
    
    pub = rospy.Publisher('chatter', String, queue_size=10)
    # rate = rospy.Rate(1)

    # RGB_name = "hello world %s" % rospy.get_time()
    rospy.loginfo(RGB_name)
    pub.publish(RGB_name)
    # rate.sleep()

def rgb_extracter():
    # load the pose estimation model
    net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)
    camera = jetson.utils.gstCamera(640, 480, opt.input_URI)

    # create video sources & outputs
    # input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
    # output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

    # process frames until the user exits
    while True:
        # capture the next image
        # img = input.Capture()
        
        
        img, width, height = camera.CaptureRGBA(zeroCopy=True)
        jetson.utils.cudaDeviceSynchronize()
        
        img_numpy = jetson.utils.cudaToNumpy(img, width, height, 4)
        img_numpy1 = cv2.cvtColor(img_numpy.astype(np.uint8), cv2.COLOR_RGBA2BGR)
        
        # cv2.imshow('img', img_numpy1)

        # perform pose estimation (with overlay)
        poses = net.Process(img, overlay=opt.overlay)
        # poses = net.Process(img)

        # print the pose results
        # print("detected {:d} objects in image".format(len(poses)))
        # print("pose len: {:d}".format(len(poses)))
        
        """ 1. If peron is detected? """
        _num_person = len(poses)
        if _num_person < 1:
            print("Detected person number: {:d}".format(_num_person))
            continue

        for pose in poses:
            # print('new pose')
            # print('pose: ', type(pose))
            # print("pose keypoints type: ", type(pose.Keypoints))
            # print("pose keypoints:", pose.Keypoints)

            """ 2. Detect if person is seen from head to the toe. """
            pose_id = []
            for _iter in range(len(pose.Keypoints)):
                pose_id.append(pose.Keypoints[_iter].ID)    
            ## neck id is 17, it is added into pose.Keypoints.
            ## so len of pose.Keypoints increased

            try:
                pose_id.index(1)
                pose_id.index(16)
            except:
                print('No full person is detected')
                continue
            
            """ 3. Extract Keypoints """
                    
            left_shoulder = pose.Keypoints[5]
            right_shoulder = pose.Keypoints[6]
            left_hip = pose.Keypoints[11]
            right_hip = pose.Keypoints[12]
            # print('left_shoulder x: ', left_shoulder.x, 'left_shoulder y: ', left_shoulder.y)
            
            
            """ 4. image crop based on these points """
                # print('width: {:f}, height: {:f} '.format(width, height))
                # 640, 480 
            '''
            print('left_shoulder: {:f} '.format(left_shoulder.x))
            print(int(left_shoulder.x), int(left_shoulder.y))        
            print(int(right_shoulder.x), int(right_shoulder.y))
            print(int(left_hip.x), int(left_hip.y))	
            print(int(right_hip.x), int(right_hip.y))
            '''
            cropped_frame = img_numpy1[int(right_shoulder.y):int(right_hip.y), int(left_shoulder.x):int(right_shoulder.x)]
            
            if int(right_shoulder.y) < int(left_shoulder.y):
                    y_up = int(right_shoulder.y)
            else:
                    y_up = int(left_shoulder.y)
            if int(right_hip.y) < int(left_hip.y):
                    y_down = int(left_hip.y)
            else:
                    y_down = int(right_hip.y)
            if int(right_shoulder.x) < int(left_shoulder.x):
                    x_left = int(right_shoulder.x)	
            else:   
                    x_left = int(left_shoulder.x)
            if int(right_shoulder.x) < int(left_shoulder.x):
                    x_right = int(left_shoulder.x)
            else:
                    x_right = int(right_shoulder.x)
            print('cropping box index: {:d}, {:d}, {:d}, {:d}'.format(y_up, y_down,x_left,x_right))      

            cropped_image = img_numpy1[y_up:y_down, x_left:x_right]       
            
            
            """ 5. Extract RGB value and Convert into Color Name """

            # initialize class
            _brown = [165, 45, 42]
            _gray  = [80, 80, 80]
            _red   = [120, 0, 0]
            _black = [35, 35, 35]
            _blue  = [0, 0, 120]

            color_rgb_list = np.array([_red, _black, _blue, _brown, _gray])
            color_name_list = ["Red", "black", "Blue,", "Brown", "Gray"]
            color_original = ColorOrg(color_list=color_name_list, rgb_list=color_rgb_list)

            """ 5-1. Extract Color """
            _color_num = 2
            pil_image = Image.fromarray(cropped_image)

            colors = colorgram.extract(pil_image, _color_num)
            # error here. numpy array is not the same with image file. array has no read attribute.
            # numpy array into PIL image? find it!
            # done.

            image_rgb = np.empty((0,3), int)
            image_prop = []

            for _i in range(_color_num):

                image_rgb = np.append(image_rgb,np.array([[colors[_i].rgb[0], colors[_i].rgb[1], colors[_i].rgb[2]]]), axis = 0)
                image_prop.append(colors[_i].proportion)

            print(image_rgb)

            """ 5-2. Convert RGB into color name """
            for _iter in range(_color_num):
                _temp = image_rgb[_iter]
                print('image_rgb: ', _temp, 'type: ' ,type(_temp))
                print('image_prop: ',image_prop[_iter])
                target_name = color_original.GetColorName(_temp)
                # print('target name is: ', target_name)
                talker(target_name, _iter)
            
            #####
            # left thing: why hip coordinate is like ankle??



            """ Print and show the Image and Cropped Image """
            cv2.imshow("img", img_numpy1)
            cv2.imshow("img_cropped2", cropped_image)


            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            # print('Links', pose.Links)
        
        
        # render the image
        # output.Render(img)

        # update the title bar
        # output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

        # print out performance info
        net.PrintProfilerTimes()

        # exit on input/output EOS
        #if not input.IsStreaming() or not output.IsStreaming():
        #    break
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        rospy.init_node('rgb_extract_node', anonymous=True)
        rgb_extracter()
    except rospy.ROSInterruptException:
        pass


