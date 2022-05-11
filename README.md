# rospy_poseRGB
## environments
jeton-nano

## requirements
opencv
jetson-inference
colorgram

## Description
You can extract human's upper body clothes color(estimated color RGB)
It does not generate RGB. Input RGB value is manually written inside the code.
To use this more effectively, you need to input example color name, RGB values.

By estimating human body pose(model:jetson-inference - resnet18-body), it detect human torso first. Upper body part(x, y coordinate) will be remain and color of inside the box(the x, y) will be extracted.
# poseRGB
