from imutils.video import FPS
import numpy as np
import cv2
import time

calibration = np.load('e:\Projects\Python\stereovision\calib.param.npz', allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
leftCameraMatrix = calibration["leftCameraMatrix"]
rightCameraMatrix = calibration["rightCameraMatrix"]
dispartityToDepthMap = calibration["dispartityToDepthMap"]

#dispartityToDepthMap = np.asarray([[1 ,0, 0, -640], [0, 1, 0, -480], [0, 0, 0, 348], [0, 0, 1/60, 0]])

#dispartityToDepthMap[0:3,3] = dispartityToDepthMap[0:3,3] / 2
#dispartityToDepthMap[3,2] = dispartityToDepthMap[3,2] * 100

#dispartityToDepthMap[0,3] = dispartityToDepthMap[0,3] +  320
#dispartityToDepthMap[0,3] = dispartityToDepthMap[1,3] +  220


print(dispartityToDepthMap)
leftROI = tuple(calibration["leftROI"])
rightROI = tuple(calibration["rightROI"])
# crop the image

#print(dispartityToDepthMap)

CAMERA_WIDTH = imageSize[0]
CAMERA_HEIGHT = imageSize[1]

CAMERA_WIDTH_2 = CAMERA_WIDTH / 2
CAMERA_HEIGHT_2 = CAMERA_HEIGHT / 2

cropPercent = 1

def getCenter(image):
    centerY = int(image.shape[0] / 2) - 150
    centerX = int(image.shape[1] / 2)
    #centerX = int(mm[0, 2]*2.0)
    #centerY = int(mm[1, 2]*2.0)
    #print(centerX)
    #print(centerY)
    return image[(centerY - int(CAMERA_HEIGHT_2 * cropPercent)):(centerY + int(CAMERA_HEIGHT_2 * cropPercent)),
           (centerX - int(CAMERA_WIDTH_2 * cropPercent)):(centerX + int(CAMERA_WIDTH_2 * cropPercent))]


# loop over the frames from the video stream

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

#https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html

#Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
stereo.setMinDisparity(1)
#Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.
stereo.setNumDisparities(64)
#Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
stereo.setBlockSize(21)#41
#	Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
stereo.setSpeckleRange(1)
#Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
stereo.setSpeckleWindowSize(100)#100
stereo.setROI1(leftROI)
stereo.setROI2(rightROI)

#leftFrame = cv2.imread('e:\\Projects\\Python\\stereovision\\images\\L003461.jpg')
#rightFrame = cv2.imread('e:\\Projects\\Python\\stereovision\\images\\R003461.jpg')
leftFrame = cv2.imread('e:\\Projects\\Python\\stereovision\\images\\L003000.jpg')
rightFrame = cv2.imread('e:\\Projects\\Python\\stereovision\\images\\R003000.jpg')


REMAP_INTERPOLATION = cv2.INTER_LINEAR

fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)

#fixedLeft = getCenter(fixedLeft)
#fixedRight = getCenter(fixedRight)

imgL1 = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
imgR1 = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)

cv2.imshow("Left", getCenter(fixedLeft))
cv2.imshow("Right", getCenter(fixedRight))
disparity = stereo.compute(imgL1, imgR1)

#disparity = getCenter(disparity)

DEPTH_VISUALIZATION_SCALE = 1024
disp = (disparity + 0) / DEPTH_VISUALIZATION_SCALE
disp = np.array(disp * 255, dtype=np.uint8)

im_color = cv2.applyColorMap(disp, cv2.COLORMAP_HOT)

cv2.imshow('depth', getCenter(im_color))

#disparity = np.array(disparity / 1.0, dtype=np.int)
threeDImage = cv2.reprojectImageTo3D(disparity, dispartityToDepthMap)

threeDImage = getCenter(threeDImage)
disparity = getCenter(disparity)

fixedLeft = getCenter(fixedLeft)

#print(threeDImage.shape)
#http://openpcl.sourceforge.net/


import pptk
import numpy as np
import plyfile
#P = np.random.rand(100,3)
#ra = np.reshape(threeDImage, (640*480,3))
#print(ra.shape)

threeDImage2 = np.empty([640*480, 3])
threeOnDisp  = np.empty([640*480, 3])
rgb = np.empty([640*480, 3])
i = 0

for a in range(480):
    for b in range(640):
        #threeDImage2[i, 0] = threeDImage[a, b, 0]
        #threeDImage2[i, 1] = threeDImage[a, b, 1]
        #threeDImage2[i, 2] = threeDImage[a, b, 2]
        threeOnDisp[i, 2] = -a
        threeOnDisp[i, 0] = -b
        if disp[a, b] > 0.0001:
            threeOnDisp[i, 1] = disparity[a, b]
        else:
            threeOnDisp[i, 1] = -np.inf

        rgb[i, 2] = fixedLeft[a, b, 0]
        rgb[i, 1] = fixedLeft[a, b, 1]
        rgb[i, 0] = fixedLeft[a, b, 2]

        i = i + 1

#6cm distance 9mm

#threeDImage3 = 6*0.9/dispdistance

i = 0
if True:
    for a in range(480):
        for b in range(640):
            threeDImage2[i, 0] = threeDImage[a, b, 0]
            threeDImage2[i, 2] = -threeDImage[a, b, 1]
            threeDImage2[i, 1] = threeDImage[a, b, 2]
            i = i + 1

print(disp)
#np.savetxt("e:\\Projects\\Python\\stereovision\\arr2.txt", threeDImage)
#np.savetxt("e:\\Projects\\Python\\stereovision\\arr.txt", threeDImage2)

rgb = rgb[~np.isinf(threeDImage2).any(axis=1)]
rafilter = threeDImage2[~np.isinf(threeDImage2).any(axis=1)]

#rgb = rgb[~np.isinf(threeOnDisp).any(axis=1)]
#rafilter = threeOnDisp[~np.isinf(threeOnDisp).any(axis=1)]


print(rafilter.shape)
print(rafilter)
v = pptk.viewer(rafilter)
v.attributes(rgb / 255.)
#v = pptk.viewer(ra[0:10000,:])

#data = plyfile.PlyData.read('e:\\Projects\\Python\\stereovision\\ankylosaurus_mesh.ply')['vertex']
#xyz = np.c_[data['x'], data['y'], data['z']]
#rgb = np.c_[data['red'], data['green'], data['blue']]
#n = np.c_[data['nx'], data['ny'], data['nz']]
#v = pptk.viewer(xyz)
#v.attributes(rgb / 255., 0.5 * (1 + n))

key = cv2.waitKey()
