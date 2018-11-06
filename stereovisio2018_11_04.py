from imutils.video import FPS
import numpy as np
import cv2
import time

#calibration = np.load('e:\Projects\Python\stereovision\calib_no_rectification.param.npz', allow_pickle=False)
calibration = np.load('e:\Projects\Python\stereovision\calib.param.npz', allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
leftCameraMatrix = calibration["leftCameraMatrix"]
rightCameraMatrix = calibration["rightCameraMatrix"]
dispartityToDepthMap = calibration["dispartityToDepthMap"]
leftROI = tuple(calibration["leftROI"])
rightROI = tuple(calibration["rightROI"])
# crop the image




print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

vs.set(3,1280)
vs.set(4,480)

time.sleep(2.0)
fps = FPS().start()

cropPercent = 1
CAMERA_WIDTH = imageSize[0]
CAMERA_HEIGHT = imageSize[1]

CAMERA_WIDTH_2 = CAMERA_WIDTH / 2
CAMERA_HEIGHT_2 = CAMERA_HEIGHT / 2

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

while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    ret, frame = vs.read()
    #frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    #frame1 = frame[0:40, 0:40]
    imgL = frame[0:480, 0:640]
    imgR = frame[0:480:, 640:(2 * 640)]
    cv2.imshow("Stream", frame)

    REMAP_INTERPOLATION = cv2.INTER_LINEAR

    fixedLeft = cv2.remap(imgL, leftMapX, leftMapY, REMAP_INTERPOLATION)
    fixedRight = cv2.remap(imgR, rightMapX, rightMapY, REMAP_INTERPOLATION)

    #fixedLeft = getCenter(fixedLeft)
    #fixedRight = getCenter(fixedRight)

    imgL1 = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    imgR1 = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Left", getCenter(fixedLeft))
    cv2.imshow("Right", getCenter(fixedRight))

    disparity = stereo.compute(imgL1, imgR1)

    threeDImage = cv2.reprojectImageTo3D(disparity, dispartityToDepthMap)

    disparity = getCenter(disparity)

    DEPTH_VISUALIZATION_SCALE = 1024
    disp = (disparity + 0) / DEPTH_VISUALIZATION_SCALE
    disp = np.array(disp * 255, dtype=np.uint8)

    im_color = cv2.applyColorMap(disp, cv2.COLORMAP_HOT)

    cv2.imshow('depth', im_color)
    ########################################
    fixedRight_copy = getCenter(fixedLeft)

    fixedRight_copy = fixedRight_copy.astype('float')
    im_color = im_color.astype('float')
    additionF = fixedRight_copy
    #additionF[:,:,0] = (fixedRight_copy[:,:,0] + im_color[:,:,0])
    #additionF[:, :, 1] = (fixedRight_copy[:, :, 1] + im_color[:, :, 1])
    #additionF[:, :, 2] = (fixedRight_copy[:, :, 2] + im_color[:, :, 2])
    additionF = (1. / 3. * fixedRight_copy + 2./3 *im_color)
    addition = additionF.astype('uint8')

    cv2.imshow('colored', addition)
    ########################################
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
#vs.stop()
cv2.destroyAllWindows()


