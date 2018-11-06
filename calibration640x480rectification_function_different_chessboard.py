import numpy as np
import cv2
import glob
import time
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*6,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.



def CalibrateCamera(pathtofiles, debugimage = False):
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(pathtofiles)

    import random
    MAX_IMAGES = 64
    if (len(images) > MAX_IMAGES):
        print("Too many images to calibrate, using {0} randomly selected images"
                .format(MAX_IMAGES))
        images = random.sample(images, MAX_IMAGES)

    h = 0
    w = 0
    for fname in sorted(images):
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
        #print("1")
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            #print("2")
            corners2 = cv2.cornerSubPix(gray,corners,(11,10),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            if(debugimage):
                cv2.imshow(fname,img)
                cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    #mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, None, (w * 2, h * 2), 5)
    #print(ret)
    #print(mtx)
    #print(dist)
    #print(rvecs)
    #print(tvecs)
    return (ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints)


retL, leftCameraMatrix, leftDistortionCoefficients, _, _, leftObjectPoints, leftImagePoints = CalibrateCamera('e:\Projects\Python\stereovision\images\L640\*.jpg', False)

retR, rightCameraMatrix, rightDistortionCoefficients, _, _, rightObjectPoints, rightImagePoints = CalibrateCamera('e:\Projects\Python\stereovision\images\R640\*.jpg', False)
#print("*********************************************")
#print(leftObjectPoints)
#print("*********************************************")
#print(rightObjectPoints)
#print("*********************************************")
#print(leftImagePoints)
#print("*********************************************")
#print(rightImagePoints)
#print("*********************************************")
#cv2.waitKey()
print(retL)
print(retR)

objectPoints = leftObjectPoints

print("Calibrating cameras together...")
#focal length 9 mm
#imageSize = tuple([640, 480])
imageSize = tuple([2 * 640, 2 * 480])

TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)

termination_criteria_extrinsics = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.00001)
(retval, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
#(retval, leftCameraMatrix, leftDistortionCoefficients, rightCameraMatrix, rightDistortionCoefficients, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
        objectPoints, leftImagePoints, rightImagePoints,
        leftCameraMatrix, leftDistortionCoefficients,
        #leftCameraMatrix, leftDistortionCoefficients,
        rightCameraMatrix, rightDistortionCoefficients,
        imageSize, criteria=termination_criteria_extrinsics,
        flags=cv2.CALIB_FIX_INTRINSIC)
        #flags=cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_ZERO_TANGENT_DIST )



#retval, _, _, _, _, rotationMatrix, translationVector, E, F=cv2.stereoCalibrate(objectPoints,  leftImagePoints, rightImagePoints, imageSize,leftCameraMatrix, leftDistortionCoefficients,rightCameraMatrix, rightDistortionCoefficients,flags=cv2.CALIB_FIX_INTRINSIC)



print(retval)
print(rotationMatrix)
print(translationVector)
#imageSize = tuple([2 * 640, 2 * 480])
#(_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(objectPoints, leftImagePoints, rightImagePoints, leftCameraMatrix, leftDistortionCoefficients, rightCameraMatrix, rightDistortionCoefficients,  imageSize, criteria=TERMINATION_CRITERIA, flags=0)
OPTIMIZE_ALPHA = -1
#print(rotationMatrix)
#print(translationVector)
#
print("Rectifying cameras...")
# TODO: Why do I care about the disparityToDepthMap?
(leftRectification, rightRectification, leftProjection, rightProjection,
        dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                leftCameraMatrix, leftDistortionCoefficients,
                rightCameraMatrix, rightDistortionCoefficients,
                imageSize, rotationMatrix, translationVector,
                None, None, None, None, None, cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)
print(leftROI)
#(leftRectification, rightRectification, leftProjection, rightProjection,
#        dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(leftCameraMatrix, leftDistortionCoefficients, rightCameraMatrix, rightDistortionCoefficients,  imageSize, rotationMatrix, translationVector, alpha=0.25)


w = 640
h = 480

print("Saving calibration...")
leftMapX, leftMapY = cv2.initUndistortRectifyMap(leftCameraMatrix, leftDistortionCoefficients, leftRectification, leftProjection, (w * 2, h * 2), 5)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(rightCameraMatrix, rightDistortionCoefficients,  rightRectification, rightProjection, (w * 2, h * 2), 5)
#leftMapX, leftMapY = cv2.initUndistortRectifyMap(leftCameraMatrix, leftDistortionCoefficients, None, None, (w * 2, h * 2), 5)
#rightMapX, rightMapY = cv2.initUndistortRectifyMap(rightCameraMatrix, rightDistortionCoefficients, None, None, (w * 2, h * 2), 5)

print(leftCameraMatrix)
print(rightCameraMatrix)




np.savez_compressed('e:\Projects\Python\stereovision\calib.param', imageSize=[w,h],
        leftMapX=leftMapX, leftMapY=leftMapY,
        rightMapX=rightMapX, rightMapY=rightMapY,
        leftCameraMatrix=leftCameraMatrix,rightCameraMatrix=rightCameraMatrix,
        dispartityToDepthMap = dispartityToDepthMap,
        leftROI=leftROI, rightROI=rightROI)

stereoMatcher = cv2.StereoBM_create(numDisparities=16, blockSize=15)

leftFrame = cv2.imread('e:\\Projects\\Python\\stereovision\\images\\L640\\000700.jpg')
rightFrame = cv2.imread('e:\\Projects\\Python\\stereovision\\images\\R640\\000700.jpg')

cv2.imshow("al", leftFrame)
cv2.imshow("ar", rightFrame)

REMAP_INTERPOLATION = cv2.INTER_LINEAR

fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CAMERA_WIDTH_2 = 640 / 2
CAMERA_HEIGHT_2 = 480 / 2

cropPercent = 1
def getCenterOLD(image, mm):
    #centerY = int(image.shape[0] / 2)
    #centerX = int(image.shape[1] / 2)
    centerX = int(mm[0, 2]*2.0)
    centerY = int(mm[1, 2]*2.0)
    print(centerX)
    print(centerY)
    return image[(centerY - int(CAMERA_HEIGHT_2 * cropPercent)):(centerY + int(CAMERA_HEIGHT_2 * cropPercent)),
           (centerX - int(CAMERA_WIDTH_2 * cropPercent)):(centerX + int(CAMERA_WIDTH_2 * cropPercent))]


def getCenter(image):
    centerY = int(image.shape[0] / 2)
    centerX = int(image.shape[1] / 2)
    #centerX = int(mm[0, 2]*2.0)
    #centerY = int(mm[1, 2]*2.0)
    #print(centerX)
    #print(centerY)
    return image[(centerY - int(CAMERA_HEIGHT_2 * cropPercent)):(centerY + int(CAMERA_HEIGHT_2 * cropPercent)),
           (centerX - int(CAMERA_WIDTH_2 * cropPercent)):(centerX + int(CAMERA_WIDTH_2 * cropPercent))]

fixedLeft = getCenter(fixedLeft)
fixedRight = getCenter(fixedRight)

cv2.imshow('alr', fixedLeft)
cv2.imshow('arr', fixedRight)

grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)

NumDisparities = 32
NumDisparitiesHalf = int(NumDisparities / 2)

stereoMatcher.setMinDisparity(1)
stereoMatcher.setNumDisparities(32)
stereoMatcher.setBlockSize(41)
stereoMatcher.setSpeckleRange(1)
stereoMatcher.setSpeckleWindowSize(100)

depth = stereoMatcher.compute(grayLeft, grayRight)

DEPTH_VISUALIZATION_SCALE = 512

disp = (depth + 0) / DEPTH_VISUALIZATION_SCALE
disp = np.array(disp * 255, dtype=np.uint8)
im_color = cv2.applyColorMap(disp, cv2.COLORMAP_HOT)
#cv2.imshow('depth', im_color)


stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html

# stereoMatcher.setMinDisparity(4)
# stereoMatcher.setNumDisparities(128)
# stereoMatcher.setBlockSize(21)
# stereoMatcher.setSpeckleRange(16)
# stereoMatcher.setSpeckleWindowSize(45)

NumDisparities = 256
NumDisparitiesHalf = int(NumDisparities / 2)

stereo.setMinDisparity(1)
stereo.setNumDisparities(32)
stereo.setBlockSize(41)
stereo.setSpeckleRange(1)
stereo.setSpeckleWindowSize(100)

disparity = stereo.compute(grayLeft, grayRight)

# disparity = disparity[NumDisparitiesHalf:(disparity.shape[0]-NumDisparitiesHalf),
#            (NumDisparities + NumDisparitiesHalf):(disparity.shape[1] - NumDisparitiesHalf)]

# DEPTH_VISUALIZATION_SCALE = 2048
DEPTH_VISUALIZATION_SCALE = 512

disp = (disparity + 0) / DEPTH_VISUALIZATION_SCALE
disp = np.array(disp * 255, dtype=np.uint8)
# disp = cv2.cvtColor(disparity / DEPTH_VISUALIZATION_SCALE, cv2.CV_8UC1)
# https://www.learnopencv.com/applycolormap-for-pseudocoloring-in-opencv-c-python/
im_color = cv2.applyColorMap(disp, cv2.COLORMAP_HOT)
cv2.imshow('depth', im_color)
fixedRight_copy = fixedRight


fixedRight_copy = fixedRight_copy.astype('float')
im_color = im_color.astype('float')
additionF = (fixedRight_copy+im_color)/2
addition = additionF.astype('uint8')

cv2.imshow('colored', addition)
cv2.waitKey(0)
cv2.destroyAllWindows()
