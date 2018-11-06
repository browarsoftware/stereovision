from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
#from matplotlib import pyplot as plt

# construct the argument parse and parse the arguments

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

vs.set(3,1280)
vs.set(4,480)

time.sleep(2.0)
fps = FPS().start()

LEFT_PATH = "e:\\Projects\\Python\\stereovision\\images\\left\\{:06d}.jpg"
RIGHT_PATH = "e:\\Projects\\Python\\stereovision\\images\\right\\{:06d}.jpg"

# Filenames are just an increasing number
frameId = 0

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    #frame = vs.read()
    ret, frame = vs.read()
    #frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    cv2.imshow("Frame2", frame)

    imgL = frame[0:480, 0:640]
    imgR = frame[0:480:, 640:1280]

    cv2.imwrite(LEFT_PATH.format(frameId), imgL)
    cv2.imwrite(RIGHT_PATH.format(frameId), imgR)
    frameId += 1

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
cv2.destroyAllWindows()
vs.stop()
