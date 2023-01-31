import cv2
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('video',help='path to input video file')
args = parser.parse_args()
cap = cv2.VideoCapture(args.video)
allValues = np.empty((0,3))
angVelList = []
angleList = []
numFrames = 0
if args.video == ".\evaluation.mp4" or args.video == ".\slowmo.mp4":
    fps = 240
else:
    fps = 30
step = False
while cap.isOpened():
    numFrames+=1
    ret, frame = cap.read()
    if not ret:
        break
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurImage = cv2.GaussianBlur(grayImage, (11,11), 2)
    circles = cv2.HoughCircles(blurImage, cv2.HOUGH_GRADIENT, 2, 100)
    circleMask = np.zeros_like(frame)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for myCircle in circles[0, :]:
            cv2.circle(frame, (myCircle[0],myCircle[1]), 1, (255,255,255), 3)
            cv2.circle(frame, (myCircle[0],myCircle[1]), myCircle[2], (0,255,0), 3)
            circleMask = cv2.circle(circleMask, (myCircle[0],myCircle[1]), myCircle[2], (255,255,255), -1)
        result = cv2.bitwise_and(frame, circleMask)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    markerMask = cv2.inRange(hsv, np.array([110,50,50]), np.array([130,255,255]))
    cv2.bitwise_and(frame, frame, mask=markerMask)
    conts = cv2.findContours(markerMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in conts[0]:
        area = cv2.contourArea(c)
        if area > 100:
            m = cv2.moments(c)
            if m["m00"] != 0:
                x = int(m["m10"] / m["m00"])
                y = int(m["m01"] / m["m00"])
            else:
                continue
            cv2.drawContours(result,[c], -1, (36,255,12), 2)
            cv2.circle(result, (x, y), 5, (255, 255, 255), -1)
            cv2.putText(result, "centroid", (x - 25, y - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if numFrames == 1:
                prevX = x
                prevY = y
                vectorPrev = (x-myCircle[0], y-myCircle[1])
            else:
                vectorCurr = (x-myCircle[0], y-myCircle[1])
                dot = (vectorCurr[0]*vectorPrev[0]) + (vectorCurr[1]*vectorPrev[1])
                magCurr = ((vectorCurr[0])**2 + (vectorCurr[1])**2)**0.5
                magPrev = ((vectorPrev[0])**2 + (vectorPrev[1])**2)**0.5
                val = dot / (magCurr*magPrev)
                try:
                    rad = math.acos(val)
                    angle = math.degrees(rad)
                    vectorPrev = vectorCurr
                    print("From frame", numFrames-1, "to", numFrames)
                    print(angle)
                    angleList.append(angle)
                    angularVelocity = rad / (1/fps)
                    angVelList.append(angularVelocity)
                except ValueError as E :
                    print("skipped", E)
    cv2.imshow("result", result)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        step = True
    if step:
        cv2.waitKey()
print("----------------------------------------")
print("Total number of frames is: ", numFrames)
print("FOR EVALUATION PURPOSES (Reccomended video of 1 full rotation): ")
print("Sum of change in angles: ", sum(angleList))
print("Average Angular Velocity: ", sum(angVelList) / len(angVelList))
avgs = []
historyLen = 5
lastVels = []
for i, angVel in enumerate(angVelList):
    lastVels.append(angVel)
    if i < historyLen - 1:
        avgs.append(-1)
    else:
        avgs.append(sum(lastVels) / historyLen)
        lastVels.pop(0)
plt.plot(angVelList)
plt.plot(avgs)
plt.title("Angular Velocity of Ball")
plt.ylabel("Rad/s")
plt.xlabel("Frame #")
plt.show()