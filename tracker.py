from collections import deque
import numpy as np
import argparse
#import imutils
import pygame
import cv2

pywindow = pygame.display.set_mode((600, 600), 0, 32)
prevx = 0
prevy = 0
# CLI arguments to export a lecture as video
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video ")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max video size")
args = vars(ap.parse_args())

# Define upper and lower bound for the color blue led HSV color space, then initialize the list of tracked points

blueLower = (110, 150, 150)
blueUpper = (130, 255, 255)
pts = deque(maxlen=args["buffer"])
l = []

# Start video capture if video is present, else go for the camera

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    # Gets each frame infinitely
    (grabbed, frame) = camera.read()
    for i in pygame.event.get():
        if i.type == pygame.KEYDOWN:
            if i.key == pygame.K_a:

                pywindow.fill((0, 0, 0))

    # Videos don't need this frame setup and can be left alone
    if args.get("video") and not grabbed:
        break

    # Resize the frame, blur it, and convert it to the HSV
    #frame = imutils.resize(frame, width=600)
    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Construct a mask for blue and perform dilation and erosion to remove blurs and edge effects from the mask

    mask = cv2.inRange(hsv, blueLower, blueUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Get contours and initalize the circle (led light as a circular point in space)
    # (x, y) center of the point
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # If contour found
    if len(cnts) > 0:
        # Find the largest contour in the mask then calculate the smallest enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Minimum enclosing radius for the point
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Update collated points
    pts.appendleft(center)

    # Check untracked points. If any are none, ignore them. Else compute height of the line and draw connecting lines
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        # int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        height = int(np.sqrt(args["buffer"] / float(10 + 1)) * 2.5)
        # l.append((pts[i-1],pts[i]))

        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), height)
        print(pts[i - 1][0], pts[i][0], pts[i - 1][1], pts[i][1],
              ((pts[i - 1][0] - pts[i][0])**2 +
               (pts[i - 1][1] - pts[i][1])**2)**(0.5))
        if ((pts[i - 1][0] - pts[i][0])**2 +
            (pts[i - 1][1] - pts[i][1])**2)**(0.5) > 10:
            pygame.draw.line(pywindow, (255, 0, 0),
                             (600 - pts[i - 1][0], pts[i - 1][1]),
                             (600 - pts[i][0], pts[i][1]), 5)

    # Show frame overlayed
    fframe = cv2.flip(frame, 1)
    cv2.imshow("Frame", fframe)
    key = cv2.waitKey(1) & 0xFF

    # Quit in Q
    if key == ord("q"):
        break

    pygame.display.update()  # refresh pygame screen

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
