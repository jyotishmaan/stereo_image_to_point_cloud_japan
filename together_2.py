import numpy as np
from sklearn.preprocessing import normalize
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
from util import calculate_disparity_michael, calculate_disparity_tim
import pandas as pd

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
                help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
                help="OpenCV object tracker type")
ap.add_argument("-LMiX", "--left_min_x", type=int, default=60,
                help="For finding left image from stereo input, left minimum x coordinate")
ap.add_argument("-LMaX", "--left_max_x", type=int, default=1260,
                help="For finding left image from stereo input, left maximum x coordinate")
ap.add_argument("-LMiY", "--left_min_y", type=int, default=160,
                help="For finding left image from stereo input, left minimum y coordinate")
ap.add_argument("-LMaY", "--left_max_y", type=int, default=910,
                help="For finding left image from stereo input, left maximum y coordinate")
ap.add_argument("-RMiX", "--right_min_x", type=int, default=1340,
                help="For finding left image from stereo input, right minimum x coordinate")
ap.add_argument("-RMaX", "--right_max_x", type=int, default=2540,
                help="For finding left image from stereo input, right maximum x coordinate")
ap.add_argument("-RMiY", "--right_min_y", type=int, default=160,
                help="For finding left image from stereo input, right minimum y coordinate")
ap.add_argument("-RMaY", "--right_max_y", type=int, default=910,
                help="For finding left image from stereo input, right maximum y coordinate")
args = vars(ap.parse_args())


# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.MultiTracker_create(args["tracker"].upper())
    print('done')

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    trackers = cv2.MultiTracker_create()

# initialize the bounding box coordinates of the object we are going
# to track
#R1(252, 187, 56, 59) R2 (311, 182, 38, 87) L1(125, 224, 76, 63) L2(77, 218, 47, 78)
#initBB = [(252, 192, 25, 25) ,(272, 195, 25, 25), (302, 195, 25, 25),(87, 238, 25, 25), (100, 238, 25, 25),(125, 240, 25, 25)]
initBB = [(245, 237, 60, 60), (225, 242, 60, 60),(195, 245, 60, 60),(16, 115, 60, 60) ,(42, 120, 60, 60), (80, 122, 60, 60)]

#initBB = []
df = pd.DataFrame(columns =['R1x', 'R1y','R2x','R2y','R3x','R3y','L1x', 'L1y','L2x','L2y', 'L3x','L3y' ]) 
df.to_csv(r'testing.csv')

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    #raise Exception("Live stream version not implemented yet")
    vs = VideoStream(src=0).start()
 
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# initialize the FPS throughput estimator
fps = None
fps = FPS().start()
frame = vs.read()
for box in initBB:
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        trackers.add(tracker,frame[1], box)
        fps = FPS().start()
    
# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        print("End of stream, bye bye")
        break
    imgL = frame[args.get("left_min_y"):args.get("left_max_y"),
                 args.get("left_min_x"):args.get("left_max_x")]
    imgR = frame[args.get("right_min_y"):args.get("right_max_y"),
                 args.get("right_min_x"):args.get("right_max_x")]


    # displayed image dimensions
    displayed_image = imutils.resize(imgL, width = 500)
    (H, W) = displayed_image.shape[:2]
    # check to see if we are currently tracking an object
    if len(initBB)!=0:
        # only compute disparity if we have a bounding box
        disparity = calculate_disparity_tim(imgL, imgR, show_disparity=True)
        # grab the new bounding box coordinates of the object
        (success, boxes) = trackers.update(displayed_image)
        print(boxes)
        print(success)
        # check to see if the tracking was a success
        coordinates = []

        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(displayed_image, (x, y), (x + w, y + h),
                        (0, 255, 0), 2)

            # update the FPS counter
            fps.update()
            fps.stop()

            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Status", "{} is {} at FPS {:.2f}".format(args["tracker"],
                                                        "Success" if success else "Failed",
                                                        fps.fps())),
                # depth = baseline * focal / disparity
                ("Tip Location", "x={} | y={} | z={}".format(x,
                                                            y+h,
                                                            np.mean(disparity.shape[0] * 0.02 / disparity[x:x+w, y:y+h]
                )))
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(displayed_image, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            coordinates = coordinates + [x + w/2]
            coordinates = coordinates + [y + h/2]
        print(coordinates)
        dataframe = pd.DataFrame([coordinates], columns =['R1x', 'R1y','R2x','R2y','R3x','R3y','L1x', 'L1y','L2x','L2y','L3x','L3y' ]) 
        print(dataframe)
        dataframe.to_csv(r'testing.csv',mode='a', header=False)

            
    # show the output frame
    cv2.imshow("Frame", displayed_image)
    key = cv2.waitKey(1) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", displayed_image, fromCenter=False,
                               showCrosshair=True)

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        initBB.append(box)
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        trackers.add(tracker, frame, box)
        fps = FPS().start()

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break
# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
