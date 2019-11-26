"""
Loads and displays a video.
"""

# Importing OpenCV
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
 
ap.add_argument("-LMiX", "--left_min_x", type=int, default=60,
                help="For finding left image from stereo input, left minimum x coordinate")
ap.add_argument("-LMaX", "--left_max_x", type=int, default=1260,
                help="For finding left image from stereo input, left maximum x coordinate")
ap.add_argument("-LMiY", "--left_min_y", type=int, default=160,
                help="For finding left image from stereo input, left minimum y coordinate")
ap.add_argument("-LMaY", "--left_max_y", type=int, default=1110,
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

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('data/raw_data/both_eye_0304_1.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
fgbg = cv2.createBackgroundSubtractorMOG2(history=10,varThreshold=2,detectShadows=False)

# Read the video
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    imgL = frame[args.get("left_min_y"):args.get("left_max_y"),
                 args.get("left_min_x"):args.get("left_max_x")]
    # Converting the image to grayscale.
    gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

    # Extract the foreground
    edges_foreground = cv2.bilateralFilter(gray, 9, 75, 75)
    foreground = fgbg.apply(edges_foreground)
    
    # Smooth out to get the moving area
    kernel = np.ones((50,50),np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    # Applying static edge extraction
    edges_foreground = cv2.bilateralFilter(gray, 9, 75, 75)
    edges_filtered = cv2.Canny(edges_foreground, 60, 120)
    
    # Crop off the edges out of the moving area
    cropped = (foreground // 255) * edges_filtered

    # Stacking the images to print them together for comparison
    images = np.hstack((gray, edges_filtered, edges_filtered))

    # Display the resulting frame
    #cv2.imshow('Frame', images)
    # Display the resulting frame
    cv2.imshow('Frame', cropped)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

