######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Tobie Abel
# Date: 20/8/2020
# Description:
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# it was further added to by edje electronics https://www.youtube.com/watch?v=aimSGOAUI8Y&t=2s adding
# methods of drawing boxes and labels using OpenCV.
#
# I have added all the code for moving the servo's of the goal keeper, predicting positions when the ball is lost
# and sending emails with attached photos when a goal is scored

# Import packages
import os
import argparse
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
from threading import Thread
import importlib.util
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Email variables
SMTP_SERVER = 'smtp.gmail.com'  # Email Server
SMTP_PORT = 587  # server port for gmail when using TLS security
GMAIL_USERNAME = "emailaddress"
GMAIL_PASSWORD = "password"
sendTo = "tobie.abel@vrpconsulting.com"
emailSubject = "VAR"
emailContent = "Decision is...Goal Allowed!"
lastTime = 0


class Emailer:
    def sendmail(self, recipient, subject, content):
        global lastTime
        # check if email has been sent in last min
        thisTime = time.time()
        if thisTime - lastTime > 60:
            print("goal!")

            # create headers
            emailData = MIMEMultipart()
            emailData['subject'] = subject
            emailData['To'] = recipient
            emailData['From'] = GMAIL_USERNAME

            # store latest frame as jpg.  frame 1 is the name of the variable used for videostream.read()
            cv2.imwrite("/home/pi/tflite1/Goal/frame1.jpg", frame1)

            # Attach text
            emailData.attach(MIMEText(content))

            # Attach image
            with open("/home/pi/tflite1/Goal/frame1.jpg",
                      "rb") as attachment:  # this is file location where frame was saved
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="frame1.jpg"')
            emailData.attach(part)

            # connect to gmail server
            session = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            session.ehlo()
            session.starttls()
            session.ehlo()

            # login to gmail
            session.login(GMAIL_USERNAME, GMAIL_PASSWORD)

            # send email & exit
            session.sendmail(GMAIL_USERNAME, recipient, emailData.as_string())
            session.quit
            lastTime = thisTime


sender = Emailer()

# set GPIO numbering mode
GPIO.setmode(GPIO.BOARD)

# set pin 11 and 15 as an output, and set servo1 on pin 11 and servo2 von pin 12 as PWM
GPIO.setup(11, GPIO.OUT)
servo1 = GPIO.PWM(11, 50)  # note 11 is pin, 50 = 50Hz pulse
GPIO.setup(15, GPIO.OUT)
servo2 = GPIO.PWM(15, 50)  # note 15 is pin, 5 = 50Hz pulse
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # pin connected to reset button

#Start PWM running, but with a value of 5 (90 degree)
servo1.start(5)
servo2.start(5)
time.sleep(0.25)

#Define variable duty
duty1 = 0
duty2 = 0

# Move servo
servo1.ChangeDutyCycle(duty1)
servo2.ChangeDutyCycle(duty2)

#Define duty dictionary, position dictionary and initiate position variables
dutys = {"left":[2,4],"leftCentre": [3,4.5], "centre": [5,5], "rightCentre": [7,5.5], "right": [8,6]}
positions = {"left": 128, "leftCentre": 256, "centre": 384, "rightCentre": 512, "right": 641}
currentPosition = "centre"
newPosition = "centre"
#initating last x and y variables as program needs a sarting point for the ball
lastx = 0
lasty = 0
prevx = None
prevy = None
ticker = 0

#function to move the ball
def moveball (xcentre, ycentre):
            #Need to declare these are global variables for some reason
            global currentPosition
            global newPosition
     #Loop through the positions dictionary. and find the category which the ball position is in
            if ycentre > 270: #if the ball is close to the bottom of the screen, its near the goal so jitter around
                if xcentre < prevx: #if the ball is moving left to right, move right first
                    servo2.ChangeDutyCycle(3)
                    time.sleep(0.05)
                    servo2.ChangeDutyCycle(7)
                    time.sleep(0.05)
                else: #else move left first
                    servo2.ChangeDutyCycle(7)
                    time.sleep(0.05)
                    servo2.ChangeDutyCycle(3)
                    time.sleep(0.05)
                #servo2.ChangeDutyCycle(3)
                #time.sleep(0.075)
                #servo2.ChangeDutyCycle(7)
                #time.sleep(0.075)
                servo2.ChangeDutyCycle(dutys[newPosition][1])
                time.sleep(0.05)
                
            else:
                for x in positions.items():
                    if not xcentre > x[1]: #if x is not greater than the upper limit of a category
                           newPosition = x[0] #assign newValue to that category
                           print("the new position = ", newPosition)
                           print("the current position = ", currentPosition)
                           if not newPosition == currentPosition:
                                #write some code to retrieve values from duty dictionary and move servos' assign new value to current position
                                duty1 = dutys[newPosition][0]
                                duty2 = dutys[newPosition][1]
                                print("The new duty 2 = ", duty2)
                                servo1.ChangeDutyCycle(duty1)
                                servo2.ChangeDutyCycle(duty2)
                                time.sleep(0.1)
                                duty1 = 0
                                duty2 = 0
                                servo1.ChangeDutyCycle(duty1)
                                servo2.ChangeDutyCycle(duty2)
                                currentPosition = newPosition
                                print("current position has been changed to ",currentPosition)
                           break
           
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=90):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=90).start()
time.sleep(1)

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    #Check if rest button has been pressed, if so set gaolie back to centre and stop whole program
    input_state= GPIO.input(22)
    if input_state == False:
        print("Button is pressed")
        duty1 = 5
        duty2 = 5
        # reset goalie to centre
        servo1.ChangeDutyCycle(duty1)
        servo2.ChangeDutyCycle(duty2)
        #wait
        time.sleep(0.1)
        break
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    maxscore = np.argmax(scores)# return the highest confidence score object only
    maxobject_name = labels[int(classes[maxscore])]
    #print(maxobject_name)
    
    # Loop over all detections and draw detection box if confidence is above minimum threshold i.e found the ball
    #for i in range(len(scores)):
    if ((scores[maxscore] > min_conf_threshold) and (scores[maxscore] <= 1.0)):

        # Get bounding box coordinates and draw box
        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
        ymin = int(max(1, (boxes[maxscore][0] * imH)))
        xmin = int(max(1, (boxes[maxscore][1] * imW)))
        ymax = int(min(imH, (boxes[maxscore][2] * imH)))
        xmax = int(min(imW, (boxes[maxscore][3] * imW)))
        ycentre = int((ymin + ymax) * 0.5)  # calculate centre of bounding box
        xcentre = int((xmin + xmax) * 0.5)  # calculate centre of bounding box
        print("The ball is at position", xcentre, ycentre)
        # save the last x and y to prev x and y
        prevx = lastx
        prevy = lasty
        # save current x and y to last x and y
        lastx = xcentre
        lasty = ycentre
        print("the last position of X and Y has been saved as ", lastx, lasty)
        print("the previous position of X and Y has been saved as ", prevx, prevy)
        ticker = 0
        print("ticker set back to ", ticker)
        # draw bounding box with circle in centre
        # cv2.circle(frame, (xcentre, ycentre), 50, (10,255,0), thickness=2,lineType=8, shift=0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
        # Calculate error
        # x_error = int (640 - xcentre) #640 is half the 1280 camera horizontal pixel resolution
        # print (" X Error = ", x_error)
        # Translate X into duty cycle
        # assign x to a category and only change servos if the category has changed since last frame
        moveball(xcentre, ycentre)

        # Draw label
        object_name = labels[int(classes[maxscore])]  # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[maxscore] * 100))  # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
        label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10),
                      (255, 255, 255), cv2.FILLED)  # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                    2)  # Draw label text

    # if no ball is found
    else:
        if prevx is not None and ticker < 3:  # if we've got at least 2 previous frames with the ball
            print("ticker = ", ticker)
            changeInx = prevx - lastx
            changeIny = prevy - lasty
            predx = lastx - changeInx
            predy = lasty - changeIny
            if changeIny < 0:
                predy = lasty - (changeIny *1.15) #add proprtionate gain if ball is moving towards camera as it will naturally cover more pixels 
            print("ball canot be found.  We predict it would be in position ", predx, predy)
            moveball(predx, predy)
            ticker = ticker + 1
            prevy = lasty
            lasty = predy
            # if this is the third time ticker and x/y coordinates mean ball should be in the goal, send email with attached photo
            if ticker == 3 and lasty > 460:
                sender.sendmail(sendTo, emailSubject, emailContent)

    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    duty1 = 0
    duty2 = 0
    # turn servo off
    servo1.ChangeDutyCycle(duty1)
    servo2.ChangeDutyCycle(duty2)
    #wait
    time.sleep(0.025)

# Clean up
cv2.destroyAllWindows()
videostream.stop()
servo1.stop()
servo2.stop()
GPIO.cleanup()
