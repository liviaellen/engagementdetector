"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""
from tensorflow.keras.preprocessing.image import img_to_array
import imutils
import cv2
from gaze_tracking import GazeTracking
import time,datetime
import numpy as np

gaze = GazeTracking()

ongaze=0
offgaze=0
sumtask=0
absgaze=0
focuspercent=0
distractedpercent=0
abspercent=0
onscreen=0
offscreen=0
onscreenpercent=0
offscreenpercent=0
maxpresence=0
att=""

cap = cv2.VideoCapture("input.mov")

if (cap.isOpened() == False):
    print("Unable to read video")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'output_gaze.mp4' file.
# Define the fps to be equal to 10. Also frame size is passed.


out = cv2.VideoWriter('output_eyegaze.mp4',cv2.VideoWriter_fourcc("m", "p", "4", "v"), 20, (frame_width,frame_height))

while True:
    # We get a new frame from the webcam
    ret, frame = cap.read()
    
    if ret == True:
        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)
        frame = gaze.annotated_frame()
        
        # use grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        text = ""
        #left_pupil = gaze.pupil_left_coords()
        #right_pupil = gaze.pupil_right_coords()
        vertical_c = gaze.vertical_ratio()
        horizontal_c = gaze.horizontal_ratio()
        
        if vertical_c is None or horizontal_c is None:
            text = "Mata Tidak Terdeteksi"
            absgaze+=1
        elif vertical_c<=0.37 or vertical_c>=0.80 or horizontal_c<=0.44 or horizontal_c>=0.74:
            text = "Mata Tidak Memperhatikan"
            offgaze+=1
            
        elif vertical_c>0.37 and vertical_c<0.80 and horizontal_c>0.44 and horizontal_c<0.74:
            text = "Mata Memperhatikan"
            ongaze+=1
        
        #if gaze.no_eye():
        #    text = "Mata Tidak Terdeteksi"
        #    absgaze+=1
        #elif gaze.is_on():
        #    text = "Mata Memperhatikan"
        #    ongaze+=1
        
        #elif gaze.is_off():
        #    text = "Mata Tidak Memperhatikan"
        #    offgaze+=1
        

        # detect face(s)
        sumtask=ongaze+offgaze+absgaze
        focuspercent=round((ongaze * 100 / sumtask),2) if sumtask != 0 else 0
        #distractedpercent=round((offgaze*100/sumtask),2)
        abspercent=round((absgaze*100/sumtask),2) if sumtask != 0 else 0
        onscreen=ongaze
        offscreen=offgaze+absgaze
        onscreenpercent=focuspercent
        offscreenpercent=round((offscreen*100/sumtask),2) if sumtask != 0 else 0
        
        maxpresence=max(onscreenpercent,offscreenpercent,abspercent)
        if onscreenpercent==maxpresence:
            att="On Screen"
        elif offscreenpercent==maxpresence and abspercent!=100:
            att="Off Screen"
        elif abspercent==100:
            att="No Attendance (null)"
        
        
        cv2.putText(frame,text,(50,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        cv2.putText(frame,"Overall Attendance : " + str(att),(50,490), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        cv2.putText(frame, "Persentasi On Screen               : " + str(onscreenpercent) + " %", (50, 530), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255, 0), 2)
        cv2.putText(frame, "Persentasi Off Screen              : " + str(offscreenpercent) + " %", (50, 560), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255, 0), 2)
        cv2.putText(frame, "Persentasi Mata Tidak Terdeteksi   : " + str(abspercent) + " %", (50, 590), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255, 0), 2)
        cv2.putText(frame, "Jumlah Mata Memperhatikan       : " + str(ongaze), (50, 620), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255, 0), 2)
        cv2.putText(frame, "Jumlah Mata Tidak Memperhatikan : " + str(offgaze), (50, 650), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255, 0), 2)
        cv2.putText(frame, "Jumlah Mata Tidak Terdeteksi      : " + str(absgaze), (50, 680), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255, 0), 2)

        
            
            
        
            #if gaze.is_blinking():
            #    text = "Blinking"
            #elif gaze.is_right():
            #    text = "Looking right"
            #elif gaze.is_left():
            #    text = "Looking left"
            #elif gaze.is_up():
            #    text = "Looking up"
            #elif gaze.is_down():
            #    text = "Looking down"
            #elif gaze.is_center():
            #    text = "Looking center"
            
        #cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 3,cv2.LINE_AA)
        
        
        
        
        #cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        #cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Iris Horizontal:  " + str(horizontal_c), (30, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Iris Vertical:  " + str(vertical_c), (30, 80), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        out.write(frame)
        #cv2.imshow("EyeGaze On Screen-Off Screen Detection", frame)
        
        
        
        f = open("resultEyegaze.txt", "w")   # 'r' for reading and 'w' for writing
        f.write("EyeGaze On Screen-Off Screen Detection \n")
        f.write("Livia Ellen-1606887560 \n")
        f.write("Teknik Komputer, Universitas Indonesia \n\n")
        
        f.write("  Overall Attendance               : " + str(att) + " \n\n")
        f.write("  Persentasi On Screen             : " + str(onscreenpercent) + " %\n")
        f.write("  Persentasi Off Screen            : " + str(offscreenpercent) + " % \n")
        f.write("  Persentasi Mata Tidak Terdeteksi : " + str(abspercent) + " %\n \n")
        f.write("  Jumlah Mata Memperhatikan        : " + str(ongaze)+ "\n")
        f.write("  Jumlah Mata Tidak Memperhatikan  : " + str(offgaze) + "\n")

                              # Close file
        
        
        # quit with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
