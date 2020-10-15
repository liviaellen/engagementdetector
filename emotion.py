import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode

faceexist=0
faceexistpercent=0
posinterest=0
neginterest=0
suminterest=0
absinterest=0
netinterest=0
pospercent=0
negpercent=0
netpercent=0

abspercent=0
alcond=0
facetrue=0
facefalse=0
angrycount=0
disgustcount=0
scaredcount=0
happycount=0
neutralcount=0
sadcount=0
surprisedcount=0
angrypercent=0
disgustpercent=0
scaredpercent=0
happypercent=0
neutralpercent=0
sadpercent=0
surprisedpercent=0
facecond= ""
temp=1
maxrespon=0
respon=""
impression=""
faceval=0
presence2=""


# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'
num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')




# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Negatif - Marah", 1: "Negatif - Jijik", 2: "Negatif - Takut", 3: "Positif - Senang", 4: "Netral - Netral", 5: "Negatif - Sedih", 6: "Positif - Terkejut"}

# start the webcam feed
cap = cv2.VideoCapture('input.mov')
# while True:
    # Find haar cascade to draw bounding box around face

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Unable to read video")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Define the codec and create VideoWriter object.The output is stored in 'outemotion.mp4' file.
# Define the fps to be equal to 10. Also frame size is passed.


out = cv2.VideoWriter('output_emotion.mp4',cv2.VideoWriter_fourcc("m", "p", "4", "v"), 20, (frame_width,frame_height))

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    face_found = False
    for (x, y, w, h) in faces:
        if w > 0 :                 #--- Set the flag True if w>0 (i.e, if face is
            face_found = True
        #   facetrue+=1
        #else:
        #   face_found = False
        #   facefalse=+1
            
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
            #label
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if maxindex ==0:
            angrycount+=1
            neginterest+=1
        elif maxindex ==1:
            disgustcount+=1
            neginterest+=1
        elif maxindex ==2:
            scaredcount+=1
            neginterest+=1
        elif maxindex ==3:
            happycount+=1
            posinterest+=1
        elif maxindex ==4:
            neutralcount+=1
            netinterest+=1
        elif maxindex == 5:
            sadcount+=1
            neginterest+=1
        elif maxindex == 6:
            surprisedcount+=1
            posinterest+=1
        else:
            absinterest+=1
       
        netpercent
        suminterest=angrycount+disgustcount+scaredcount+happycount+neutralcount+sadcount+surprisedcount
        faceexist=posinterest+neginterest
        
        #if suminterest != 0:
        #faceexistpercent=round(((faceexist*100)/suminterest),2)
        pospercent=round(((posinterest * 100) / suminterest),2) if suminterest != 0 else 0
        negpercent=round((neginterest*100/suminterest),2) if suminterest != 0 else 0
        netpercent=round((netinterest*100/suminterest),2) if suminterest != 0 else 0
        neutralpercent=round((neutralcount*100/suminterest),2) if suminterest != 0 else 0
        disgustpercent=round((disgustcount*100/suminterest),2) if suminterest != 0 else 0
        angrypercent=round((angrycount*100/suminterest),2) if suminterest != 0 else 0
        happypercent=round((happycount*100/suminterest),2) if suminterest != 0 else 0
        sadpercent=round((sadcount*100/suminterest),2) if suminterest != 0 else 0
        surprisedpercent=round((surprisedcount*100/suminterest),2) if suminterest != 0 else 0
        scaredpercent=round((scaredcount*100/suminterest),2) if suminterest != 0 else 0
        
        
        
    if face_found is True :
        facecond="Ya"
        faceval+=1
    else:
        facecond="Tidak"
        
    maxrespon=max(netpercent,pospercent,negpercent)
    if netpercent==maxrespon and netpercent!=0:
        respon="Netral"
    elif pospercent==maxrespon and pospercent!=0:
        respon="Positif"
    elif negpercent==maxrespon and negpercent!=0:
        respon="Negatif"
    else:
        respon="No Attendance (null)"

       
    if netpercent+pospercent < negpercent and (netpercent+pospercent!=0):
        impression="Bad"
    elif netpercent+pospercent >= negpercent and negpercent!=0:
        impression="Good"
    else:
        impression="No Attendance (null)"
        
    #cv2.putText(frame, "Persentasi Wajah Terdeteksi:  " + str(faceexistpercent) + " %", (50, 530), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,0, 0), 2)
    #cv2.putText(frame, "Persentasi Wajah Tidak Terdeteksi:  " + str(abspercent) + " %", (50, 560), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255, 0), 2)
    cv2.putText(frame,"  Presensi: " + str(facecond),(20,410), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    cv2.putText(frame,"  Impresi: " + str(impression),(20,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    cv2.putText(frame,"  Overall Respon Emosi : " + str(respon),(20,490), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    cv2.putText(frame, "Persentasi Reaksi Netral  : " + str(netpercent) + " %", (50, 530), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255, 0), 2)
    cv2.putText(frame, "Persentasi Reaksi Positif  : " + str(pospercent) + " %", (50, 560), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255, 0), 2)
    cv2.putText(frame, "Persentasi Reaksi Negatif : " + str(negpercent) + " %", (50, 590), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255, 0), 2)
    cv2.putText(frame, "Jumlah Reaksi Netral     : " + str(netinterest), (50, 620), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255, 0), 2)
    cv2.putText(frame, "Jumlah Reaksi Positif     : " + str(posinterest), (50, 650), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255, 0), 2)
    cv2.putText(frame, "Jumlah Reaksi Negatif    : " + str(neginterest), (50, 680), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255, 0), 2)
    
    out.write(frame)    #label
    #cv2.imshow("Participant Detection and Emotion Recognition", frame)
    #cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    
    if faceval>=1:
        presence2="Ya"
    else:
        presence2="Tidak"
    
    f = open("resultEmotion.txt", "w")   # 'r' for reading and 'w' for writing
    f.write("Participant Detection and Emotion Recognition \n")
    f.write("Livia Ellen-1606887560 \n")
    f.write("Teknik Komputer, Universitas Indonesia \n\n")
    f.write("  Presensi                  : " + str(presence2) + "\n")
    f.write("  Impresi                   : " + str(impression)+"\n")
    f.write("  Overall Respon Emosi      : " + str(respon)+"\n\n")
    
    f.write("  Persentasi Respon Netral  : " + str(netpercent) + " % \n")
    f.write("  Persentasi Respon Positif : " + str(pospercent) + " % \n")
    f.write("  Persentasi Respon Negatif : " + str(negpercent) + " % \n\n")
    
    f.write("  Jumlah Reaksi Netral      : " + str(netinterest)+"\n")
    f.write("  Jumlah Reaksi Positif     : " + str(posinterest)+"\n")
    f.write("  Jumlah Reaksi Negatif     : " + str(neginterest)+"\n\n\n")
    
    f.write("  Netral:   " + str(neutralpercent)+" % \n")
    f.write("  Senang:   " + str(happypercent)+" % \n")
    f.write("  Sedih:    " + str(sadpercent)+" % \n")
    f.write("  Marah:    " + str(angrypercent)+" % \n")
    f.write("  Takut:    " + str(scaredpercent)+" % \n")
    f.write("  Terkejut: " + str(surprisedpercent)+" % \n")
    f.write("  Jijik:    " + str(disgustpercent)+" % \n\n")
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
