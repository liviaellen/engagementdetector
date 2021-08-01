**Student Engagement Detection System in E-Learning Environment using OpenCV and CNN**

The student engagement detection system works by detecting student eye gaze and facial expression using OpenCV technology, FER-2013 dataset and CNN (convolutional neural network) method, receiving input through video file input or real-time webcam feed. The system will report on
the student engagement level “engaged” if the student's eyes are staring at the screen
and student facial expression showing a neutral or positive impression.

Paper : https://bit.ly/thesis-paper

Project Link : https://github.com/liviaellen/engagementdetector 

Video Presentation : https://bit.ly/ellenskripsi 

The demo could be found here: 
The demo video is in Indonesian language.


**How to Install the engagement detector:**
1. Install prerequisites: homebrew, pip, python3, and mkvirtualenv, GNU Parallel
2. Create and access a python virtual environment
3. Install the prequisited python library by typing this command
``` pip3 install -r requirements.txt ```

**How to Run the Engagement Detector**

**Input : Existing Video**
1. If you want to process an existing video, run this command on the root directory
``` parallel ::: "python eyegaze.py" "python emotion.py" ```
The command will process the input.mov video on the root directory as the input, make sure you rename the video you want to process as input.mov. If you want to change the default input name, change it in the .py code.
2. After the process finished, it will give you 4 output, resultEyegaze.txt and resultEmotion.txt contaning the analyzed result and video files output_eyegaze.mp4 and output_emotion.mp4 containing the anotated video.

**Input : Real Time Webcam**
1. If you want to process a real time video, run this command on the ./cam directory
``` parallel ::: "python eyegaze_cam.py" "python emotion_cam.py" ```
The program will open your webcam and analyze your engagement.
2. After the process finished, it will give you 4 output, resultEyegaze.txt and resultEmotion.txt contaning the analyzed result and video files output_eyegaze.mp4 and output_emotion.mp4 containing the anotated video.


---
Bahasa - Indonesian Language

SISTEM PENDETEKSI ENGAGEMENT SISWA DALAM LINGKUNGAN E-LEARNING DENGAN TEKNOLOGI OPENCV BERBASIS CNN
Livia Ellen 



1. Sebelum menjalankan program, harap install prequisite berupa homebrew, pip, python3 dan mkvirtualenv, GNU Parallel
2. Masuk ke virtual environment python
3. Install library yang dibutuhhkan dengan command
pip3 install -r requirements.txt
4. Setelah semua library di-install, maka program siap dijalankan

Langkah-langkah menjalankan program

1. Pastikan sudah ada file input.mov pada root directory sebagai input dari program
2. Jalankan program eyegaze.py dan emotion.py secara bersamaan menggunakan command
parallel ::: "python eyegaze.py" "python emotion.py"
3. Setelah progream dijalankan, program akan memberikan output berupa file teks resultEyegaze.txt dan resultEmotion.txt berisi nilai hasil analisa python script serta file video output_eyegaze.mp4 dan output_emotion.mp4 berisi video yang telah dianotasi oleh sistem pendeteksi engagement siswa.


Notes: Jika anda menjalankan program untuk kedua kalinya, pastikan anda telah memindahkan file text dan video hasil output sebelumnya, jika tidak, program akan melakukan rewrite pada data tersebut.
