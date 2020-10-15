
SISTEM PENDETEKSI ENGAGEMENT SISWA DALAM LINGKUNGAN E-LEARNING DENGAN TEKNOLOGI OPENCV BERBASIS CNN
Livia Ellen 
1606887560
livia.ellen@ui.ac.id



1. Sebelum menjalankan program, harap install prequisite berupa homebrew, pip, python3 dan mkvirtualenv, GNU Parallel
2. Masuk ke virtual environment python
3. Install lybrary yang dibutuhhkan dengan command
pip3 install -r requirements.txt
4. Setelah semua library di-install, maka program siap dijalankan

Langkah-langkah menjalankan program

1. Pastikan sudah ada file input.mov pada root directory sebagai input dari program
2. Jalankan program eyegaze.py dan emotion.py secara bersamaan menggunakan command
parallel ::: "python eyegaze.py" "python emotion.py"
3. Setelah progream dijalankan, program akan memberikan output berupa file teks resultEyegaze.txt dan resultEmotion.txt berisi nilai hasil analisa python script serta file video output_eyegaze.mp4 dan output_emotion.mp4 berisi video yang telah dianotasi oleh sistem pendeteksi engagement siswa.


Notes: Jika anda menjalankan program untuk kedua kalinya, pastikan anda telah memindahkan file text dan video hasil output sebelumnya, jika tidak, program akan melakukan rewrite pada data tersebut.
