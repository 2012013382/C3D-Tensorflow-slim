# C3D-Tensorflow-slim
A simple Tensorflow code for C3D
## Reqirements
Tensorflow(>=1.4)
Python(>=2.7)
## Usage
```Bash
sudo ./convert_video_to_images.sh UCF101/ 5
```
to convert videos into images(5FPS per-second).
```Bash
./convert_images_to_list.sh UCF101/ 4
```
to obtain train and test sets.
```Bash
python train.py
```
for training.
```Bash
python test.py
```
for testing.
## Result
