# C3D-Tensorflow-slim
A simple Tensorflow code for C3D
## Reqirements
Tensorflow(>=1.4)
Python(>=2.7)
## Dataset
UCF101. You need to place it in the root directory of the workspace.
## Usage
```Bash
sudo ./convert_video_to_images.sh UCF101/ 5
```
to convert videos into images(5FPS per-second).
```Bash
./convert_images_to_list.sh UCF101/ 4
```
to obtain train and test sets.(3/4 train; 1/4 test)
```Bash
python train.py
```
for training.
```Bash
python test.py
```
for testing.
## Attention
Radom choose 16 frames as a clip to represent a video for training and testing.
## Result
0.71 on the test set after 40 epoch.
