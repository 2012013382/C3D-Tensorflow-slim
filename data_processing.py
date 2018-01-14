from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import PIL.Image as Image
import random
import numpy as np
import os
import time
CLIP_LENGTH = 16
import cv2
VALIDATION_PRO = 0.2

np_mean = np.load('crop_mean.npy').reshape([CLIP_LENGTH, 112, 112, 3])

def get_video_indices(filename):
    lines = open(filename, 'r')
    #Shuffle data
    lines = list(lines)
    video_indices = range(len(lines))
    random.seed(time.time())
    random.shuffle(video_indices)
    validation_video_indices = video_indices[:int(len(video_indices) * 0.2)]
    train_video_indices = video_indices[int(len(video_indices) * 0.2):]
    return train_video_indices, validation_video_indices

def frame_process(clip, clip_length=CLIP_LENGTH, crop_size=112, channel_num=3):
    frames_num = len(clip)
    croped_frames = np.zeros([frames_num, crop_size, crop_size, channel_num]).astype(np.float32)


    #Crop every frame into shape[crop_size, crop_size, channel_num]
    for i in range(frames_num):
        img = Image.fromarray(clip[i].astype(np.uint8))
        if img.width > img.height:
            scale = float(crop_size) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
            scale = float(crop_size) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x: crop_x + crop_size, crop_y : crop_y + crop_size, :]
        croped_frames[i, :, :, :] = img - np_mean[i]

    return croped_frames


def convert_images_to_clip(filename, clip_length=CLIP_LENGTH, crop_size=112, channel_num=3):
    clip = []
    for parent, dirnames, filenames in os.walk(filename):
        filenames = sorted(filenames)
        if len(filenames) < clip_length:
            for i in range(0, len(filenames)):
                image_name = str(filename) + '/' + str(filenames[i])
                img = Image.open(image_name)
                img_data = np.array(img)
                clip.append(img_data)
            for i in range(clip_length - len(filenames)):
                image_name = str(filename) + '/' + str(filenames[len(filenames) - 1])
                img = Image.open(image_name)
                img_data = np.array(img)
                clip.append(img_data)
        else:
            s_index = random.randint(0, len(filenames) - clip_length)
            for i in range(s_index, s_index + clip_length):
                image_name = str(filename) + '/' + str(filenames[i])
                img = Image.open(image_name)
                img_data = np.array(img)
                clip.append(img_data)
    if len(clip) == 0:
       print(filename)
    clip = frame_process(clip, clip_length, crop_size, channel_num)
    return clip#shape[clip_length, crop_size, crop_size, channel_num]

def get_batches(filename, num_classes, batch_index, video_indices, batch_size=10, crop_size=112, channel_num=3):
    lines = open(filename, 'r')
    clips = []
    labels = []
    lines = list(lines)
    for i in video_indices[batch_index: batch_index + batch_size]:
        line = lines[i].strip('\n').split()
        dirname = line[0]
        label = line[1]
        i_clip = convert_images_to_clip(dirname, CLIP_LENGTH, crop_size, channel_num)
        clips.append(i_clip)
        labels.append(int(label))
    clips = np.array(clips).astype(np.float32)
    labels = np.array(labels).astype(np.int64)
    oh_labels = np.zeros([len(labels), num_classes]).astype(np.int64)
    for i in range(len(labels)):
        oh_labels[i, labels[i]] = 1
    batch_index = batch_index + batch_size
    #Convert to numpy
    batch_data = {'clips': clips, 'labels': oh_labels}
    return batch_data, batch_index
