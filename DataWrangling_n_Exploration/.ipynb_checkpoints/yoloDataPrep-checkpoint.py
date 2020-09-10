"""
yoloPerfMeasure.py: This program is for:
    * Reading all the video files from a given folder of VIRAT Dataset
    * Read respective annotation files from the folder of VIRAT Dataset
    * Create dataframe of video files and respective annotation files
    * Return Training and Test Data frames with 75% and 25% split
"""
__author__ = 'Nilesh Jha'
__email__ = "nilesh2sonu@gmail.com"

import glob
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def pathLookup(clip, lst):
    '''
    This function looks up the clip name from a list of given path.
    It will be used to associate the video clip to its respective
    annotation files path. Assumption is that the annotation file
    name contains the video clip name and it is true for VIRAT dataset.
    :param clip: Expects video clip name
    :param lst: Expects list of paths
    :return: Path of the annotation file for a given clip
    '''
    clip = clip.split('.')[0]                       # remove .mp4 from the clip ID for lookup
    lupath = [path for path in lst if clip in path]
    if len(lupath) !=0:
        return lupath[0]
    else:
        return np.nan

def getDataFrame(videopath, annotpath, dispWrt='display'):
    '''
    This function reads video path, annotations path and
    gets data set information in dataframe which can be
    displayed or saved into the database
    :param videopath: Expects path for video files
    :param annotpath: Expects path for annotations files
    :param dispWrt: Expects either 'display' or 'writedb'
    :return: Displays or writes pandas dataframe based on
    the paramater 'dispWrt' or error message
    '''
    if os.path.exists(videopath) and os.path.exists(annotpath):
        videoFiles  = glob.glob(videopath + '/*.mp4')
        objectFiles = glob.glob(annotpath + '/*.viratdata.objects.txt')  # Contains annotation information about objects
        eventFiles  = glob.glob(annotpath + '/*.viratdata.events.txt')   # Contains annotation information about events
    else:
        print('Path not valid')
    # Getting absolute path of all the files
    videoFiles  = [os.path.abspath(path) for path in videoFiles]
    objectFiles = [os.path.abspath(path) for path in objectFiles]
    eventFiles  = [os.path.abspath(path) for path in eventFiles]

    # print(len(videoFiles), len(objectFiles), len(eventFiles))
    # 329 315 275
    # As evident, the numbers of files are not same.
    # Few video clips do not have corresponding objects/event file
    # We will not use those clips for training which doesn't contain
    # corresponding object files. This means we will use around
    # 315 clips out of 329 clips for training
    # As far as missing event files are concerned, there may be a
    # valid reason for it. The clips may not have any events
    # We will create a dataframe containing the columns:
    # [video_name, video_path, object_path, event_path]

    dataset = pd.DataFrame(columns=['video_name', 'video_path', 'object_path', 'event_path'])
    dataset['video_name']   = [path.split('\\')[-1] for path in videoFiles]
    dataset['video_path']   = dataset.video_name.apply(pathLookup, lst=videoFiles)
    dataset['object_path']  = dataset.video_name.apply(pathLookup, lst=objectFiles)
    dataset['event_path']   = dataset.video_name.apply(pathLookup, lst=eventFiles)

    # Removing files which do not have objects annotation files
    dataset = dataset[dataset.object_path.notnull()]

    # Split dataset for training and testing (70%-30%)
    traindf, testdf = train_test_split(dataset, test_size=0.3, random_state=42)

    if dispWrt == 'display':
        print('Traning Dataset shape: {}'.format(traindf.shape))
        print(traindf.head())
        print('Traning Dataset shape: {}'.format(testdf.shape))
        print(testdf.head())
    elif dispWrt == 'writedb':
        
    else:
        print('Incorrect parameter value: {}'.format(dispWrt))






if __name__ == '__main__':
    vidFolder = 'C:/DS_ML/Video_Analytics_YOLOv4/VIRAT_Dataset/videos'
    annotFolder = 'C:/DS_ML/Video_Analytics_YOLOv4/VIRAT_Dataset/annotations'
    getDataFrame(vidFolder, annotFolder)
