"""yoloInference.py: Program for inferencing YOLOv4 on videos and/or saving detections in database"""
__author__ = 'Nilesh Jha'
__email__ = "nilesh2sonu@gmail.com"

import os
import subprocess
import pandas as pd
import re
import sqlite3

# As 'darknet detector test' expects a image path, for videos, every single frame mist be saved as image, its path
# must be read and then the detector will generate bounding box and coordinates. This process is takes a long time,
# as for a video of 1 minute of 60FPS, number of frames are 3600 which is a large number of file i/o. This is extremely
# inefficient way to generate inference. Even generating inference on SSD drive for a 90 seconds clip took more than
# 45 minutes and hence it completely overcomes the very virtue of YOLO, that is speed. We will inference video using
# 'darknet detector demo'

# rootDir     =   os.path.dirname(os.path.abspath(__file__))   # use it for running on MAC/Linux
rootDir     =   'C:\\DS_ML\\Video_Analytics_YOLOv4'
darknetDir  =   os.path.join(rootDir, 'YOLOv4', 'darknet')
cocoData    =   'cfg\coco.data'
cfgFile     =   'cfg\yolov4.cfg'
weightPath  =   os.path.join(rootDir, 'Database_n_Files', 'yolov4.weights')
detPath     =   os.path.join(rootDir, 'Database_n_Files')

def detVidBB(vidPath, dbwrt='table', suffix='pretrain'):
    '''
    This function run YOLOv4 inference on video and saves detections data
    (i.e bounding box and confidence scores) in the database
    :param vidPath: Path of video file
    :param dbwrt: Return annotations as dataframe or write in table. Possible values 'table' and 'retdf'
    :param suffix:  Table name suffix. Possible values 'pretrain' and 'posttrain'
    :return: Error Message if any
    '''
    os.chdir(darknetDir)
    yoloCmd = ['darknet.exe', 'detector', 'demo', cocoData, cfgFile, weightPath, vidPath, '-ext_output']
    vidrun = subprocess.run(yoloCmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    detections = [line.decode().strip()
                  for line in vidrun.stdout.splitlines()
                  if ((line.decode().strip() != '') and
                      (('Objects' in line.decode().strip()) or ('person' in line.decode().strip())))]
    os.chdir(rootDir)

    # Parsing detections, adding frame number based on 'Object' count, to the detection entries and generating DataFrame
    frameCnt = 0
    finalArr = []
    for line in detections:
        if line.startswith('Objects'):
            frameCnt += 1
        else:
            confScore   =   re.search('person:[ -]+(\d+)', line, re.IGNORECASE).group(1)
            left_x      =   re.search('left_x:[ -]+(\d+)', line, re.IGNORECASE).group(1)
            top_y       =   re.search('top_y:[ -]+(\d+)', line, re.IGNORECASE).group(1)
            width       =   re.search('width:[ -]+(\d+)', line, re.IGNORECASE).group(1)
            height      =   re.search('height:[ -]+(\d+)', line, re.IGNORECASE).group(1)
            finalArr.append([frameCnt, 'person', confScore, left_x, top_y, width, height])

    df = pd.DataFrame(finalArr, columns=['frameNo', 'object', 'confScore', 'left_x', 'top_y', 'width', 'height'])

    if dbwrt == 'table':
        tabName = vidPath.split('\\')[-1].split('.')[0] + '_'+ suffix
        os.chdir('Database_n_Files')
        connection = sqlite3.connect('videoAnalytics.db')                           # Open connection to database
        df.to_sql(tabName, connection, if_exists='replace', index=False)            # Write detections in the table
        os.chdir(rootDir)
    elif dbwrt == 'retdf':
        return df
    else:
        print("Annotations not saved")


def saveInf(vidPath):
    '''
    This function takes video as input and generates YOLOv4 annotated video file
    :param vidPath: Video file path expected
    :return: Error message if any
    '''
    os.chdir(darknetDir)
    infVidName = vidPath.split('\\')[-1].split('.')[0] + '_inference.mp4'
    infVidPath = os.path.join(rootDir, 'Database_n_Files', infVidName)
    yoloCmd = ['darknet.exe', 'detector', 'demo', cocoData, cfgFile, weightPath, vidPath, '-out_filename', infVidPath]
    vidrun = subprocess.run(yoloCmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    os.chdir(rootDir)



if __name__ == '__main__':

    vidPath = os.path.join(rootDir, 'Database_n_Files', 'Test_Video.mp4')
    annotdf = pd.DataFrame()
    annotdf = detVidBB(vidPath, dbwrt='retdf')
    print(annotdf.head())
    # saveInf(vidPath)


'''
TODO:   Verify Weight file and Video file path after parsing arguments
'''
