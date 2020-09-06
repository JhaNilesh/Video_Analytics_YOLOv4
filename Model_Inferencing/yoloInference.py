'''
TODO:   Verify Weight file and Video file path after parsing arguments
'''

import os
import subprocess
import cv2
import pandas as pd
import sqlite3


darknetPath =   'darknet.exe'
cocoData =      'cfg\coco.data'
cfgFile =       'cfg\yolov4.cfg'
weightPath =    'C:\DS_ML\Video_Analytics_YOLOv4\Database_n_Files\yolov4.weights'
vidPath =       'C:\DS_ML\Video_Analytics_YOLOv4\Database_n_Files\Test_Video.mp4'



def runVid(vidPath):
    '''
    This function run YOLOv4 inference on video
    :param vidPath: Path of video file
    :return: Error Message if any
    '''
    os.chdir('YOLOv4\darknet')
    yoloCmd = [darknetPath, 'detector', 'demo', cocoData, cfgFile, weightPath, vidPath, '-ext_output']
    vidrun = subprocess.run(yoloCmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    return(vidrun.stdout.decode())

def detVidBB(vidPath):
    '''
    This function runs YOLOv4 detections on individual frame of the video,
     parses detection data in tabular format and saves the detections in a
     separate table in the database
    :param vidPath: Path of video file
    :return: Success/failure message
    '''
    os.chdir('YOLOv4\darknet')
    objects = ('dog', 'truck', 'bicycle', "pottedplant") # Object detections to be captured
    column_names = ['object', 'confScore', 'left_x', 'top_y', 'width', 'height']
    detectionDF = pd.DataFrame(columns = column_names)

    vidcap = cv2.VideoCapture(vidPath)
    frame_count = 0

    while(vidcap.isOpened()):

        response, frame = vidcap.read()

        if response == True:

            yoloCmd = [darknetPath, 'detector', 'test', cocoData, cfgFile, weightPath, '-ext_output', '-dont_show', frame]
            detfrm = subprocess.run(yoloCmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            detection = [line.decode() for line in detfrm.stdout.splitlines() if line.decode().startswith(objects)]
            df = pd.DataFrame([line.split() for line in detection])
            df.drop([2, 4, 6, 8], axis=1, inplace=True) # Dropping un-necessary columns
            df.columns = column_names                                                   # Naming the columns
            df['object'] = df['object'].str.strip(':')                                  # Removing un-necessary characters
            df['height'] = df['height'].str.strip(')')                                  # Removing un-necessary characters
            df['confScore'] = df['confScore'].str.strip('%').astype(float) / 100        # Converting confidence score as percetage
            df['frameNo'] = frame_count
            
            detectionDF.append(df, ignore_index=True)

        else:
            break

        frame_count += 1

    vidcap.release()
    tabName = vidPath.split('\\')[-1].split('.')[0]
    os.chdir('Database_n_Files')
    connection = sqlite3.connect('videoAnalytics.db')                           # Open connection to database
    detectionDF.to_sql(tabName, connection, if_exists='replace', index=False)   # Write detections in the table




# print(runVid(vidPath))
print(detVidBB(vidPath))