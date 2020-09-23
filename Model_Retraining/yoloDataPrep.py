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
import cv2
import shutil

vidFolder = 'C:/DS_ML/Video_Analytics_YOLOv4/VIRAT_Dataset/videos'                      # convert to argument with default value
annotFolder = 'C:/DS_ML/Video_Analytics_YOLOv4/VIRAT_Dataset/annotations'               # convert to argument with default value
dataFolder = 'C:/DS_ML/Video_Analytics_YOLOv4/YOLOv4/darknet/build/darknet/x64/data'    # convert to argument with default value

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

def loadMetaData(videopath, annotpath, dispWrt='display'):
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
        import sqlite3
        rootDir = 'C:\\DS_ML\\Video_Analytics_YOLOv4'
        dbDir = 'C:\\DS_ML\\Video_Analytics_YOLOv4\\Database_n_Files'
        os.chdir(dbDir)
        connection = sqlite3.connect('videoAnalytics.db')
        traindf.to_sql('trainvids', connection, if_exists='replace', index=False)
        testdf.to_sql('testvids', connection, if_exists='replace', index=False)
        os.chdir(rootDir)

    else:
        print('Incorrect parameter value: {}'.format(dispWrt))

def readMetaData():
    '''
    This function read trainvids and testvids table from database and returns dataframes
    :return: Returns Train and Test Video dataframes
    '''
    import sqlite3        
    rootDir = 'C:\\DS_ML\\Video_Analytics_YOLOv4'
    dbDir = 'C:\\DS_ML\\Video_Analytics_YOLOv4\\Database_n_Files'
    os.chdir(dbDir)
    connection = sqlite3.connect('videoAnalytics.db')
    cursor = connection.cursor()
    # Check if table trainvids exists
    cursor.execute('SELECT name FROM sqlite_master WHERE name = "trainvids"')
    trvid = tsvid = 0
    if cursor.fetchall()[0][0] == 'trainvids':
        traindf = pd.read_sql("SELECT * FROM trainvids", connection)
        print('Training Table Present and contains {} videos'.format(traindf.shape[0]))
        trvid = 1
        # print(traindf.head())
    else:
        print('Training table not present')

    cursor.execute('SELECT name FROM sqlite_master WHERE name = "testvids"')
    if cursor.fetchall()[0][0] == 'testvids':
        testdf = pd.read_sql("SELECT * FROM testvids", connection)
        print('Testing Table Present and contains {} videos'.format(testdf.shape[0]))
        tsvid = 1
        # print(testdf.head())
    else:
        print('Testing table not present')

    if (trvid == 1) and (tsvid == 1):
        return traindf, testdf
    os.chdir(rootDir)

def loadAnnotObj(df, tabname):
    '''
    This function reads the path of annotation file of all the videos in dataframe
    and loads it into given table name
    :param df: Train/Test Dataframe with valid path is expected
    :param tabname: table name is expected
    :return: success/failure messages
    '''
    columns = ['obj_id', 'obj_dur', 'frame_num', 'bb_lt_x', 'bb_lt_y', 'bb_width', 'bb_height', 'obj_type', 'video_name']
    resultdf = pd.DataFrame(columns=columns)
    del columns[-1]                                         # Delete video name as its not in the file

    # This section of commented code was used identify video files with corrupted object files
    # ignorevids = []
    # ignoreindex = df[df.video_name.isin(ignorevids)].index
    # df.drop(ignoreindex, inplace=True)

    for index, row in df.iterrows():
        print('Processing file ({}): {}'.format(index + 1, row['video_name']))
        objects = pd.read_csv(row['object_path'], sep='\s', header=None, engine='python')
        objects.columns = columns
        objects['video_name'] = row['video_name']           # Adding video name to the data
        objects = objects[objects.obj_type == 1]            # Filtering only Person data
        resultdf = resultdf.append(objects, ignore_index=True, sort=False)

    resultdf.drop('obj_dur', axis=1, inplace = True)        # dropping un-necessary column

    import sqlite3
    rootDir = 'C:\\DS_ML\\Video_Analytics_YOLOv4'
    dbDir = 'C:\\DS_ML\\Video_Analytics_YOLOv4\\Database_n_Files'
    os.chdir(dbDir)
    connection = sqlite3.connect('videoAnalytics.db')
    resultdf.to_sql(tabname, connection, if_exists='replace', index=False)
    os.chdir(rootDir)

def readAnnotData():
    '''
    This function read trainannot and testannot table from database and returns dataframes
    :return: Returns Train and Test Video dataframes
    '''
    import sqlite3
    rootDir = 'C:\\DS_ML\\Video_Analytics_YOLOv4'
    dbDir = 'C:\\DS_ML\\Video_Analytics_YOLOv4\\Database_n_Files'
    os.chdir(dbDir)
    connection = sqlite3.connect('videoAnalytics.db')
    cursor = connection.cursor()
    # Check if table trainannot exists
    cursor.execute('SELECT name FROM sqlite_master WHERE name = "trainannot"')
    trvid = tsvid = 0
    if cursor.fetchall()[0][0] == 'trainannot':
        traindf = pd.read_sql("SELECT * FROM trainannot", connection)
        print('Training annotaions table is present and contains {} objects'.format(traindf.shape[0]))
        trvid = 1
        # print(traindf.head())
    else:
        print('Training annotations table is not present')

    cursor.execute('SELECT name FROM sqlite_master WHERE name = "testannot"')
    if cursor.fetchall()[0][0] == 'testannot':
        testdf = pd.read_sql("SELECT * FROM testannot", connection)
        print('Testing annotation table is present and contains {} objects'.format(testdf.shape[0]))
        tsvid = 1
        # print(testdf.head())
    else:
        print('Testing annotations table is not present')

    if (trvid == 1) and (tsvid == 1):
        return traindf, testdf
    os.chdir(rootDir)

def miniBatchData(traindf, testdf, batches):
    '''
    This function can be used to divide large dataset into smaller chunks
    :param traindf: Expects training data frame containing videos and annotation file path
    :param testdf: Expects testing data frame containing videos and annotation file path
    :param batches: Expects number of batches data needs to be split into
    :return: Returns training and test dataframes with additional column containing batch number
    '''
    trainVidNo  =   len(traindf) // batches + 1
    testVidNo   =   len(testdf) // batches + 1
    traindf['batchno']  = [i for i in range(1 , batches+1) for vid in range(trainVidNo)][:len(traindf)]
    testdf['batchno']   = [i for i in range(1 , batches+1) for vid in range(testVidNo)][:len(testdf)]
    return traindf, testdf

def genYoloTrnDt(mddf, andf, tgtfld):
    '''
    Function to generate YOLO training data:-
     - Generate images from video paths in given folder, as YOLO can
        train on video, it needs individual image files
     - Generate text file containing annotation for individual image file
        * Annotations should be scaled based on network size (height and width)
        * Left top coordinates should be converted to centroid of the bounding box
     - If the frame doesn't contain any annotation data, create an empty text file
        This will be a negative sample
     - Generate text file containing path of all the image file
    :param mddf: Meta Data dataframe containing valid video paths
    :param andf: Annotation Data dataframe containing bounding box coordinates
    :param tgtfld: New folder name
    :return:        N/A
    '''
    imglist = []

    for index, row in mddf.iterrows():
        if index == 0:  # Create directory for the first video and keep adding files for remaining videos
            if os.path.exists(tgtfld):
                shutil.rmtree(tgtfld)
            os.mkdir(tgtfld)

        print('Processing file ({}): {}'.format(index + 1, row['video_name']))
        vidcap = cv2.VideoCapture(row['video_path'])
        imgWd = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        imgHt = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('Width = {}, Height = {}'.format(imgWd, imgHt))

        frame_count = 0

        while(vidcap.isOpened()):
            res, frame = vidcap.read()
            if res == True:
                vidfile = os.path.join(tgtfld, row['video_name'].split('.')[0] + '_{}.jpg'.format(frame_count))
                cv2.imwrite(vidfile, frame)
                # print('Image File: {}'.format(vidfile))
                annotfile = os.path.join(tgtfld, row['video_name'].split('.')[0] + '_{}.txt'.format(frame_count))
                framedf = andf[(andf['video_name'] == row['video_name']) & (andf['frame_num'] == frame_count)].copy()
                # Convert left top coordinates to centre coordinates
                framedf['bb_lt_x'] = framedf['bb_lt_x'] + framedf['bb_width'] / 2
                framedf['bb_lt_y'] = framedf['bb_lt_y'] + framedf['bb_height'] / 2
                # Normalize the coordinates
                framedf['bb_lt_x']   =   framedf['bb_lt_x']   / imgWd
                framedf['bb_lt_y']   =   framedf['bb_lt_y']   / imgHt
                framedf['bb_width']  =   framedf['bb_width']  / imgWd
                framedf['bb_height'] =   framedf['bb_height'] / imgHt
                file = open(annotfile, mode='w+')
                # file.write(framedf.to_string(header=False, index=False))    # this option is generating extra spaces
                file.writelines('{} {} {} {} {}\n'.format(0, line[2], line[3], line[4], line[5]) for line in framedf.values)
                file.close()
                # print('Image File: {}'.format(annotfile))
                imglist.append(vidfile)

                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break

        print('No. of images written for this video is {}'.format(frame_count))
        vidcap.release()

    imgpath = os.path.join(tgtfld + '/image_path.txt')
    file = open(imgpath, mode='w+')
    file.writelines('{}\n'.format(line) for line in imglist)
    file.close()
    print('Image Path is written in the file {}'.format(imgpath))



if __name__ == '__main__':
    # loadMetaData(vidFolder, annotFolder, dispWrt='writedb')
    trainmd, testmd = readMetaData()
    # print('Training videos:\t{}\nTesting videos:\t{}'.format(trainmd.shape[0], testmd.shape[0]))
    # loadAnnotObj(df = trainmd, tabname = 'trainannot')
    # loadAnnotObj(df = testmd, tabname='testannot')
    trainad, testad = readAnnotData()
    # print('Train data size:\t{}\nTest data size:\t{}'.format(trainad.shape[0], testad.shape[0]))
    trainmb, testmb = miniBatchData(traindf = trainmd, testdf = testmd, batches = 10)
    # print(trainmb[trainmb.batchno == 1], testmb[testmb.batchno == 1])
    # # Get training image count for all the batches
    # for i in range(11):
    #     trvideos = trainmb[trainmb.batchno == i]['video_name']
    #     print('Batch {} Image count: {}'.format(i,len(trainad[trainad.video_name.isin(trvideos)])))

    genYoloTrnDt(mddf=trainmb[trainmb.batchno == 1], andf=trainad, tgtfld=os.path.join(dataFolder, 'train'))
    # genYoloTrnDt(df=testmb[testmb.batchno == 1], tgtfld=os.path.join(dataFolder, 'test'))