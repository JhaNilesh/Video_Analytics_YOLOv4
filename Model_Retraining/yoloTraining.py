"""yoloInference.py: Program for training YOLOv4 on custom data"""
__author__ = 'Nilesh Jha'
__email__ = "nilesh2sonu@gmail.com"

import os
import subprocess

def yolo4Train():
    '''
    This function initiates Training for YOLO4 model on custom dataset
    :return: NA
    '''
    rootDir     =   'C:\\DS_ML\\Video_Analytics_YOLOv4'
    darknetDir  =   os.path.join(rootDir, 'YOLOv4', 'darknet')
    objData     =   'build\\darknet\\x64\\data\\obj.data'
    cfgFile     =   'cfg\\yolo-obj.cfg'
    weightPath  =   os.path.join(darknetDir, 'build', 'darknet', 'x64', 'yolov4.conv.137')

    os.chdir(darknetDir)
    command = ['darknet.exe', 'detector', 'train', objData, cfgFile, weightPath, '-map']
    train = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    os.chdir(rootDir)
    trnLog = [line.decode.strip() for line in train.stdout.splitlines()]
    logpath = os.path.join(rootDir, 'Database_n_Files', 'trainlog.txt')
    file = open(logpath, mode='w+')
    file.writelines('{}\n'.format(line) for line in trnLog)
    file.close()
    print('Training log is written in the file {}'.format(logpath))

if __name__ == '__main__':
    yolo4Train()





'''
TODO:   Function for training and capturing mAP values during training

TODO:   Function for plotting following curves
        - mAP and loss curve vs iteration #
        - Error vs Iteration # for training and validation set
'''