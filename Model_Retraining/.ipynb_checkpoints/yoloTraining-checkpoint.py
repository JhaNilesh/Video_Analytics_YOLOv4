"""yoloInference.py: Program for building YOLOv4 training pipeline for videos"""
__author__ = 'Nilesh Jha'
__email__ = "nilesh2sonu@gmail.com"
from DataWrangling_n_Exploration.yoloDataPrep import readMetaData


'''
TODO:   Function to generate training and testing images in respective folder
        - Input parameters: training and testing video paths as array and target folder  
        - Create a train folder
        - Generate images with name <video name>_<frame_no>.jpg
        - Generate ground truth bb for each images
        - Do the same in testing folder

TODO:   Function to convert bounding box information in YOLO format
        - Left top coordinates to be converted to centroid by adding half
          of width and height to x and y coordinates respectively

TODO:   Function for scaling the bounding box information based on input size

TODO:   Function for training and capturing mAP values during training

TODO:   Function for plotting following curves
        - mAP and loss curve vs iteration #
        - Error vs Iteration # for training and validation set
'''

if __name__ == '__main__':
    dataFolder = 'C:\DS_ML\Video_Analytics_YOLOv4\Database_n_Files'
    trainmd, testmd = readMetaData()
    genImageFrmVid(df=trainmd, tgtfld=dataFolder, fldname='train')