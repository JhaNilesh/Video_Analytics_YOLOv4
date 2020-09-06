import os
import subprocess

os.chdir('C:\DS_ML\Retail_Stores_Video_Analytics\YOLOv4\darknet')
darknetPath =   'darknet.exe'
cocoData =      'cfg\coco.data'
cfgFile =       'cfg\yolov4.cfg'
weightPath =    'C:\DS_ML\Retail_Stores_Video_Analytics\yolov4.weights'
vidPath =       'C:\DS_ML\Retail_Stores_Video_Analytics\Test_Video.mp4'
yoloCmd = [darknetPath, 'detector', 'demo', cocoData, cfgFile, weightPath, vidPath, '-ext_output', '-dont_show']

yoloRes = subprocess.run(yoloCmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

print(yoloRes.stdout.decode())