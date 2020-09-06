# VIDEO ANALYTICS USING YOLOv4

## **Introduction  :**

Artificial Intelligence (AI) and Machine Learning (ML) are the top technologies that has great impact on Retail Industry. Both online and traditional brick and mortar retail store can leverage the power of deep learning and computer vision to improve customer experience, increase sales and reduce costs.
Although the possibilities are endless, in this project we will explore few applications of computer vision and Deep Learning for a typical brick and mortar store:

- **Activity recognition and behavioral tracking via video analytics:**
Computer vision can be used to recognize face and characteristics like age range, gender, gaze, body language etc. Factoring variables like seasonal patterns, weekday/weekend, day of week with the behavioral data we can generate insights that can help to dynamically modify product placements and create efficient promotions.

- **Multi-camera tracking to analyze navigational routes:**
We can track customers' journey in the store to understand walking patterns, direction of gaze etc. This information can then be used to discover locations that gets more traffic and visual attention and can be used to restructure store layouts, measure interest in products, product placements, signage placements etc. Syncing all cameras will help overcome challenge of a customer getting blocked from one of the cameras view.

## **Scope  :** 
- Recognize activities like a person is moving or not, sitting or not, dwelling etc
- Generate trend for total number of people w.r.t time
- No. of people in specific region of store at a time to analyze navigational routes. We will use multi-camera tracking to sync up all the cameras installed in the store
- Monitor the amount of time a customer is spending at a location to identify inefficiencies in product placement

## **Datasets  :**
Ideally, we would like to have Retail store camera recordings however its difficult to get it becasue of privacy concerns. We would however use pedestrain survilience data. We have obtained the data from following sources:
- **3dPeS Data:** 3DPeS (3D People Surveillance Dataset) is a surveillance dataset, designed mainly for people re-identification in multi camera systems with non-overlapped field of views, but also applicable to many other tasks, such as people detection, tracking, action analysis and trajectory analysis.
Data size is around 72 GB. I've placed a small slice (~6 GB) of this data on [google drive](https://drive.google.com/open?id=1dXcLk2FqDz53d_u-en4fU2lu7-FVx4M_ "google drive").
[Source](https://aimagelab.ing.unimore.it/imagelab/datasets.asp "Source")
- **HDA Person Dataset:** The HDA dataset is a fully labeled, multi-camera high-resolution image sequence dataset for research on high-definition surveillance. 18 cameras (including VGA, HD and Full HD resolution) were recorded simultaneously during 30 minutes in a typical indoor office scenario at a busy hour (lunch time) involving more than 80 persons.
Data size is around 3.5 GB. It is also placed on [google drive](https://drive.google.com/open?id=1WFGPn0SIaVfdeKttiW5jcJ0HhH6o2rCa "google drive").
 [Source](http://vislab.isr.ist.utl.pt/hda-dataset/ "Source")
- **VIRAT Video Dataset:** The VIRAT Video Dataset is designed to be realistic, natural and challenging for video surveillance domains in terms of its resolution, background clutter, diversity in scenes, and human activity/event categories than existing action recognition datasets. It has become a benchmark dataset for the computer vision community. 
Data Size is around 70 GB.
[Source](https://viratdata.org/ "Source")
