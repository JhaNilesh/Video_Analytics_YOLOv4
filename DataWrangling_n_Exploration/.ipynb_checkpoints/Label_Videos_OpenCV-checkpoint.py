'''
Draw label in video dataset using OpenCV annotation drawing.

==========================
Annotation Files
==========================

    Annotations files include three types of annotations per clip
        1) Event file (selected events annotated)
        2) Mapping file (from event to object)
        3) Object file (all objects annotated)

    The detailed formats of the above three types of (linked) files are described below.


        1) Event file format

    Files are named as '%s.viratdata.events.txt' where %s is clip id.
    Each line in event file captures information about a bounding box of an event at the corresponding frame

    Event File Columns
        1: event ID        (unique identifier per event within a clip, same eid can exist on different clips)
        2: event type      (event type)
        3: duration        (event duration in frames)
        4: start frame     (start frame of the event)
        5: end frame       (end frame of the event)
        6: current frame   (current frame number)
        7: bbox lefttop x  (horizontal x coordinate of left top of bbox, origin is lefttop of the frame)
        8: bbox lefttop y  (vertical y coordinate of left top of bbox, origin is lefttop of the frame)
        9: bbox width      (horizontal width of the bbox)
        10: bbox height    (vertical height of the bbox)

    Event Type ID (for column 2 above)
        1: Person loading an Object to a Vehicle
        2: Person Unloading an Object from a Car/Vehicle
        3: Person Opening a Vehicle/Car Trunk
        4: Person Closing a Vehicle/Car Trunk
        5: Person getting into a Vehicle
        6: Person getting out of a Vehicle
        7: Person gesturing
        8: Person digging
        9: Person carrying an object
        10: Person running
        11: Person entering a facility
        12: Person exiting a facility


    2) Object file format

    Files are named as '%s.viratdata.objects.txt'
    Each line captures information about a bounding box of an object (person/car etc) at the corresponding frame.
    Each object track is assigned a unique 'object id' identifier.
    Note that:
        - an object may be moving or static (e.g., parked car).
        - an object track may be fragmented into multiple tracks.

    Object File Columns
        1: Object id        (a unique identifier of an object track. Unique within a file.)
        2: Object duration  (duration of the object track)
        3: Currnet frame    (corresponding frame number)
        4: bbox lefttop x   (horizontal x coordinate of the left top of bbox, origin is lefttop of the frame)
        5: bbox lefttop y   (vertical y coordinate of the left top of bbox, origin is lefttop of the frame)
        6: bbox width       (horizontal width of the bbox)
        7: bbox height      (vertical height of the bbox)
        8: Objct Type       (object type)

    Object Type ID (for column 8 above for object files)
        1: person
        2: car              (usually passenger vehicles such as sedan, truck)
        3: vehicles         (vehicles other than usual passenger cars. Examples include construction vehicles)
        4: object           (neither car or person, usually carried objects)
        5: bike, bicylces   (may include engine-powered auto-bikes)


    3) Mapping file format

    Files are named as '%s.viratdata.mapping.txt'
    Each line in mapping file captures information between an event (in event file) and associated objects
    (in object file)

    Mapping File Columns
        1: event ID         (unique event ID, points to column 1 of event file)
        2: event type       (event type, points to column 2 of event file)
        3: event duration   (event duration, points to column 3 of event file)
        4: start frame      (start frame of event)
        5: end frame        (end frame of event)
        6: number of obj    (total number of associated objects)
        7-end:              (variable number of columns which captures the associations maps for variable number
                            of objects in the clip. If '1', the event is associated with the object. Otherwise,
                            if '0', there's none. The corresponding oid in object file can be found by 'column
                            number - 7')
'''

import cv2 as cv
import pandas as pd
import numpy as np

# Generating Events Dataframe
events = pd.read_csv('VIRAT Ground Video Dataset 2.0/annotations/VIRAT_S_000002.viratdata.events.txt',
                     sep = '\s', header = None)
events.columns = ['Event_ID','Event_Type','Duration','Start_frame','End_Frame',
                  'Curr_Frame','BB_LT_X','BB_LT_Y','BB_Width','BB_Height']
event_description = {
                            1: 'Person loading an Object to a Vehicle',
                            2: 'Person Unloading an Object from a Car/Vehicle',
                            3: 'Person Opening a Vehicle/Car Trunk',
                            4: 'Person Closing a Vehicle/Car Trunk',
                            5: 'Person getting into a Vehicle',
                            6: 'Person getting out of a Vehicle',
                            7: 'Person gesturing',
                            8: 'Person digging',
                            9: 'Person carrying an object',
                            10: 'Person running',
                            11: 'Person entering a facility',
                            12: 'Person exiting a facility'
                    }
events['Event_Des'] = events['Event_Type'].apply(lambda x: event_description.get(x))
unq_eid = events['Event_ID'].drop_duplicates()
unq_eid['Random'] = unq_eid.apply(lambda x: np.random.randint(0,255))
events['Color'] = events['Event_ID'].apply(lambda x: np.random.randint(0,255))
print(unq_eid)


# Generating Objects Dataframe
objects = pd.read_csv('VIRAT Ground Video Dataset 2.0/annotations/VIRAT_S_000002.viratdata.objects.txt',
                     sep = '\s', header = None)
objects.columns = ['Obj_ID','Obj_Dur','Curr_Frame','BB_LT_X','BB_LT_Y','BB_Width','BB_Height','Obj_Type']
object_description = {
                            1: 'person',
                            2: 'car',
                            3: 'vehicles',
                            4: 'object',
                            5: 'bike or bicycles'
                    }
objects['Obj_Des'] = objects['Obj_Type'].apply(lambda x: object_description.get(x))
objects['Color'] = objects['Obj_ID'].apply(lambda x:  np.random.randint(0,255))

objects = objects[(objects.Obj_Type >= 1) & (objects.Obj_Type <= 5)]
# Generating Mapping Dataframe

mapping = pd.read_csv('VIRAT Ground Video Dataset 2.0/annotations/VIRAT_S_000002.viratdata.mapping.txt',
                     sep = '\s', header = None)
#mapping = mapping.T
#mapping.columns = ['Event_ID','Event_Type','Duration','Start_frame','End_Frame','No_of_Obj']


# Capturing Video File

vidcap = cv.VideoCapture('VIRAT Ground Video Dataset 2.0/videos_original/VIRAT_S_000002.mp4')

# Window Parameters
cv.namedWindow('Video', cv.WINDOW_NORMAL)
cv.resizeWindow('Video', 1000, 750)

# Binding box parameters, events and objects text
font = cv.FONT_ITALIC

#color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
frame_count = 0

while(vidcap.isOpened()):

    res, frame = vidcap.read()

    if res == True:                                                   # Check for read response

        # Labelling Events
        epf = events[events.Curr_Frame == frame_count].values.tolist()
        if len(epf) > 0:                                               # If there are non-zero events for a frame
            for epft in epf:                                           # There can be multiple events for a frame
                scoord = (epft[6], epft[7])
                event_str = str(epft[0]) + ': ' + epft[10]
                pt2_x = epft[6] + epft[8]
                pt2_y = epft[7] + epft[9]
                ecoord = (pt2_x, pt2_y)
                tcoord = (epft[6], epft[7] - 5)
                color = (epft[11], 255-epft[11], epft[11])
                frame = cv.putText(frame, event_str, tcoord, font, 1, color, 2)
                frame = cv.rectangle(frame, scoord, ecoord, color, 3)

        # Labelling Objects
        opf = objects[objects.Curr_Frame == frame_count].values.tolist()
        if len(opf) > 0:                                               # If there are non-zero objects for a frame
            for opft in opf:                                           # There can be multiple objects for a frame
                scoord = (opft[3], opft[4])
                object_str = str(opft[0]) + ': ' + str(opft[8])
                pt2_x = opft[3] + opft[5]
                pt2_y = opft[4] + opft[6]
                ecoord = (pt2_x, pt2_y)
                tcoord = (opft[3], opft[4] - 5)
                color = (opft[9], 255 - opft[9], 255 - opft[9])
                frame = cv.putText(frame, object_str, tcoord, font, 1, color, 2)
                frame = cv.rectangle(frame, scoord, ecoord, color, 3)


        cv.imshow('Video', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    frame_count += 1

vidcap.release()
cv.destroyAllWindows()



#frm_rt = vidcap.get(cv.CAP_PROP_FPS)
#vidcap.set(cv.CAP_PROP_FRAME_WIDTH, 750)
#vidcap.set(cv.CAP_PROP_FRAME_HEIGHT, 500)
#print(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
#print(frame_count,'\t',vidcap.get(cv.CAP_PROP_FRAME_COUNT))
#event_fl = open('VIRAT Video Dataset 2.0/VIRAT Ground Dataset/annotations/VIRAT_S_000002.viratdata.events.txt','r')
#events = event_fl.read()
#objects_fl = open('VIRAT Video Dataset 2.0/VIRAT Ground Dataset/annotations/VIRAT_S_000002.viratdata.objects.txt','r')
#objects = objects_fl.read()
#mapping_fl = open('VIRAT Video Dataset 2.0/VIRAT Ground Dataset/annotations/VIRAT_S_000002.viratdata.mapping.txt','r')
#mapping = mapping_fl.read()