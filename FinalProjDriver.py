import cv2
import numpy as np
from collections import deque
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path
import pandas as pd


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

#red, orange, yellow, green, blue, purple, pink
#i have RGB
#it goes BGR
#colors = [(38, 38, 255)]
colors = [(38, 38, 255), (19, 148, 249), (17, 250, 250), (17, 250, 48), (255, 207, 47), (255, 47, 165), (248, 107, 253), (208, 77, 255)]
colorIndx = 0

redPts = [deque(maxlen=1000)]
orangePts = [deque(maxlen=1000)]
yellowPts = [deque(maxlen=1000)]
greenPts = [deque(maxlen=1000)]
bluePts = [deque(maxlen=1000)]
purplePts = [deque(maxlen=1000)]
pinkPts = [deque(maxlen=1000)]

redIndx = 0
orangeIndx = 0
yellowIndx = 0
greenIndx = 0
blueIndx = 0
purpleIndx = 0
pinkIndx = 0

clearcolor = 0
color0 = 0
color1 = 0
color2 = 0
color3 = 0
color4 = 0
color5 = 0
color6 = 0

def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for(percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX

    return bar, color, percent


while(1):

    _, frame = cap.read()
    jamDraw = frame.copy()
    blackImg = np.zeros((512, 512, 3), np.uint8)
    flippedIMG = cv2.flip(frame, 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #lowerRed = np.array([90, 100, 100])
    #upperRed = np.array([100, 255, 255])

    lowerRed = np.array([80, 250, 100])
    upperRed = np.array([100, 250, 255])

    #lowerRed = np.array([145, 100, 100])
    #upperRed = np.array([175, 255, 255])

    mask = cv2.inRange(hsv, lowerRed, upperRed)

    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(mask, kernel, iterations=2)

    ret, thresh = cv2.threshold(dilate, 15, 275, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #parameters: input, contours to be passed in, draw all contours (-1) or index to a specific one, color, thickness
    img = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    center = None
    img2 = None
    
    #drawing the colors to choose from
    points = [redPts, orangePts, yellowPts, greenPts, bluePts, purplePts, pinkPts]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                img2 = cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 23)

    img = cv2.putText(frame, "Sketch ", (160, 35), font, 1, (130, 100, 80), 1, cv2.LINE_AA)
    img = cv2.putText(frame, "Your ", (280, 35), font, 1, (255, 207, 47), 1, cv2.LINE_AA)
    img = cv2.putText(frame, "Mind!", (360, 35), font, 1, (205, 207, 47), 1, cv2.LINE_AA)
    img = cv2.putText(frame, "Clear", (0, 55), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    #clear
    img = cv2.rectangle(frame, (0, 60), (80, 90), (255, 255, 255), -1)
    
    #red
    img = cv2.rectangle(frame, (80, 60), (160, 90), colors[0], -1)
    #orange
    img = cv2.rectangle(frame, (160, 60), (240, 90), colors[1], -1)
    #yellow
    img = cv2.rectangle(frame, (240, 60), (320, 90), colors[2], -1)
    #green
    img = cv2.rectangle(frame, (320, 60), (400, 90), colors[3], -1)
    #blue
    img = cv2.rectangle(frame, (400, 60), (480, 90), colors[4], -1)
    #purple
    img = cv2.rectangle(frame, (480, 60), (560, 90), colors[5], -1)
    #pink
    img = cv2.rectangle(frame, (560, 60), (650, 90), colors[6], -1)
    #take picture
    img = cv2.rectangle(frame, (560, 390), (650, 420), (255, 255, 255), 3)
    img = cv2.rectangle(frame, (561, 391), (649, 419), (40, 40, 40), -1)


    img = cv2.putText(frame, "Snapshot", (490, 460), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if len(contours) > 0:

        M = cv2.moments(thresh)
 
        if (M['m00'] > 0):

            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 700, 700

        cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)

        center = cX, cY

        if center[1] <= 90:
            if 0 <= center[0] <= 80:
                redPts = [deque(maxlen=1000)]
                orangePts = [deque(maxlen=1000)]
                yellowPts = [deque(maxlen=1000)]
                greenPts = [deque(maxlen=1000)]
                bluePts = [deque(maxlen=1000)]
                purplePts = [deque(maxlen=1000)]
                pinkPts = [deque(maxlen=1000)]

                redIndx = 0
                orangeIndx = 0
                yellowIndx = 0
                greenIndx = 0
                blueIndx = 0
                purpleIndx = 0
                pinkIndx = 0

                clearcolor += 1

            elif 80 <= center[0] <= 160:
                colorIndx = 0
                color0 += 1
            elif 160 <= center[0] <= 240:
                colorIndx = 1
                color1 += 1
            elif 240 <= center[0] <= 320:
                colorIndx = 2
                color2 += 1
            elif 320 <= center[0] <= 400:
                colorIndx = 3
                color3 += 1
            elif 400 <= center[0] <= 480:
                colorIndx = 4
                color4 += 1
            elif 480 <= center[0] <= 560:
                colorIndx = 5
                color5 += 1
            elif 560 <= center[0] <= 650:
                colorIndx = 6
                color6 += 1
        elif center[1] >= 390:
            if center[0] >= 560:

                cv2.imwrite('jamDrawImg.jpg', img2)

                drawing = cv2.imread('jamDrawImg.jpg')
                drawing = drawing[90:390, 0:700]
             
                drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2RGB)

                drawing = drawing.reshape((drawing.shape[0] * drawing.shape[1], 3))
                clt = KMeans(n_clusters=3)
                clt.fit(drawing)

                hist = find_histogram(clt)
                bar, color, percent = plot_colors2(hist, clt.cluster_centers_)

                

                ##second histogram
                drawing2 = cv2.imread('jamDrawImg.jpg')
                drawing2 = drawing2[90:390, 0:700]
             
                drawing2 = cv2.cvtColor(drawing2, cv2.COLOR_BGR2RGB)

                drawing2 = drawing2.reshape((drawing2.shape[0] * drawing2.shape[1], 3))
                clt2 = KMeans(n_clusters=8)
                clt2.fit(drawing2)

                hist2 = find_histogram(clt2)
                bar2, color2, percent2 = plot_colors2(hist2, clt2.cluster_centers_)

                val1 = hist[0]
                val2 = hist[1]
                val3 = hist[2]

                a1 = min(val1, val2, val3)
                a3 = max(val1, val2, val3)
                a2 = (val1 + val2 + val3) - a1 - a3

                ranks = ([a1, a2, a3])

                tempo = ranks[0]
                valence = ranks[1]
                dance = ranks[2]  

                print('danceability: ' + str(dance))
      
                angryMood = color0 
                neutralMood = color3 + color6
                sadMood = color4 + color5

                print('sad: ' + str(sadMood))
                print('neutral: ' + str(neutralMood))
                print('angry: ' + str(angryMood))

                upperDance = dance + 0.002
                lowerDance = dance - 0.002  

                #data_path = Path('.') / 'Data'
                #data_files = list(data_path.glob("*.xlsx"))

                #data_pathMood = Path('.') / 'SecondData'
                #data_filesMood = list(data_pathMood.glob("*.xlsx"))

                df = pd.read_excel('cleanedMusicData.xlsx', 'Sheet1')
                dfMood = pd.read_excel('HPI-Countries.xlsx', 'Sheet1')

                #fileMood = data_filesMood[0]
                #dfMood = pd.read_excel(str(fileMood))

                #file = data_files[0]
                #df = pd.read_excel(str(file))

                df_dance = df[(df.danceability >= lowerDance) & (df.danceability <= upperDance)]

                moodVals = [sadMood, neutralMood, angryMood]

                sortedMood = sorted(moodVals)

                if(sortedMood[2] == sadMood):
                    HPI_r = dfMood[(dfMood.HPI_Rank > 50) & (dfMood.HPI_Rank < 70)]
                elif(sortedMood[2] == neutralMood):
                    HPI_r = dfMood[(dfMood.HPI_Rank > 0) & (dfMood.HPI_Rank < 50)]
                elif(sortedMood[2] == angryMood):
                    HPI_r = dfMood[(dfMood.HPI_Rank > 70)]

                finalMoodCOuntry = HPI_r[0:1]
                print("You should live here: ")
                print(finalMoodCOuntry['Country'])


                finalSong_df = df_dance[0:10]

                print('Here are 10 songs that represent your drawing: ')

                for i in range(0, 10):
                   
                    dateValues = finalSong_df[i:i+1]
                    songTitle = dateValues['title']
                    artist = dateValues['artist']
                    print('Song: ' + songTitle.astype(str) + " by " + artist.astype(str))
                    
               
                plt.axis("off")
                plt.imshow(bar)
                #plt.imshow(bar2)
                plt.show()

                break

        else:
            if colorIndx == 0:
                redPts[redIndx].appendleft(center)
            elif colorIndx == 1:
                orangePts[orangeIndx].appendleft(center)
            elif colorIndx == 2:
                yellowPts[yellowIndx].appendleft(center)
            elif colorIndx == 3:
                greenPts[greenIndx].appendleft(center)
            elif colorIndx == 4:
                bluePts[blueIndx].appendleft(center)
            elif colorIndx == 5:
                purplePts[purpleIndx].appendleft(center)
            elif colorIndx == 6:
                pinkPts[pinkIndx].appendleft(center)
    else:
        redPts.append(deque(maxlen=1000))
        redIndx += 1
        orangePts.append(deque(maxlen=1000))
        orangeIndx += 1
        yellowPts.append(deque(maxlen=1000))
        yellowIndx += 1
        greenPts.append(deque(maxlen=1000))
        greenIndx += 1
        bluePts.append(deque(maxlen=1000))
        blueIndx += 1
        purplePts.append(deque(maxlen=1000))
        purpleIndx += 1
        pinkPts.append(deque(maxlen=1000))
        pinkIndx += 1


    cv2.imshow("Frame", frame)

    k = cv2.waitKey(5) & 0xff

    if k == 27:
        break

cv2.destroyAllWindows()