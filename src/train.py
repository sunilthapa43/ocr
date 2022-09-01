
import sys
import numpy as np
import cv2

im = cv2.imread('../data/train.png')
im3 = im.copy()
def sort_contours(contours, x_axis_sort='LEFT_TO_RIGHT', y_axis_sort='TOP_TO_BOTTOM'):
    # initialize the reverse flag
    x_reverse = False
    y_reverse = False
    if x_axis_sort == 'RIGHT_TO_LEFT':
        x_reverse = True
    if y_axis_sort == 'BOTTOM_TO_TOP':
        y_reverse = True
    
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    
    # sorting on x-axis 
    sortedByX = zip(*sorted(zip(contours, boundingBoxes),
    key=lambda b:b[1][0], reverse=x_reverse))
    
    # sorting on y-axis 
    (contours, boundingBoxes) = zip(*sorted(zip(*sortedByX),
    key=lambda b:b[1][1], reverse=y_reverse))
    # return the list of sorted contours and bounding boxes
    return (contours, boundingBoxes)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

#################      Now finding Contours         ###################

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours, boundingBoxes = sort_contours(contours, x_axis_sort='RIGHT_TO_LEFT', y_axis_sort='TOP_TO_BOTTOM')

samples = np.empty((0, 100), np.float32)
responses = []
keys = [i for i in range(48, 58)]

for cnt in contours:

    if cv2.contourArea(cnt) > 50:
        [x, y, w, h] = cv2.boundingRect(cnt)

        if h > 20:

            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0),0)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            cv2.imshow('norm', im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1, 100))
                samples = np.append(samples, sample, 0)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255),0)
responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
print("training complete")

samples = np.float32(samples)
responses = np.float32(responses)

cv2.imwrite("../data/train_result.png", im)
np.savetxt('../data/generalsamples.data', samples)
np.savetxt('../data/generalresponses.data', responses)
