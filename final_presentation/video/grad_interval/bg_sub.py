from __future__ import print_function
import cv2 as cv
import argparse
import time
import numpy as np
from numpy.fft import fft2, ifft2

smear = 2
threshold = smear*smear
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='people_pavement_sea2.mov')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()



##############################################################
def bkgSub(inFrame,show=False) :
    frame = inFrame
    #update the background model
    fgMask = backSub.apply(frame)
    original_size = frame.shape
    kernel = np.ones((4,4))
    fgMask = cv.erode(fgMask, kernel)
    kernel = np.array([[0,0,3,0,0],
                        [0,0,2,0,0],
                        [1,2,4,2,1],
                        [0,0,3,0,0],
                        [0,0,3,0,0]]).astype(np.uint8)
    fgMask = cv.erode(fgMask, kernel)

    # Transfer mask to numpy
    iMask = np.array(fgMask.reshape((im_height, im_width)).astype(np.uint8))
    
    # Clean up mask
    iMask = (iMask>127).astype(np.uint8)

    # # dilson
    kernel = np.ones((2,2))
    iMask = cv.dilate(iMask, kernel)
    iMask = cv.morphologyEx(iMask, cv.MORPH_CLOSE, kernel)

    # # Eliminate isolated pixels
    oMask = iMask
    for k in range(1,smear) :
        oMask[:,:im_width-k-1] += iMask[:,k:im_width-1]
        oMask[:,k:im_width-1] += iMask[:,:im_width-k-1]
    iMask = oMask
    for k in range(1,smear) :
        oMask[:im_height-k-1,:] += iMask[k:im_height-1,:]
        oMask[k:im_height-1,:] += iMask[:im_height-k-1,:]
    oMask = oMask>threshold
    iMask = oMask
    for k in range(smear) :
        oMask[:,:im_width-k-1] |= iMask[:,k:im_width-1]
        oMask[:,k:im_height-1] |= iMask[:,:im_height-k-1]
    iMask = oMask
    for k in range(smear) :
        oMask[:im_height-k-1,:] |= iMask[k:im_height-1,:]
        oMask[k:im_height-1,:] |= iMask[:im_height-k-1,:]
    oMask = 255 * oMask.astype(np.uint8)

    #show the current frame and the fg masks
    if (show) :
        cv.imshow('FG Mask', fgMask)
        cv.imshow('Contours', oMask)
    frame = cv.bitwise_and(frame,frame,mask = oMask)
    return frame

##############################################################
#create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

# Open input
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
#capture = cv.VideoCapture(1)
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
# Open output
video_output = 'BkgOut.mp4'
fourcc = cv.VideoWriter_fourcc(*'mp4v')
im_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
im_width= int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
output_fps = capture.get(cv.CAP_PROP_FPS)
out = cv.VideoWriter(video_output, fourcc, output_fps, (im_width, im_height))


while True:
    ret, frame = capture.read()
    if frame is None:
        break

    #background suppression
    outFrame = bkgSub(frame,True)

    #get the frame number and write it on the current frame
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    #show
    cv.imshow('Frame', frame)
    cv.imshow('Masked', outFrame)
    out.write(outFrame)

    keyboard = cv.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break
out.release()
capture.release()
cv.destroyAllWindows()
