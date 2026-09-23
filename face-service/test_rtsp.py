import cv2
import time

t = time.time()
cap = cv2.VideoCapture('rtsp://mediamtx:8554/1', cv2.CAP_FFMPEG)
print('opened:', cap.isOpened(), 'took', time.time() - t, 's')

if cap.isOpened():
    t2 = time.time()
    ret, frame = cap.read()
    print('read:', ret, 'shape:', None if frame is None else frame.shape, 'took', time.time() - t2, 's')