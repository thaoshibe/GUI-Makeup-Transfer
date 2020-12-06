import cv2
import numpy as np
import tensorflow as tf

def testcam(start_index):
  for index in range(start_index, 10):
     cap = cv2.VideoCapture(index) 
     if cap is None or not cap.isOpened():
         print('Warning: unable to open video source: ', index)
     else:
      print('Cam {} is OK'.format(index))
      return index
  print('NO AVAILABLE CAMERA')
  return None

def check_biggest_face(bbox):
    arr=[]
    for i in range(0, len(bbox)):
        y1, x1, y2, x2 = bbox[i].T
        arr.append((y2-y1)*(x2-x1))
    return np.argmax(arr)

def extend_bb(bbox, index):
#     print('Intro ', rect, img.shape)
    y1, x1, y2, x2 = bbox[index].T
    w = y2 - y1
    h = x2 - x1
    ext = [h, w][np.argmax([h, w])]
    ext = int(ext*0.2)
    return int(x1-ext), int(x2+ext), int(y1-ext), int(y2+ext)

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def crop_image(image, det):
	top, bottom, left, right = rect_to_tuple(det, image)
	return image[top:bottom, left:right, :]

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

def check_tf():
  with tf.Session() as sess:
    devices = sess.list_devices()