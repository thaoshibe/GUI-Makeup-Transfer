# from mtcnn import MTCNN
import cv2, time, os
from utils import testcam
from facenet_pytorch import MTCNN
from PIL import Image
from utils import get_rotation_matrix, angle_between_2_points, extend_bb, preprocess, deprocess, check_biggest_face
import tensorflow as tf
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
return_img_size=365
class Magican():
	"""docstring for ClassName"""
	def __init__(self, refer_img = './makeup/0.png'):
		self.img_size = 256
		# self.refer_img = cv2.resize(cv2.imread(refer_img), (self.img_size, self.img_size))
		self.refer_img = cv2.imread(refer_img)
		self.refer_img = cv2.resize(self.refer_img, (return_img_size, return_img_size))
		self.refer_img = cv2.cvtColor(self.refer_img, cv2.COLOR_RGB2BGR)
		self.Y_img = np.expand_dims(preprocess(cv2.resize(self.refer_img, (256, 256))), 0)
		self.mtcnn = MTCNN(select_largest=False, post_process=False, device='cuda:0')
		# tf.reset_default_graph()
		with tf.device('/device:GPU:0'):
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			config.allow_soft_placement = True
			self.sess = tf.Session(config=config)
			self.sess.run(tf.global_variables_initializer())
			saver = tf.train.import_meta_graph(os.path.join('model', 'model.meta'))
			saver.restore(self.sess, tf.train.latest_checkpoint('model'))
		graph = tf.get_default_graph()
		self.X = graph.get_tensor_by_name('X:0')
		self.Y = graph.get_tensor_by_name('Y:0')
		self.Xs = graph.get_tensor_by_name('generator/xs:0')


	def detector(self, frame):
		boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
		return boxes, landmarks

	def new_refer(self, refer_path):
		self.refer_img = cv2.imread(refer_path)
		self.refer_img = cv2.cvtColor(self.refer_img, cv2.COLOR_RGB2BGR)
		self.refer_img = cv2.resize(self.refer_img, (return_img_size, return_img_size))
		self.Y_img = np.expand_dims(preprocess(cv2.resize(self.refer_img, (256, 256))), 0)

	def algin_face(self, frame, algin=False):
		self.frame = frame
		boxes, landmarks = self.detector(frame)
		if boxes is None:
			self.no_face = True
			null = cv2.imread('./static/null.png')
			self.face = cv2.resize(null, (return_img_size, return_img_size))
			return self.face
		else:
			index_face = check_biggest_face(boxes)
			x1, x2, y1, y2 = extend_bb(boxes, index_face)
			left_eye = list(landmarks[0][0])
			right_eye = list(landmarks[0][1])
			height, width = frame.shape[:2]
			s_height, s_width = height, width
			cropped = frame[x1:x2, y1:y2, :]
			self.no_face = False
			self.face = cv2.resize(cropped, (256, 256))
			cv2.rectangle(self.frame, (y2, x2), (y1, x1), (250, 124, 24),3)
			return cv2.resize(cropped, (return_img_size, return_img_size))

	def makeup(self):
		if self.no_face:
			return self.face
		else:
			self.face = cv2.cvtColor(self.face, cv2.COLOR_RGB2BGR)
			no_makeup = cv2.resize(self.face, (self.img_size, self.img_size))
			X_img = np.expand_dims(preprocess(no_makeup), 0)
			Xs_ = self.sess.run(self.Xs, feed_dict={self.X: X_img, self.Y: self.Y_img})
			Xs_ = deprocess(Xs_)
			return cv2.cvtColor(cv2.resize(Xs_[0]*255, (return_img_size, return_img_size)), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
	Magican()
