import sys
import model
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import read_dataset
from sklearn import preprocessing

def main():
	# set mode
	try:
		mode = sys.argv[1]
		assert (mode == 'dnn' or mode == 'cnn')
	except:
		print('Error: Model mode not found')
		exit()
	read_dataset()
	# load data
	try:
		img_name = sys.argv[2]
		img = mpimg.imread(img_name)
	except:
		print('Error: Img not found')
		exit()
	img = np.array([img])
	o_shape = img.shape
	img = np.reshape(img, (1, -1))
	#img = preprocessing.scale(img)
	img = (img-0.5)*4
	img = np.reshape(img, (1, 48, 48, 1))
	classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
	
	# load model
	emotion_classifier = model.build_model(mode)
	emotion_classifier.load_weights(mode+'.h5')
	
	# predict
	predictions = emotion_classifier.predict_classes(img)
	print(classes[predictions[0]])

if __name__ == '__main__':
	main()
