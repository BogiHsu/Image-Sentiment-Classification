import os
import sys
import model
from utils import *

def main():
	# set mode
	try:
		mode = sys.argv[1]
		assert (mode == 'dnn' or mode == 'cnn')
	except:
		print('Error: Model mode not found')
		exit()

	# set parameters
	tpath = 'data.csv'
	div = True

	# load data	
	_, te_feats, _, te_labels = read_dataset(tpath, div)
	
	# load model
	emotion_classifier = model.build_model(mode)
	emotion_classifier.load_weights(mode+'.h5')
	
	# evaluation
	res = emotion_classifier.evaluate(te_feats, te_labels)
	print('Eval loss: ', round(res[0], 3), ' Eval acc: ', round(res[1], 3))

if __name__ == "__main__":
	main()
