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
	norm = True
	batch = 32
	epoch = 5
	vs = 0.05

	# load data	
	tr_feats, _, tr_labels, _ = read_dataset(tpath, norm)
	
	# build model
	emotion_classifier = model.build_model(mode)
	
	# start training
	emotion_classifier.fit(tr_feats, tr_labels, batch_size = batch, epochs = epoch, validation_split = vs)
	
	# save model
	emotion_classifier.save_weights(mode+'.h5')

if __name__ == "__main__":
	main()
