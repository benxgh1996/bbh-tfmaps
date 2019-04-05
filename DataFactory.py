from TfInstance import *
from TfMaker import *
import random as rand
import utils
import os
PATH = os.path
from utilsWaveform import *
import matplotlib.pyplot as plt
from PIL import Image
import copy


# This class represents an object that holds an array of labelled
# time-frequency maps. It is also meant to be a wrapper class that
# contains functions to manipulate data for machine learning.
class DataFactory(object):
	# @param tfData (ndarray)
	def __init__(self, tfData):
		assert isinstance(tfData, np.ndarray)
		assert isinstance(tfData[0], TfInstance)
		self.tfData = tfData

	# @param numTrainWave (int): the number of waveforms whose
	#	annotated instances will be used for training.
	# @param numTestWave (int): the number of waveforms whose
	#	annotated instances will be used for testing.
	def splitData(self, numTrainWave, numTestWave):
		# A set that contains the waveform names in the entire
		# data set.
		nameSet = set()
		for instance in self.tfData:
			nameSet.add(instance.waveformName)
		nameList = list(nameSet)
		assert len(nameList) == numTrainWave + numTestWave

		# Getting the names for the test set.
		testNames = rand.sample(nameList, numTestWave)

		# Getting the training set and testing set.
		trainSet = set()
		testSet = set()
		for instance in self.tfData:
			if instance.waveformName in testNames:
				testSet.add(instance)
			else:
				trainSet.add(instance)
		assert len(trainSet) + len(testSet) == len(self.tfData)
		assert len(trainSet & testSet) == 0
		trainSet = np.array(list(trainSet))
		testSet = np.array(list(testSet))
		return trainSet, testSet

	# Helper function to obtain the absolute path to the directory
	# 	that contains the HDF5 waveform files.
	# 	This path seeking procedure is specific only to my (Gonghan Xu)
	# 	own machine. It needs to be changed if this script is run
	# 	on some other machine.
	@staticmethod
	def getWaveDirPath():
		BBH_DIR = PATH.dirname(PATH.abspath(__file__))
		return PATH.join(BBH_DIR, "..", "lvcnr-lfs", "GeorgiaTech")

	# Generate the actual time-frequency images (as image files) from
	# the labelled data. The generated image data are to be used for
	# training the neural net.
	# @param tfData (array<TfInstance>): An object array that holds
	#	the annotated data.
	# @param imDirPath (str): the path to the directory that will hold
	#	the generated image files. savePath can be either an
	#	absolute path or a path relative to the current directory.
	# @param tfMaker (TfMaker): A TfMaker object that has been
	# 	preconfigured to generate time-frequency images from a
	# 	waveform file at certain parameter settings.
	# @param genMenu (bool): a flag that marks whether we want to generate
	# 	a menu (txt) file that lists each generated image and its label.
	#	The label will be 1 for double chirp and 0 for no double chirp.
	# @param menuName (str): The name (e.g. train.txt) for the menu file.
	@staticmethod
	def genIm(tfData, imDirPath, tfMaker, genMenu=True,
			  menuName=None):
		# An array of TfInstance.
		# menuFile will become a file object when appropriate.
		menuFile = None
		if not genMenu:
			# We don't want to generate a menu.
			if menuName is not None:
				# We still have a supplied menu name.
				raise Exception("An unnecessary menuName is supplied "
								"with no intent of generating menu file.")
		else:
			# We do want to generate a menu.
			if menuName is None:
				raise Exception("menuName not supplied.")
			elif PATH.isfile(menuName):
				# Note that this assumes that the menu file will always
				# be generated inside the current directory.
				raise Exception("{} already exists as a file."
								.format(menuName))
			else:
				# Now we know we do want to create a menu file and
				# we have a supplied menu file name.
				# We need to obtain a file object for the menu file for
				# writing onto.
				menuFile = open(menuName, "w")

		# Obtaining the path to the waveform directory.
		waveDirPath = DataFactory.getWaveDirPath()

		print "Starting generating a total of {} images.."\
			.format(len(tfData))
		counter = 0
		# Create and save an image for each labelled tfInstance.
		# Write to the menuFile when appropriate.
		for tfInstance in tfData:
			counter += 1
			print "Generating the {}st image..".format(counter)
			# Convert each labelled tfInstance into an image array.
			im = tfMaker.tfInstance2Im(tfInstance, waveDirPath)
			imName = DataFactory.getImFileName(tfInstance)

			# The following way has been checked to be able to
			# save the image to the desired directory. Note that
			# imDirPath can be either an absolute path or a relative
			# path to the current directory.
			# plt.imsave(PATH.join(imDirPath, imName), im, cmap="gray")
			im = Image.fromarray(im)
			im.save(PATH.join(imDirPath, imName))

			if genMenu:
				# Write the corresponding entry onto the menu file.
				hasDoubleChirp = tfInstance.hasDoubleChirp
				# There is no point of saving a menu file if the
				# time-frequency maps have not been labelled.
				assert hasDoubleChirp is not None
				menuFile.write(imName + " " +
							   str(int(hasDoubleChirp)) + "\n")
		menuFile.close()

	# Get the appropriate file name representing a time-frequency image.
	@staticmethod
	def getImFileName(tfInstance):
		x = tfInstance
		# The values of iota, phi, and motherFreq are all left shifted by
		# two decimals to make them appear as integers. This is to avoid
		# multiple dots in a file name.
		iota = x.iota * 180 / pi
		phi = x.phi * 180 / pi
		imName = "{}_{:d}_{:d}_{:d}.tiff".\
			format(x.waveformName, int(round(iota*100)),
				   int(round(phi*100)), int(round(x.motherFreq*100)))
		return imName

	# This method is used to synthesize an array of amplitude arrays
	# 	and another array of labels. The array of amplitude arrays
	# 	represents images as potential input data for machine learning.
	#	The array of labels are used for supervised learning.
	#	Note that the image data may need to be preprocessed before
	#	being supplied as features into a machine learning algorithm.
	# @param tfData (ndarray<TfInstance>): An array of heavy TfInstance.
	#	This means each object in this array should encapsulate a
	#	substantial attribute of ampArr and a substantial attribute of
	#	hasDoubleChirp.
	# @param prob (bool): If True, assign floating point numbers as
	#	labels. If False, assign string values as labels.
	# @return imArr (ndarray<*, *, *>): An array of the images of the
	#	labelled time-frequency maps.
	# @return labelList (array<str-like>): An array of str-likes,
	# 	corresponding to the labels for the synthesized image array.
	@staticmethod
	def getTrainableArrays(tfData, prob=False):
		imArr = []
		labelArr = []
		for tfInstance in tfData:
			assert tfInstance.hasDoubleChirp is not None
			assert tfInstance.ampArr is not None
			imArr.append(tfInstance.ampArr)
			if prob:
				if tfInstance.hasDoubleChirp:
					labelArr.append(1.)
				else:
					labelArr.append(0.)
			else:
				if tfInstance.hasDoubleChirp:
					labelArr.append("Double Chirp")
				else:
					labelArr.append("Not Double Chirp")
		imArr = np.array(imArr)
		labelArr = np.array(labelArr)
		assert len(imArr) == len(labelArr)
		return imArr, labelArr

	# This method is used to create machine-trainable data from an
	#	existing set of light-weight labelled time-frequency map
	# 	data. So essentially, this method is converting an light-weight
	#	array of labelled TfInstance data into a heavy array of
	# 	labelled TfInstance data, where each TfInstance object will
	#	encapsulate substantial attributes of freqArr, timeArr, and
	#	ampArr.
	# @param tfData array<TfInstance>: An array of TfInstance that
	#	contains light-weight TfInstance objects.
	# @return tfData array<TfInstance>: An array of heavy TfInstance.
	@staticmethod
	def getTrainableData(tfData, tfMaker):
		numIns = len(tfData)
		print "Generating a total of {} trainable TfInstance..."\
			.format(numIns)
		# Making a deep (recursive) copy of tfData because we want to
		# build up an array of heavy TfInstance directly upon an array
		# of light TfInstance, without messing with the passed-in
		# tfData. I have tested deepcopy, and it seems to work as
		# as intended in this use case.
		assert tfData[0].isLight()
		tfData = copy.deepcopy(tfData)
		wavePathDir = DataFactory.getWaveDirPath()
		for idx, tfIns in enumerate(tfData, start=1):
			print "Generating the {}st/{} TfInstance..".format(idx, numIns)
			assert tfIns.isLight()
			assert tfIns.hasDoubleChirp is not None
			ampArr, timeArr, freqArr = \
				tfMaker.tfInstance2Im(tfIns, wavePathDir, includeTf=True)
			tfIns.ampArr = ampArr
			tfIns.timeArr = timeArr
			tfIns.freqArr = freqArr
		return tfData

	# Convert an light-weight, labelled TfInstance array into
	# a heavy array, and then save this array for persistence.
	@staticmethod
	def saveTrainableData(tfData, tfMaker, savePath):
		print "Saving trainable data..."
		dat = DataFactory.getTrainableData(tfData, tfMaker)
		np.save(savePath, dat)


if __name__ == "__main__":
	trainSet = np.load("testSet.npy")
	imMaker = TfMaker()
	DataFactory.genIm(trainSet, "testTiffDir/", imMaker, genMenu=True,
					  menuName="testTiffMenu.txt")
	pass
