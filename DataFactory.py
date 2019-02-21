from TfInstance import *
from TfMaker import *
import random as rand
import utils
import os
PATH = os.path
from utilsWaveform import *
import matplotlib.pyplot as plt
from PIL import Image


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

	# Generate the actual time-frequency images (as jpg files) from
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
		BBH_DIR = PATH.dirname(PATH.abspath(__file__))
		# Gonghan Xu: This path seeking procedure is specific only to my
		# own machine. It needs to be changed if this script is run
		# on some other machine.
		waveDirPath = PATH.join(BBH_DIR, "..", "lvcnr-lfs", "GeorgiaTech")
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


if __name__ == "__main__":
	trainSet = np.load("trainSet.npy")
	imMaker = TfMaker()
	DataFactory.genIm(trainSet, "trainTiffDir/", imMaker, genMenu=True,
					  menuName="trainTiffMenu.txt")
	pass
