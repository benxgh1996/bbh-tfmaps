from TfInstance import *
import random as rand
import utils
import os
PATH = os.path
from utilsWaveform import *
import matplotlib.pyplot as plt


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
	# training the nerual net.
	# @param annoData (str): the path to the npy file that holds the
	# 	annotated data.
	# @param savePath (str): the path to the directory that will hold
	#	the generated image files.
	# @param genMenu (bool): a flag that marks whether we want to generate
	# 	a menu (txt) file that lists each generated image and its label.
	#	The label will be 1 for double chirp and 0 for no double chirp.
	# @param menuName (str): The name (e.g. train.txt) for the menu file.
	@staticmethod
	def genIm(annoPath, savePath, tfMaker, genMenu=True,
			  menuName=None):
		tfData = np.load(annoPath)
		if not genMenu:
			if menuName is not None:
				raise Exception("An unnecessary menuName is supplied "
								"with no intent of generating menu file.")
		else:
			if menuName is None:
				raise Exception("menuName not supplied.")
			elif PATH.isfile(menuName):
				raise Exception("menuName already exists as a file.")
			else:
				f = open(menuName)


		BBH_DIR = PATH.dirname(PATH.abspath(__file__))
		waveDirPath = PATH.join(BBH_DIR, "..", "lvcnr-lfs", "GeorgiaTech")

		for tfInstance in tfData:
			im = tfMaker.tfInstance2Im(tfInstance, waveDirPath)
			plt.imshow(im)
			imName = DataFactory.getImFileName(tfInstance)
			# TODO: Need to check whether this will save the image to
			# the correct directory.
			plt.savefig(PATH.join(savePath, imName))

	# Get the appropriate file name representing a time-frequency image.
	@staticmethod
	def getImFileName(tfInstance):
		x = tfInstance
		imName = "{}_{:d}_{:d}_{:d}.jpg".\
			format(x.waveformName, x.iota*100, x.phi*100, x.motherFreq*10)
		return imName


# A class that is used to generate time-frequency images at given
# configurations. The main configuration parameters will be related to
# to how we would like to downsample a time-frequency map.
class TfMaker(object):
	# @param midTime (float): The middle point of time with respect
	#	to the time window. leftTime and rightTime are measured with
	#	respect to this middle point of time.
	# @param numTimes (int): The number of time points to be selected
	# 	along the time-axis.
	# @param numFreqs (int): The number of frequency points to be selected
	# 	along the frequency axis.
	# @param leftTimeWindow (non-positive float): The left time window,
	# 	measured as between the ideal starting time and the middle time.
	# @param rightTimeWindow (non-negative float): The right time window,
	# 	measured as between the ideal ending time and the middle time.
	# @param freqWindow (int): The total number of interesting frequency
	#	points.
	# 	The entire frequency window will be (0, 0 + freq_window - 1).
	# 	The default frequency window may be carefully selected to be 505
	# 	because 505=63*8+1. Therefore, 505 is able to give 64 selected
	# 	points exactly.
	def __init__(self, midTime, numTimes, numFreqs, leftTimeWindow,
				 rightTimeWindow, freqWindow):
		self.midTime = midTime
		self.numTimes = numTimes
		self.numFreqs = numFreqs
		self.leftTimeWindow = leftTimeWindow
		self.rightTimeWindow = rightTimeWindow
		self.freqWindow = freqWindow

	# Generate an image from a TfInstance.
	# @param tfInstance (TfInstance): An TfInstance from which an image
	#	will be generated.
	# @param waveformPath (str): The path to the directory where
	#	the HDF5 waveform files are stored.
	# @return im (ndarray<*, *>): A 2-D array representing the image
	#	of a time-frequency map. This array is already flipped along
	#	the frequency axis so that imshow will be able to show the image
	#	in normal looking.
	def tfInstance2Im(self, tfInstance, waveDirPath):
		waveformName = tfInstance.waveformName
		waveformPath = PATH.join(waveDirPath, waveformName)
		im, _, _ = self.getTfIm(waveformPath, tfInstance.iota,
								tfInstance.phi, tfInstance.motherFreq)
		# Flipping the frequency axis to make the image look normal
		# using imshow.
		im = np.flip(im, axis=0)
		return im


	# Generate a time-frequency image.
	def getTfIm(self, wavePath, iota, phi, motherFreq):
		MAX_SCALE = 512
		wfData = gen_waveform(wavePath, iota, phi)
		tfData = tf_decompose(wfData['hp'], wfData["sample_times"],
							  motherFreq, MAX_SCALE)
		wplane = tfData["wplane"]
		wfreqs = tfData["wfreqs"]
		sampleTimes = wfData["sample_times"]
		sampledWplane, sampledFreqs, sampledTimes = \
			utils.select_wplane(wplane, wfreqs, sampleTimes,
								mid_t=self.midTime, xnum=self.numFreqs,
								ynum=self.numTimes,
								left_t_window=self.leftTimeWindow,
								right_t_window=self.rightTimeWindow,
								freq_window=self.freqWindow)

		assert isinstance(sampledWplane, np.ndarray)
		assert isinstance(sampledTimes, np.ndarray)
		assert isinstance(sampledFreqs, np.ndarray)
		assert sampledWplane.ndim == 2
		assert sampledWplane.shape[0] == len(sampledFreqs) == self.numFreqs
		assert sampledWplane.shape[1] == len(sampledTimes) == self.numTimes

		return sampledWplane, sampledTimes, sampledFreqs


if __name__ == "__main__":
	# tfData = np.load("tfInstances.npy")
	# tfFactory = DataFactory(tfData)
	# trainArr, testArr = tfFactory.splitData(6, 2)
	# assert len(trainArr) == 600
	# assert len(testArr) == 200
	# np.save("trainSet.npy", trainArr)
	# np.save("testSet.npy", testArr)
	pass
