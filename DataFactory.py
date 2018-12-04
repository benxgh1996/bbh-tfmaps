from TfInstance import *
import random as rand


class DataFactory(object):
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


if __name__ == "__main__":
	# tfData = np.load("tfInstances.npy")
	# tfFactory = DataFactory(tfData)
	# trainArr, testArr = tfFactory.splitData(6, 2)
	# assert len(trainArr) == 600
	# assert len(testArr) == 200
	# np.save("trainSet.npy", trainArr)
	# np.save("testSet.npy", testArr)
	pass
