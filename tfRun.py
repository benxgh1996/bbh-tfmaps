# This module contains scripts for running common use cases for the
# time-frequency map project. The scripts are encapsulated in functions.
from TfInstance import TfInstance
from DataFactory import *
import numpy as np
from TfMaker import TfMaker


# Saving a light-weight pickled training set into as a heavy
# training set.
def saveHeavyTrainingSet():
	lightTrainSet = np.load("newTrainSet.npy")
	# No Downsampling
	tfMaker = TfMaker(numTimes=None, numFreqs=None, freqWindow=None,
					  midTime=0.0, leftTimeWindow=-0.05,
					  rightTimeWindow=0.03, downSample=False)
	DataFactory.saveTrainableData(lightTrainSet, tfMaker,
								  "heavyTrainSet_noDS.npy")

# Converting a legacy light-weight pickled training set into
# an updated light-weight training set, and then save it.
def renewTrainSet():
	newSet = []
	oldSet = np.load("trainSet.npy")
	for oldIns in oldSet:
		newSet.append(TfInstance.factory(oldIns))
	np.save("newTrainSet.npy", newSet)


# Create a new training set with filled hp attributes in addition
# to the original light-weight training set.
def saveHpTrainDat():
	trainSet = np.load("newTrainSet.npy")
	tfMaker = TfMaker(leftTimeWindow=-0.05, rightTimeWindow=0.03)
	waveDirPath = DataFactory.getWaveDirPath()
	for ins in trainSet:
		waveformName = ins.waveformName + ".h5"
		waveformPath = PATH.join(waveDirPath, waveformName)
		ins.hp = tfMaker.getHp(waveformPath, ins.iota, ins.phi)
	np.save("hpTrainSet.npy", trainSet)


if __name__ == "__main__":
	# renewTrainSet()
	# saveHeavyTrainingSet()
	saveHpTrainDat()
	pass