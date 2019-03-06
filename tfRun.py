# This module contains scripts for running common use cases for the
# time-frequency map project. The scripts are encapsulated in functions.
from TfInstance import TfInstance
from DataFactory import DataFactory
import numpy as np
from TfMaker import TfMaker


# Saving a light-weight pickled training set into as a heavy
# training set.
def saveHeavyTrainingSet():
	lightTrainSet = np.load("newTrainSet.npy")
	tfMaker = TfMaker()
	DataFactory.saveTrainableData(lightTrainSet, tfMaker,
								  "heavyTrainSet.npy")

# Converting a legacy light-weight pickled training set into
# an updated light-weight training set, and then save it.
def renewTrainSet():
	newSet = []
	oldSet = np.load("trainSet.npy")
	for oldIns in oldSet:
		newSet.append(TfInstance.factory(oldIns))
	np.save("newTrainSet.npy", newSet)

if __name__ == "__main__":
	# renewTrainSet()
	# saveHeavyTrainingSet()
	pass