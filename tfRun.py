# This module contains scripts for running common use cases for the
# time-frequency map project. The scripts are encapsulated in functions.
from TfInstance import TfInstance
from DataFactory import *
import numpy as np
from TfMaker import TfMaker
from Classifier import *
import copy
from skimage import filters


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


# This function uses cross validation to compute probabilities
# resulting from learning a SVM.
# @param dat (array<TfInstance>): Training data used for
# 	cross-validation.
def getCrossValProbs(dat, nfolds):
	# clf = Classifier(svm.SVR(kernel="linear", gamma="auto"))

	clf = Classifier(
		svm.SVC(kernel="linear", gamma="auto", probability=True,
				max_iter=10000))

	# Loading data
	print "Loading training data.."
	clf.imSet, clf.labelSet = DataFactory.getTrainableArrays(dat)

	# # Checking HOG feature vector.
	# print "Checking HOG features.."
	# clf.checkHOGFeat()
	# print

	# Extracting features
	ncomps = 50
	print "Extracting features from training data.."
	startExtractTime = time.time()
	percentVarCovered = clf.extractFeatsPCA(ncomps)
	# clf.extractHOGFeats()
	endExtractTime = time.time()
	extractTime = endExtractTime - startExtractTime
	print "Original Image Size:", clf.imSet[0].shape
	print "Number of selected principal components:", ncomps
	print "Percentage of variance covered:", percentVarCovered
	print "Training data feature extraction time:", extractTime, "sec"
	print

	# Preface for cross-validation
	assert len(clf.featSet) == len(clf.labelSet) >= nfolds
	numIns = len(clf.featSet)
	# Obtain a list of indices that will be randomly permuted
	# to facilitate the cross-validation process.
	shuffIndices = range(numIns)
	np.random.shuffle(shuffIndices)
	shuffFeats = clf.featSet[shuffIndices]
	shuffLabels = clf.labelSet[shuffIndices]

	foldSize = numIns // nfolds
	# foldStarts stores a list of starting indices for each fold.
	# In addition to these indices, the last element of foldStarts
	# will be numIns.
	foldStarts = []
	for i in range(nfolds):
		foldStarts.append(i * foldSize)
	foldStarts.append(numIns)
	print "foldStarts:", foldStarts

	# probs will ultimately become an array that will store the
	# probability of each instance being predicted to have double chirp.
	probs = []
	accus = []
	fails = [False] * numIns
	confMat = np.array([[0, 0], [0, 0]])

	# Now do cross-validation.
	for i in range(nfolds):
		print "Validating fold", i + 1, ".. "
		restFeats = Classifier.restSet(shuffFeats,
									   foldStarts[i], foldStarts[i + 1])
		restLabels = Classifier.restSet(shuffLabels,
										foldStarts[i], foldStarts[i + 1])
		assert len(restFeats) == len(restLabels)

		valFeats = shuffFeats[foldStarts[i]: foldStarts[i + 1]]
		valLabels = shuffLabels[foldStarts[i]: foldStarts[i + 1]]
		assert len(valFeats) == len(valLabels)
		assert len(restFeats) + len(valFeats) == numIns

		clf.model.fit(restFeats, restLabels)
		valProbs = clf.model.predict_proba(valFeats)
		assert valProbs.shape == (foldStarts[i + 1] - foldStarts[i], 2)
		# valPredicts = clf.model.predict(valFeats)
		numAccurates = 0

		# Building confusion matrix
		for j, prob in enumerate(valProbs, start=foldStarts[i]):
			assert abs(prob[0] + prob[1] - 1) < 0.00001
			if dat[shuffIndices[j]].hasDoubleChirp:
				if prob[0] > 0.5:
					numAccurates += 1
					confMat[0, 0] += 1
				else:
					fails[j] = True
					confMat[0, 1] += 1
			else:
				if prob[0] <= 0.5:
					numAccurates += 1
					confMat[1, 1] += 1
				else:
					fails[j] = True
					confMat[1, 0] += 1

		# for j, predict in enumerate(valPredicts, start=foldStarts[i]):
		# 	if dat[shuffIndices[j]].hasDoubleChirp:
		# 		numAccurates += 1 if predict == "Double Chirp" else 0
		# 	else:
		# 		numAccurates += 1 if predict == "Not Double Chirp" else 0

		accus.append(1.0 * numAccurates / len(valFeats))
		print "fold accuracy:", accus[-1]
		print
		probs = np.concatenate((probs, valProbs[:, 0]))

	print "Cross-Validation accuracies:", accus
	print "Total number of fails:", sum(fails)
	print "Classification Accuracy", 1 - 1.0 * sum(fails) / numIns
	# assert probs.shape == (numIns,)
	print "Confusion Matrix"
	printConfMat(confMat, ["DoubleChirp", "NotDoubleChirp"])
	return shuffIndices, probs, accus, fails, confMat


# Add Gaussian noise to a time-frequency map.
# @param im (array<*,*>)
def addNoise(im, mean, std):
	noise = np.random.normal(mean, std, im.shape)
	return im + noise


# @param ins (TfInstance)
def viewNoise(ins, mean, std, verbose=True, sigma=None):
	if verbose:
		print "Noise mean: {}; noise std: {}".format(mean, std)
		print "Original image min:", ins.ampArr.min()
		print "Original image max:", ins.ampArr.max()
		print "Original image mean:", ins.ampArr.mean()
		print "Original image std", ins.ampArr.std()
	insCopy = copy.deepcopy(ins)
	insCopy.ampArr = addNoise(insCopy.ampArr, mean, std)
	fig1, ax1 = ins.getPlot()
	fig2, ax2 = insCopy.getPlot()
	ax2.set_title(ax2.get_title() + ", noise std: {:.2f}".format(std))

	if sigma is not None:
		insSmooth = copy.deepcopy(insCopy)
		insSmooth.ampArr = filters.gaussian(insSmooth.ampArr, sigma=sigma)
		fig3, ax3 = insSmooth.getPlot()
		ax3.set_title(ax2.get_title() + "\n smooth std {:.2f}"
					  .format(std, sigma))
	plt.show()


# Train and test classifier with data added with Gaussian noise.
def trainNoise(datFile="heavyTrainSet_noDS.npy", std=0.2, sigma=None):
	print "+" * 70
	print "std = ", std
	nfolds = 5
	clf = Classifier(svm.SVC(kernel="linear", gamma="auto",
							 max_iter=10000))

	# Loading data
	print "Loading training data.."
	dat = np.load(datFile)
	clf.imSet, clf.labelSet = DataFactory.getTrainableArrays(dat)

	# Addiing noise
	mean = 0
	# std = std
	for i, im in enumerate(clf.imSet):
		clf.imSet[i] = addNoise(im, mean, std)

	if sigma is not None:
		# Smooth images
		clf.smoothImages(sigma=sigma)

	# # Checking HOG feature vector.
	# print "Checking HOG features.."
	# clf.checkHOGFeat()
	# print

	# Extracting features
	ncomps = 50
	print "Extracting features from training data.."
	startExtractTime = time.time()
	percentVarCovered = clf.extractFeatsPCA(ncomps)
	# clf.extractHOGFeats(sigma=0.6)
	endExtractTime = time.time()
	extractTime = endExtractTime - startExtractTime
	print "Original Image Size:", clf.imSet[0].shape
	print "Number of selected principal components:", ncomps
	print "Percentage of variance covered:", percentVarCovered
	print "Training data feature extraction time:", extractTime, "sec"
	print

	# Preface for cross-validation
	assert len(clf.featSet) == len(clf.labelSet) >= nfolds
	numIns = len(clf.featSet)
	# Obtain a list of indices that will be randomly permuted
	# to facilitate the cross-validation process.
	shuffIndices = range(numIns)
	np.random.shuffle(shuffIndices)
	shuffFeats = clf.featSet[shuffIndices]
	shuffLabels = clf.labelSet[shuffIndices]

	foldSize = numIns // nfolds
	# foldStarts stores a list of starting indices for each fold.
	# In addition to these indices, the last element of foldStarts
	# will be numIns.
	foldStarts = []
	for i in range(nfolds):
		foldStarts.append(i * foldSize)
	foldStarts.append(numIns)
	print "foldStarts:", foldStarts

	# Will store validation accuracy for each fold.
	accus = []
	# The elements of fails have a one-to-one correspondence to those
	# of the "indices" of shuffIndices.
	fails = [False] * numIns
	confMat = np.array([[0, 0], [0, 0]])

	# Now do cross-validation.
	for i in range(nfolds):
		print "Validating fold", i + 1, ".. "
		restFeats = Classifier.restSet(shuffFeats,
									   foldStarts[i], foldStarts[i + 1])
		restLabels = Classifier.restSet(shuffLabels,
										foldStarts[i], foldStarts[i + 1])
		assert len(restFeats) == len(restLabels)

		valFeats = shuffFeats[foldStarts[i]: foldStarts[i + 1]]
		valLabels = shuffLabels[foldStarts[i]: foldStarts[i + 1]]
		assert len(valFeats) == len(valLabels)
		assert len(restFeats) + len(valFeats) == numIns

		# Core
		clf.model.fit(restFeats, restLabels)
		valPredicts = clf.model.predict(valFeats)
		assert valPredicts.shape == (foldStarts[i + 1] - foldStarts[i],)

		# In-fold accuracy
		numAccurates = 0
		# Building confusion matrix
		for j, predict in enumerate(valPredicts, start=foldStarts[i]):
			if dat[shuffIndices[j]].hasDoubleChirp:
				if predict == "Double Chirp":
					confMat[0, 0] += 1
					numAccurates += 1
				else:
					fails[j] = True
					confMat[0, 1] += 1
			else:
				if predict == "Not Double Chirp":
					confMat[1, 1] += 1
					numAccurates += 1
				else:
					fails[j] = True
					confMat[1, 0] += 1

		accus.append(1.0 * numAccurates / len(valFeats))
		print "fold accuracy:", accus[-1]
		print

	print "Cross-Validation accuracies:", accus
	print "Mean Cross-Validation Accuracy:", sum(accus) / nfolds
	print "Total number of fails:", sum(fails)
	print "Classification Accuracy", 1 - 1.0 * sum(fails) / numIns
	print "Confusion Matrix"
	printConfMat(confMat, ["DoubleChirp", "NotDoubleChirp"])
	# return shuffIndices, accus, fails, confMat
	return sum(accus) / nfolds


# Train on pure data and validate on noisy data.
def valNoise(datFile="heavyTrainSet.npy", std=0.02, sigma=None):
	print "+" * 70
	print "std = ", std
	nfolds = 5
	clf = Classifier(svm.SVC(kernel="linear", gamma="auto",
							 max_iter=10000))

	# Loading data
	print "Loading training data.."
	dat = np.load(datFile)
	clf.imSet, clf.labelSet = DataFactory.getTrainableArrays(dat)

	# # Checking HOG feature vector.
	# print "Checking HOG features.."
	# clf.checkHOGFeat()
	# print

	# Extracting features
	ncomps = 50
	print "Extracting features from training data.."
	startExtractTime = time.time()
	percentVarCovered = clf.extractFeatsPCA(ncomps)
	# clf.extractHOGFeats(sigma=0.6)
	endExtractTime = time.time()
	extractTime = endExtractTime - startExtractTime
	print "Original Image Size:", clf.imSet[0].shape
	print "Number of selected principal components:", ncomps
	print "Percentage of variance covered:", percentVarCovered
	print "Training data feature extraction time:", extractTime, "sec"
	print

	# Preface for cross-validation
	assert len(clf.featSet) == len(clf.labelSet) >= nfolds
	numIns = len(clf.featSet)
	# Obtain a list of indices that will be randomly permuted
	# to facilitate the cross-validation process.
	shuffIndices = range(numIns)
	np.random.shuffle(shuffIndices)
	shuffFeats = clf.featSet[shuffIndices]
	shuffLabels = clf.labelSet[shuffIndices]


	tmp = Classifier(model=None)
	tmp.imSet = copy.deepcopy(clf.imSet)
	mean = 0
	# Adding noise.
	for i, im in enumerate(tmp.imSet):
		tmp.imSet[i] = addNoise(im, mean, std)
	# Smoothing
	if sigma is not None:
		# Smooth images
		tmp.smoothImages(sigma=sigma)
	print "Extracting features from validation images.."
	percentVarCovered = tmp.extractFeatsPCA(ncomps)
	print "Percentage of variance covered:", percentVarCovered
	print
	noiseFeats = tmp.featSet[shuffIndices]

	idx = np.random.choice(numIns, replace=False)
	print "random idx:", idx
	plt.imshow(np.flip(tmp.imSet[idx], axis=0), cmap="gray")
	plt.title("noise std: {:.2f}".format(std))
	plt.show()


	foldSize = numIns // nfolds
	# foldStarts stores a list of starting indices for each fold.
	# In addition to these indices, the last element of foldStarts
	# will be numIns.
	foldStarts = []
	for i in range(nfolds):
		foldStarts.append(i * foldSize)
	foldStarts.append(numIns)
	print "foldStarts:", foldStarts

	# Will store validation accuracy for each fold.
	accus = []
	# The elements of fails have a one-to-one correspondence to those
	# of the "indices" of shuffIndices.
	fails = [False] * numIns
	confMat = np.array([[0, 0], [0, 0]])

	# Now do cross-validation.
	for i in range(nfolds):
		print "Validating fold", i + 1, ".. "
		restFeats = Classifier.restSet(shuffFeats,
									   foldStarts[i], foldStarts[i + 1])
		restLabels = Classifier.restSet(shuffLabels,
										foldStarts[i], foldStarts[i + 1])
		assert len(restFeats) == len(restLabels)

		valFeats = noiseFeats[foldStarts[i]: foldStarts[i + 1]]
		valLabels = shuffLabels[foldStarts[i]: foldStarts[i + 1]]
		assert len(valFeats) == len(valLabels)
		assert len(restFeats) + len(valFeats) == numIns

		# Core
		clf.model.fit(restFeats, restLabels)
		valPredicts = clf.model.predict(valFeats)
		assert valPredicts.shape == (foldStarts[i + 1] - foldStarts[i],)

		# In-fold accuracy
		numAccurates = 0
		# Building confusion matrix
		for j, predict in enumerate(valPredicts, start=foldStarts[i]):
			if dat[shuffIndices[j]].hasDoubleChirp:
				if predict == "Double Chirp":
					confMat[0, 0] += 1
					numAccurates += 1
				else:
					fails[j] = True
					confMat[0, 1] += 1
			else:
				if predict == "Not Double Chirp":
					confMat[1, 1] += 1
					numAccurates += 1
				else:
					fails[j] = True
					confMat[1, 0] += 1

		accus.append(1.0 * numAccurates / len(valFeats))
		print "fold accuracy:", accus[-1]
		print

	print "Cross-Validation accuracies:", accus
	print "Mean Cross-Validation Accuracy:", sum(accus) / nfolds
	print "Total number of fails:", sum(fails)
	print "Classification Accuracy", 1 - 1.0 * sum(fails) / numIns
	print "Confusion Matrix"
	printConfMat(confMat, ["DoubleChirp", "NotDoubleChirp"])
	# return shuffIndices, accus, fails, confMat
	return sum(accus) / nfolds


# Plot the noise training results
def plotNoise():
	stds = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
	accus = []
	sigma = 2
	for std in stds:
		# accus.append(trainNoise("heavyTrainSet_noDS.npy",
		# 						std=std, sigma=sigma))
		accus.append(valNoise("heavyTrainSet_noDS.npy",
							  std=std, sigma=sigma))
	plt.plot(stds, accus, marker="o")
	plt.ylim(ymax=1)
	for x, y in zip(stds, accus):
		plt.annotate("{:.2f}".format(y), (x, y))
	plt.title("Training on pure and validating on Gaussian-noisy data"
			  "\nNo downsampling, smoothing sigma = {:.1f}".format(sigma))
	plt.xlabel("noise standard deviation")
	plt.ylabel("average validation accuracy")
	plt.show()


if __name__ == "__main__":
	# renewTrainSet()
	# saveHeavyTrainingSet()
	# saveHpTrainDat()
	# trainNoise()
	plotNoise()
	# valNoise()
	pass

