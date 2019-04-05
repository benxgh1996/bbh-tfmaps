import numpy as np
import matplotlib.pyplot as plt
from TfInstance import TfInstance
from DataFactory import *
from TfMaker import *
from skimage import io
from PIL import Image
import copy
from sklearn import datasets
from Classifier import *
from tfRun import *

MOTHER_FREQ = 0.5
MAX_SCALE = 512
FIG_WIDTH, FIG_HEIGHT = 7.6, 5.8
pi = np.pi
# Default figure size
# FIG_WIDTH, FIG_HEIGHT = 6.4, 4.8


def testFileSave():
	f = open("testDir/testFile.txt", "w")
	f.write("Hello ")
	f.write("world!\n")
	f.write("Testing file saving.")
	f.close()


def testFigSave():
	im = np.arange(24).reshape(6, 4)
	plt.imshow(im)
	# plt.show()
	# plt.savefig("testDir/image1.jpg")
	plt.savefig("/Users/benxgh1996/desktop/"
				"rsch/bbh-tfmaps/testDir/im2.jpg")


def testImMargin():
	data = np.random.random((5, 5))
	fig = plt.imshow(data, interpolation="nearest")
	plt.axis('off')
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	plt.savefig("testMargin.png", bbox_inches="tight", pad_inches=0)


def testImGen():
	BBH_DIR = PATH.dirname(PATH.abspath(__file__))
	waveDirPath = PATH.join(BBH_DIR, "..", "lvcnr-lfs", "GeorgiaTech")
	wavePath = PATH.join(waveDirPath, "GT0577.h5")
	tfMaker = TfMaker()
	im, _, _ = tfMaker.getTfIm(wavePath, 40*pi/180, 280*pi/180, 0.5)
	im = np.flip(im, axis=0)

	# trainSet = np.load("trainSet.npy")
	# tfInstance = trainSet[15]
	# im = tfMaker.tfInstance2Im(tfInstance, waveDirPath)
	print "Original min:", im.min()
	print "Original max:", im.max()
	print "Original mean:", im.mean()
	# print im.shape
	# plt.imshow(im, cmap="gray")
	# plt.axis("off")
	# plt.show()

	newIm = Image.fromarray(im)
	# newIm = newIm.convert("L")
	# print "type:", type(newIm)
	newIm.save("testfig.tiff")
	# plt.imsave("testfig.png", cmap="gray")


def testImSize():
	im = plt.imread("testfig.tiff")
	print im.shape
	print "min:", im.min()
	print "max:", im.max()
	print "ave", im.mean()

def testCopy():
	tf1 = TfInstance("GT0577", 0.1, 0.2, 0.3)
	tf2 = TfInstance("GT0448", 0.4, 0.5, 0.6)
	l1 = np.array([tf1, tf2])
	l2 = copy.deepcopy(l1)
	print type(l2)
	print l2.dtype
	print type(l2[0])
	l2[0].iota = 0.8
	# assert l1[0] == l2[0]
	# assert l1[1] == l2[1]
	print l1
	print l2


def testEnum():
	tf1 = TfInstance("GT0577", 0.1, 0.2, 0.3)
	tf2 = TfInstance("GT0448", 0.4, 0.5, 0.6)
	l1 = np.array([tf1, tf2])
	for c, tf in enumerate(l1, start=10):
		print c, ": ", tf


def testTfIns():
	tfArr = np.load("trainSet.npy")
	print hasattr(tfArr[0], "timeArr")
	newIns = TfInstance.factory(tfArr[0])
	print hasattr(newIns, "timeArr")
	print newIns
	# print TfInstance(tfArr).timeArr is not None
	# print tfArr[0].isLight()


def testHeavyImSize():
	dat = np.load("heavyTrainSet.npy")
	for ins in dat:
		assert ins.ampArr.shape == (64, 64)


def vizIm():
	imSet = np.load("heavyTrainSet_noDS_play.npy")
	ins = imSet[1]
	print "Shape of image:", ins.ampArr.shape
	print "Size of freqs:", ins.freqArr.shape
	print "Size of times:", ins.timeArr.shape
	print "Time limits:", (ins.timeArr[0], ins.timeArr[-1])
	print "Freq limits:", (ins.freqArr[0], ins.freqArr[-1])
	print "Has double chirp:", ins.hasDoubleChirp
	plt.imshow(np.flip(ins.ampArr, axis=0), cmap="gray")
	plt.show()

def testHp():
	BBH_DIR = PATH.dirname(PATH.abspath(__file__))
	waveDirPath = PATH.join(BBH_DIR, "..", "lvcnr-lfs", "GeorgiaTech")
	wavePath = PATH.join(waveDirPath, "GT0577.h5")
	wfData = gen_waveform(wavePath, 0, 0)
	hp = wfData["hp"]
	print type(hp)
	hp = np.array(hp)
	print type(hp)
	print hp.shape
	print hp


def testHpSize():
	dat = np.load("hpTrainSet.npy")
	print dat[0].hp.dtype
	print dat[0].hp.shape


def testIris():
	iris = datasets.load_iris()
	feats = iris.data
	labels = iris.target
	print feats.shape
	print labels.shape
	print labels
	print feats

def testConfMat():
	mat = [[23, 49, 2132], [811, 26, 98435], [0, 1, 23]]
	attNames = ["cat", "a", "flowers"]
	printConfMat(mat, attNames)

def testNdsSet():
	dat = np.load("heavyTrainSet_noDS.npy")
	stdShape = dat[0].ampArr.shape
	print "Standard shape:", stdShape
	for ins in dat:
		assert ins.hasDoubleChirp is True or ins.hasDoubleChirp is False
		assert ins.ampArr.shape == stdShape
		assert (ins.ampArr >= 0).all()
		assert (ins.ampArr <= 1).all()
		assert ins.ampArr.dtype == np.float64

def testPCAFit():
	dat = np.load("heavyTrainSet_noDS.npy")
	# dat = dat[: 600]

	# clf = Classifier(svm.SVR(kernel="linear", gamma="auto"))
	clf = Classifier(
		svm.SVC(kernel="linear", gamma="auto", probability=True))

	# Loading data
	print "Loading training data.."
	clf.imSet, clf.labelSet = DataFactory.getTrainableArrays(dat)

	# Extracting features
	ncomps = 30
	print "Extracting features from training data.."
	startExtractTime = time.time()
	percentVarCovered = clf.extractFeatsPCA(ncomps)
	endExtractTime = time.time()
	extractTime = endExtractTime - startExtractTime
	print "Original Image Size:", clf.imSet[0].shape
	print "Number of selected principal components:", ncomps
	print "Percentage of variance covered:", percentVarCovered
	print "Training data feature extraction time:", extractTime, "sec"
	print

	numIns = len(clf.featSet)
	shuffIndices = range(numIns)
	# np.random.shuffle(shuffIndices)
	shuffFeats = clf.featSet[shuffIndices]
	shuffLabels = clf.labelSet[shuffIndices]
	confMat = np.array([[0, 0], [0, 0]])

	print "Start training.."
	clf.model.fit(shuffFeats, shuffLabels)
	print "Start predicting.."
	probs = clf.model.predict_proba(shuffFeats)
	assert probs.shape == (numIns, 2)

	for i, prob in enumerate(probs):
		if dat[shuffIndices[i]].hasDoubleChirp:
			if prob[0] > 0.5:
				confMat[0, 0] += 1
			else:
				confMat[0, 1] += 1
		else:
			if prob[0] <= 0.5:
				confMat[1, 1] += 1
			else:
				confMat[1, 0] += 1

	print "Training accuracy:", 1.0 * (confMat[0, 0] + confMat[1, 1]) / numIns
	print "Total number of fails:", confMat[0, 1] + confMat[1, 0]
	print "Confusion Matrix"
	printConfMat(confMat, ["DoubleChirp", "NotDoubleChirp"])


def testNoise():
	double = True
	dat = np.load("heavyTrainSet_noDS.npy")
	numIns = len(dat)
	# nsamples = 5
	mean = 0
	# std = 0.2
	stds = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
	if double:
		idx = np.random.choice(numIns, replace=False)
		while not dat[idx].hasDoubleChirp:
			idx = np.random.choice(numIns, replace=False)
	print "Random idx:", idx
	print "Random waveform:", dat[idx]

	for std in stds:
		print "=" * 50
		viewNoise(dat[idx], mean, std)


def testPlot():
	x = [1, 3, 4, 6, 9]
	y = [0.23, 0.5, 0.6, 0.1, 0.91]
	plt.plot(x, y, marker="o")
	plt.ylim(ymax=1)
	for x, y in zip(x, y):
		plt.annotate("{:.2f}".format(y), (x-0.05, y+0.05))
	plt.show()


if __name__ == "__main__":
	# testImGen()
	# testHeavyImSize()
	# testImMargin()
	# testCopy()
	# testEnum()
	# testTfIns()
	# testHeavyImSize()
	# vizIm()
	# testHp()
	# testHpSize()
	# testIris()
	# testConfMat()
	# testNdsSet()
	# testPCAFit()
	testNoise()
	# testPlot()
	pass




