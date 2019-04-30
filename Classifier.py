import numpy as np
from sklearn import svm, metrics
from skimage import feature, filters
from sklearn.model_selection import cross_validate
import time
from DataFactory import DataFactory
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Classifier:
    def __init__(self, model=svm.SVC(kernel="linear", gamma="auto"),
                 featSet=None, labelSet=None, imSet=None):
        self.model = model
        self.featSet = featSet
        self.labelSet = labelSet
        self.imSet = imSet

    # Loads the labelled data as input images and input labels.
    # @param fileName (str)
    def loadData(self, fileName):
        data = np.load(fileName)
        self.imSet, self.labelSet = \
            DataFactory.getTrainableArrays(data)
        self.labelSet = np.array(self.labelSet)
        assert isinstance(self.imSet[0, 0, 0], np.floating)
        assert (self.imSet >= 0).all() and (self.imSet <= 1).all()

    # Load the model with features and labels directly.
    # @param feats (array<*>)
    # @param labels (array<*>)
    def loadInput(self, feats, labels):
        assert len(feats) == len(labels)
        assert feats.ndim == 2
        assert labels.ndim == 1
        self.featSet = feats
        self.labelSet = labels

    # Convert an hp-enabled training set into a feature set and a label set.
    # hp means plus polarization.
    def loadHpDat(self, fileName):
        trainSet = np.load(fileName)
        feats = []
        labels = []
        for ins in trainSet:
            feats.append(ins.hp)
            if ins.hasDoubleChirp:
                labels.append("Double Chirp")
            else:
                labels.append("Not Double Chirp")
        self.loadInput(np.array(feats), np.array(labels))

    def smoothImages(self, sigma=0.6):
        smoothSet = []
        for im in self.imSet:
            smoothIm = filters.gaussian(im, sigma=sigma)
            smoothSet.append(smoothIm)
        # Rewrites self.imSet.
        self.imSet = np.array(smoothSet)
        assert self.imSet.ndim == 3
        assert isinstance(self.imSet[0, 0, 0], np.floating)

    # Extract features from image using PCA.
    # @return (floating): Total percentage of variances covered.
    def extractFeatsPCA(self, numComps):
        pca = PCA(n_components=numComps)
        tmp = self.imSet.reshape(len(self.imSet), -1)
        self.featSet = pca.fit_transform(tmp)
        assert self.featSet.shape == (len(self.imSet), numComps)
        return sum(pca.explained_variance_ratio_[0: numComps])

    # This function extracts the features from the image set. By default
    #   we will not apply the Gaussian filter.
    # @param sigma (float): If None, do not apply the Gaussian filter
    #   to smooth the images. May want to try sigma = 0.6.
    def extractHOGFeats(self, flatten=False, sigma=None):
        # print "Number of instances:", len(self.imSet)
        numImages = len(self.imSet)
        if sigma is not None:
            # Smooth the image set.
            smoothSet = []
            for im in self.imSet:
                smoothIm = filters.gaussian(im, sigma=sigma)
                smoothSet.append(smoothIm)
            # Rewrites self.imSet.
            self.imSet = np.array(smoothSet)
            assert self.imSet.ndim == 3
            assert isinstance(self.imSet[0, 0, 0], np.floating)
            # assert (self.imSet >= 0).all() and (self.imSet <= 1).all()

        # print "Extracting features.."
        if flatten:
            self.featSet = self.imSet.reshape(len(self.imSet), -1)
            assert len(self.featSet) == len(self.imSet)
            assert np.array_equal(self.featSet[0], self.imSet[0].flatten())
        else:
            # A collection that holds the feature vectors.
            self.featSet = []
            # orient, cells_p, pixels_p, vec_len
            # With Downsampling (64 * 64)
            # 8, (4, 4), (4, 4), 21632
            # 8, (4, 4), (8, 8), 3200
            # 8, (2, 2), (8, 8), 1568
            # 8, (2, 2), (4, 4), 7200
            # No Downsampling (508 * 328)
            # 8, (2, 2), (64, 40), 1344
            for idx, im in enumerate(self.imSet):
                if (idx + 1) % 50 == 0:
                    print "Extracting feature from {}st/{} image"\
                        .format(idx+1, numImages)
                feat = feature.hog(im, orientations=8,
                                   cells_per_block=(2, 2),
                                   pixels_per_cell=(64, 40),
                                   block_norm="L2-Hys")
                self.featSet.append(feat)
            print
            self.featSet = np.array(self.featSet)
        assert self.featSet.ndim == 2

    # Checks the properties of the extracted image feature.
    def checkHOGFeat(self):
        im = self.imSet[0]
        print "Shape of image:", im.shape
        feat = feature.hog(im, orientations=8,
                           cells_per_block=(2, 2),
                           pixels_per_cell=(64, 40),
                           block_norm="L2-Hys")
        print("Feature shape:", feat.shape)
        print("Feature type:", feat.dtype)
        print("Max feat value:", feat.max())
        print("min feat value:", feat.min())
        # plt.imshow(im, cmap="gray")
        # plt.show()
        # plt.imshow(hogIm, cmap="gray")
        # plt.show()

    # Resets the model.
    def setClassifier(self, classifier):
        self.model = classifier

    # Trains the model.
    def train(self):
        assert self.featSet.ndim == 2
        assert self.labelSet.ndim == 1
        assert len(self.featSet) == len(self.labelSet)
        print("Training model..")
        print("Length of feature set:", len(self.featSet))
        self.model.fit(self.featSet, self.labelSet)

    # Predicts the labels of a collection of images.
    # @param data (array<*, *>): A list of features representing images
    #   to be classified.
    # @return array: An array of predicted labels.
    def predictLabels(self, data):
        return self.model.predict(data)

    # This functions performs k-fold cross-validation on the training data.
    # @param trainArr (array<array>): The training features.
    # @param labelArr (array<str-like>): The training labels.
    # @param k (int): The number of folds for k-fold cross-validation.
    def crossVal(self, k=5):
        assert self.featSet.ndim == 2
        assert self.labelSet.ndim == 1
        assert len(self.featSet) == len(self.labelSet)
        assert isinstance(self.labelSet[0], str)
        scores = cross_validate(self.model, self.featSet,
                                self.labelSet, cv=k, return_train_score=True)
        trainAccus = scores["train_score"]
        testAccus = scores["test_score"]
        assert trainAccus.shape == (k, )
        print "On training set"
        print "Accuracies:", trainAccus
        print "Accuracy means:", trainAccus.mean()
        print "Accuracy Std:", trainAccus.std()
        print
        print "On validation set"
        print "Accuracies:", testAccus
        print "Accuracy means:", testAccus.mean()
        print "Accuracy Std:", testAccus.std()

    # Testing the effect of applying an image filter.
    def testFilter(self, sigma=1):
        im = self.imSet[0]
        assert im.ndim == 2
        assert isinstance(im[0, 0], np.floating)
        assert im.max() <= 1
        assert im.min() >= 0

        newIm = filters.gaussian(im, sigma=sigma)
        assert isinstance(newIm[0, 0], np.floating)
        assert newIm.max() <= 1
        assert newIm.min() >= 0
        plt.imshow(np.flip(im, axis=0), cmap="gray")
        plt.title("Original Image")
        plt.show()
        plt.imshow(np.flip(newIm, axis=0), cmap="gray")
        plt.title("New Image, sigma = {}".format(sigma))
        plt.show()

    # Obtain the rest (i.e. remaining) data set excluding the
    # validation set
    # @param arr (array): The original training set from which
    #   a validation set is to be extracted.
    # @param start (int): the starting idx of the validation set.
    # @param end (int): the ending idx + 1 of the validation set, which
    #   can be the length of arr.
    @staticmethod
    def restSet(arr, start, end):
        assert 0 <= start < end <= len(arr)
        return np.concatenate((arr[0: start], arr[end:]))


# def main():
#     model = Classifier()
#
#     # load images
#     model.loadData("newTrainSet.npy")
#     (test_raw, test_labels) = model.loadData('./test/')
#
#     # convert images into features
#     # Notice that an entire array of images are passed into the feature
#     # extraction function.
#     # train_data, test_data are both arrays of features that represent
#     # the collection of images.
#     train_data = model.extractHOGFeats(train_raw)
#     test_data = model.extractHOGFeats(test_raw)
#
#     # train model and test on training data
#     model.train(train_data, train_labels)
#
#     # Now that we have trained the model.
#     # We want to predict the labels of the training data to obtain
#     # the training accuracy.
#     predicted_labels = model.predictLabels(train_data)
#     print("\nTraining results")
#     print("=============================")
#     print("Confusion Matrix:\n",
#           metrics.confusion_matrix(train_labels, predicted_labels))
#     print("Accuracy: ",
#           metrics.accuracy_score(train_labels, predicted_labels))
#     print("F1 score: ", metrics.f1_score(
#         train_labels, predicted_labels, average='micro'))
#
#     # test model
#     predicted_labels = model.predictLabels(test_data)
#     print("\nTest results")
#     print("=============================")
#     print("Confusion Matrix:\n",
#           metrics.confusion_matrix(test_labels, predicted_labels))
#     print("Accuracy: ",
#           metrics.accuracy_score(test_labels, predicted_labels))
#     print("F1 score: ",
#           metrics.f1_score(test_labels, predicted_labels, average='micro'))


# This function is used for personal testing.
# @param dat (array<TfInstance>)
def crossVal(k=5, nComps=None, dat=None):
    clf = Classifier()
    # Loading training data
    print "Loading training data.."
    startLoadTime = time.time()
    if dat is not None:
        clf.imSet, clf.labelSet = \
            DataFactory.getTrainableArrays(dat)
    else:
        clf.loadData("heavyTrainSet_DS_mass200.npy")
    # clf.loadData("heavyTrainSet_noDS.npy")
    # clf.loadHpDat("hpTrainSet.npy")
    endLoadTime = time.time()
    loadTime = endLoadTime - startLoadTime
    print "Training data load time:", loadTime, "sec"
    print

    # # Checking feature vector.
    # print "Checking HOG features.."
    # model.checkHOGFeat()
    # print
	#
    # Extracting features.
    print "Extracting features from training data.."
    startExtractTime = time.time()
    # clf.extractHOGFeats()
    # clf.extractHOGFeats(flatten=True)
    percentVarCovered = clf.extractFeatsPCA(nComps)
    print "Original Image Size:", clf.imSet[0].shape
    print "Number of selected principal components:", nComps
    print "Percentage of variance covered:", percentVarCovered
    endExtractTime = time.time()
    extractTime = endExtractTime - startExtractTime
    print "Training data feature extraction time:", extractTime, "sec"
    print

    # Cross-validation.
    print "Doing {} fold cross-validation..".format(k)
    print("-" * 50)
    startCrossTime = time.time()
    clf.crossVal(k=k)
    endCrossTime = time.time()
    crossTime = endCrossTime - startCrossTime
    print("Cross-validation time:", crossTime, "sec")


# Used for testing the filtering effect.
def checkFilter(sigma=1):
    clf = Classifier()
    clf.testFilter(sigma=sigma)


# Used for testing the HOG feature vector.
def checkFeat():
    clf = Classifier()
    clf.loadData("heavyTrainSet_noDS.npy")
    clf.checkHOGFeat()


if __name__ == "__main__":
    # main()
    crossVal(k=5, nComps=50)
    # checkHOGFeat()
