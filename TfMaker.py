from TfInstance import *
import utils
import os
PATH = os.path
from utilsWaveform import *
import matplotlib.pyplot as plt


# A class that is used to generate time-frequency images at given
# configurations. The main configuration parameters will be related to
# to how we would like to downsample a time-frequency map.
class TfMaker(object):
	# @param midTime (float): The middle point of time with respect
	#	to the time window. leftTime and rightTime are measured with
	#	respect to this middle point of time.
	# @param numTimes (int): The number of time points to be selected
	# 	along the time axis.
	# @param numFreqs (int): The number of frequency points to be selected
	# 	along the frequency axis.
	# @param leftTimeWindow (non-positive float): The left time window,
	# 	measured as between the ideal starting time and the middle time.
	# @param rightTimeWindow (non-negative float): The right time window,
	# 	measured as between the ideal ending time and the middle time.
	# @param freqWindow (int): The total number of original frequency
	# 	points that you want to sample (numFreqs points) from.
	# 	The wrappee function, utils.select_wplane, always checks to make
	# 	sure freqWindow will allow perfect allocation of numFreq points.
	#
	# 	The entire frequency window will be (0, 0 + freq_window - 1),
	# 	where the range above only means indices over the corresponding
	#	frequency series rather than the actual frequency values.
	# 	Therefore, the starting sampled frequency point will be whatever
	#	starting frequency point in the frequency series, usually 4.
	# 	The default frequency window may be carefully selected to be 505
	# 	because 505=63*8+1. Therefore, 505 is able to give 64 selected
	# 	points exactly.
	def __init__(self, numTimes=64, numFreqs=64, freqWindow=505,
				 midTime=0.0, leftTimeWindow=-0.05, rightTimeWindow=0.03):
		self.midTime = midTime
		self.numTimes = numTimes
		self.numFreqs = numFreqs
		self.leftTimeWindow = leftTimeWindow
		self.rightTimeWindow = rightTimeWindow
		self.freqWindow = freqWindow

	# Generate an image from a TfInstance.
	# @param tfInstance (TfInstance): A TfInstance from which an image
	#	will be generated.
	# @param waveformPath (str): The path to the directory where
	#	the HDF5 waveform files are stored.
	# @return im (ndarray<*, *>): A 2-D array representing the image
	#	of a time-frequency map. This array is already flipped along
	#	the frequency axis so that imshow will be able to show the image
	#	in normal looking.
	def tfInstance2Im(self, tfInstance, waveDirPath):
		waveformName = tfInstance.waveformName + ".h5"
		waveformPath = PATH.join(waveDirPath, waveformName)
		im, _, _ = self.getTfIm(waveformPath, tfInstance.iota,
								tfInstance.phi, tfInstance.motherFreq)
		# Flipping the frequency axis to make the image look normal
		# using imshow.
		im = np.flip(im, axis=0)
		return im

	# Generate a time-frequency image.
	# @param wavePath (str): Path to the waveform file.
	# @param iota (float): In radians.
	# @param phi (float): In radians.
	def getTfIm(self, wavePath, iota, phi, motherFreq):
		assert isinstance(iota, float)
		assert isinstance(phi, float)
		assert isinstance(motherFreq, float)
		assert iota <= 2 * pi
		assert phi <= 2 * pi
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