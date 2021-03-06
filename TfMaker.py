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
				 midTime=0.0, leftTimeWindow=-0.05, rightTimeWindow=0.03,
				 downSample=True):
		self.midTime = midTime
		self.numTimes = numTimes
		self.numFreqs = numFreqs
		self.leftTimeWindow = leftTimeWindow
		self.rightTimeWindow = rightTimeWindow
		self.freqWindow = freqWindow
		self.downSample = downSample

	# Generate an image from a TfInstance by default. Or alternatively,
	#	generate the timeArr, freqArr, and ampArr for this TfInstance.
	#	Note that in the default mode, the returned image will be
	#	flipped so that imshow can display the returned image in a
	#	normal looking.
	# @param tfInstance (TfInstance): A TfInstance from which an image
	#	will be generated.
	# @param waveformPath (str): The path to the directory where
	#	the HDF5 waveform files are stored.
	# @param includeTf (bool): If True, then this method will return
	#	the timeArr, freqArr, along with the ampArr for this TfInstance.
	# @return (default)
	# 	im (ndarray<floating, floating>): A 2-D array
	# 		representing the image of a time-frequency map.
	# 		This array is already flipped along the frequency axis
	# 		so that imshow will be able to show the image in normal
	# 		looking.
	# @return (includeTf=True)
	# 	ampArr (array<floating, floating>)
	# 	timeArr (array<floating, floating>)
	# 	freqArr (array<floating, floating>)
	def tfInstance2Im(self, tfInstance, waveDirPath, includeTf=False):
		waveformName = tfInstance.waveformName + ".h5"
		waveformPath = PATH.join(waveDirPath, waveformName)
		ampArr, timeArr, freqArr = \
			self.getTfIm(waveformPath, tfInstance.iota,
						 tfInstance.phi, tfInstance.motherFreq)
		if includeTf:
			return ampArr, timeArr, freqArr
		else:
			# Flipping the frequency axis to make the image look normal
			# using imshow.
			return np.flip(ampArr, axis=0)

	# Obtain the time-frequency map without down-sampling while still
	# requiring an interesting window.
	def selectWplaneNoDS(self, plane, times, freqs):
		plane = np.array(plane)
		freqs = np.array(freqs)
		times = np.array(times)

		# Locating the left and right interesting indices for times.
		leftTimeIdx = np.searchsorted(times,
									  self.midTime + self.leftTimeWindow)
		rightTimeIdx = np.searchsorted(times,
									   self.midTime + self.rightTimeWindow)

		# Error checking
		timeInterval = times[1] - times[0]
		idealLeftTime = self.midTime + self.leftTimeWindow
		idealRightTime = self.midTime + self.rightTimeWindow

		if times[0] < idealLeftTime + timeInterval:
			assert idealLeftTime <= times[leftTimeIdx] \
				   < idealLeftTime + timeInterval
		if times[-1] >= idealRightTime:
			assert idealRightTime <= times[rightTimeIdx] \
				   < idealRightTime + timeInterval

		times = times[leftTimeIdx: rightTimeIdx + 1]
		plane = plane[:, leftTimeIdx: rightTimeIdx + 1]

		assert plane.shape == (len(freqs), len(times))
		return plane, times, freqs

	# Generate a time-frequency image.
	# @param wavePath (str): Path to the waveform file.
	# @param iota (float): In radians.
	# @param phi (float): In radians.
	# @return sampledWplane (array<floating, floating>)
	# @return sampledTimes (array<floating>)
	# @return sampledFreqs (array<floating>)
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

		if self.leftTimeWindow is None and self.rightTimeWindow is None:
			return wplane, sampleTimes, wfreqs

		if not self.downSample:
			return self.selectWplaneNoDS(wplane, sampleTimes, wfreqs)

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

	# Get the plus polarization of a waveform.
	# @return (array<*>)
	def getHp(self, wavePath, iota, phi):
		assert isinstance(iota, float)
		assert isinstance(phi, float)
		assert iota <= 2 * pi
		assert phi <= 2 * pi
		wfData = gen_waveform(wavePath, iota, phi)
		hp = np.array(wfData["hp"])
		times = np.array(wfData["sample_times"])

		leftTimeIdx = np.searchsorted(times,
									  self.midTime + self.leftTimeWindow)
		rightTimeIdx = np.searchsorted(times,
									   self.midTime + self.rightTimeWindow)

		return hp[leftTimeIdx: rightTimeIdx + 1]


if __name__ == "__main__":
	tfmaker = TfMaker(downSample=False)
