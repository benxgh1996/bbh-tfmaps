import matplotlib.pyplot as plt
import utils
from utilsWaveform import *
import otter
import otter.bootstrap as bt
import os
import TfInstance
import time
from Classifier import *
from tfRun import *
import sys

PATH = os.path
BBH_DIR = PATH.dirname(PATH.abspath(__file__))
# The directory that contains Georgia Tech waveforms.
WAVEFORM_DIR = PATH.join(BBH_DIR, "..", "lvcnr-lfs",
						 "GeorgiaTech")
inputWaveName = "GT0448"
motherFreq = 0.5
maxScale = 512
waveList = [inputWaveName]

# Plot annotated instances.
def plotInstances(dataSetName):
	instances = np.load(dataSetName)
	instances.sort()
	report = otter.Otter("otterHtmls/GT0453_100SampleInstances.html",
						 author="Gonghan Xu",
						 title="100 time-frequency maps for "
							   "{} with manual label".
						 format(instances[0].waveformName))
	# i = 0
	with report:
		plotNum = 0
		for instance in instances:
			# i += 1
			# if i == 5:
			# 	break
			# Loading parameters.
			plotNum += 1
			waveName = instance.waveformName
			wavePath = PATH.join(WAVEFORM_DIR, waveName+".h5")
			iota = instance.iota
			phi = instance.phi
			hasDoubleChirp = instance.hasDoubleChirp
			motherFreq = instance.motherFreq
			iotaStr = utils.ang_to_str(iota)
			phiStr = utils.ang_to_str(phi)
			# Generate the down-sampled time-freq map
			# used for training with embedded data parameters.
			wf_data = gen_waveform(wavePath, iota, phi)
			tf_data = tf_decompose(wf_data['hp'],
								   wf_data["sample_times"],
								   motherFreq, maxScale)
			wplane = tf_data["wplane"]
			wfreqs = tf_data["wfreqs"]
			sample_times = wf_data["sample_times"]
			# Get the selected region.
			wplane_sel, freqs_sel, times_sel = \
				utils.select_wplane(wplane, wfreqs,
									sample_times,
									mid_t=0, xnum=500,
									ynum=350,
									left_t_window=-0.05,
									right_t_window=0.03,
									freq_window=500)
			print "{}. {}, iota: {}, phi: {}, motherFreq: {:.2f}".\
				format(plotNum, waveName, iotaStr, phiStr, motherFreq)

			fig, ax = plt.subplots(figsize=fig_size)
			ax.pcolormesh(times_sel, freqs_sel, wplane_sel,
						  cmap="gray")
			ax.set_xlabel("time (s)")
			ax.set_ylabel("frequency (Hz)")
			ax.set_title("{}. {}, iota: {}, phi: {}, "
						 "mother freq: {:.2f}"
						 .format(plotNum, waveName,
								 iotaStr, phiStr, motherFreq))
			# plt.show()
			# Writing the down-sampled time_freq image.
			# Creating a row with three columns.
			row = bt.Row(3)
			# Putting the figure to the middle cell.
			row[1].width = 6
			row[1] + fig
			report + row
			row = bt.Row(3)
			row[1] + "({}) Double Chirp: {}".format(plotNum, hasDoubleChirp)
			report + row


# Creating TfMaps without adding labels.
def plotArbiAngles():
	# Creating a web report
	report = otter.Otter("otterHtmls/GT0448_100SampleTfMaps.html",
						 author="Gonghan Xu",
						 title="100 Time-frequency maps without labels")
	with report:
		# Putting the plots onto the web page.
		for waveName in waveList:
			plotNum = 0
			wavePath = PATH.join(WAVEFORM_DIR, waveName + ".h5")
			for iota in np.linspace(0, pi, 10, endpoint=True):
				for phi in np.linspace(0, 2*pi, 10, endpoint=True):
					iotaStr = utils.ang_to_str(iota)
					phiStr = utils.ang_to_str(phi)
					# Constructing a data row
					# Writing the data parameters
					plotNum += 1
					# Generate the downsampled time-freq map
					# used for training
					wf_data = gen_waveform(wavePath, iota, phi)
					tf_data = tf_decompose(wf_data['hp'],
										   wf_data["sample_times"],
										   motherFreq, maxScale)
					wplane = tf_data["wplane"]
					wfreqs = tf_data["wfreqs"]
					sample_times = wf_data["sample_times"]
					# Get the selected region.
					wplane_sel, freqs_sel, times_sel = \
						utils.select_wplane(wplane, wfreqs,
											sample_times,
											mid_t=0, xnum=500,
											ynum=350,
											left_t_window=-0.05,
											right_t_window=0.03,
											freq_window=500)
					print "{}. {}, iota: {}, phi: {}".\
						format(plotNum, waveName, iotaStr, phiStr)

					fig, ax = plt.subplots(figsize=fig_size)
					ax.pcolormesh(times_sel, freqs_sel, wplane_sel,
								  cmap="gray")
					ax.set_xlabel("time (s)")
					ax.set_ylabel("frequency (Hz)")
					ax.set_title("{}. {}, iota: {}, phi: {}, "
								 "mother freq: {:.2f}"
								 .format(plotNum, waveName,
										 iotaStr, phiStr, motherFreq))
					# plt.show()
					# Writing the downsampled time_freq image
					# Creating a row with three columns.
					row = bt.Row(3)
					# Putting the figure to the middle cell.
					row[1].width = 6
					row[1] + fig
					report + row


def plotLabelledSet():
	# Creating a web report
	report = otter.Otter("otterHtmls/training_set_noDS.html",
						 author="Gonghan Xu",
						 title="Training Set")
	dat = np.load("heavyTrainSet_noDS.npy")
	dat = sorted(dat)
	# dat = dat[: 10]
	startLoadTime = time.time()
	numIns = len(dat)

	with report:
		# Putting the plots onto the web page.
		for idx, ins in enumerate(dat, start=1):
			if idx % 20 == 0:
				print "Generating {}st/{} image".format(idx, numIns)
			fig, ax = ins.getPlot()

			# Writing the downsampled time_freq image
			# Creating a row with three columns.
			row = bt.Row(3)
			row[0] + idx
			# Putting the figure to the middle cell.
			row[1].width = 6
			row[1] + fig
			report + row
			if ins.hasDoubleChirp:
				row[2] + "Double Chirp"
			else:
				row[2] + "Not Double Chirp"
			plt.close(fig)

	endLoadTime = time.time()
	loadTime = endLoadTime - startLoadTime
	print "Web page generation time:", loadTime, "sec"


def plotProbs():
	# Creating a web report
	report = otter.Otter("otterHtmls/probs.html",
						 author="Gonghan Xu",
						 title="All Cases (Linear Kernel; "
							   "Using Original Image Size: 508 * 328)")

	dat = np.load("heavyTrainSet_noDS.npy")
	# dat = sorted(dat)
	# dat = dat[: 248]
	startLoadTime = time.time()
	numIns = len(dat)
	shuffIndices, probs, accus, fails, confMat \
		= getCrossValProbs(dat, nfolds=5)

	# sys.exit()
	print "Start generating plots.."
	numAccurates = 0
	counter = 0
	with report:
		# Putting the plots onto the web page.
		for c, i in enumerate(range(numIns), start=1):
			if c % 50 == 0:
				print "At {}st/{} image".format(c, numIns)
			# if not fails[i]:
			# 	continue
			counter += 1
			ins = dat[shuffIndices[i]]
			fig, ax = ins.getPlot()

			# Writing the downsampled time_freq image
			# Creating a row with three columns.
			row = bt.Row(4)
			row[0] + counter
			# Putting the figure to the middle cell.
			row[1].width = 6
			row[1] + fig
			if ins.hasDoubleChirp:
				row[2] + "Hand Label: Double Chirp"
			else:
				row[2] + "Hand Label: Not Double Chirp"
			row[3] + "Predicted Prob (Double Chirp): {:.3f}"\
				.format(probs[i])
			report + row
			plt.close(fig)

			# if ins.hasDoubleChirp:
			# 	numAccurates += 1 if probs[i] > 0.5 else 0
			# else:
			# 	numAccurates += 1 if probs[i] <= 0.5 else 0

			# if dat[shuffIndices[i]].hasDoubleChirp:
			# 	numAccurates += 1 if probs[shuffIndices[i]] > 0.5 else 0
			# else:
			# 	numAccurates += 1 if probs[shuffIndices[i]] <= 0.5 else 0

	endLoadTime = time.time()
	loadTime = endLoadTime - startLoadTime
	print "Web page generation time:", loadTime, "sec"
	print "Verifying classification accuracy: {:.5f}"\
		.format(1 - 1.0 * sum(fails) / numIns)


if __name__ == "__main__":
	# plotInstances("tfInstances.npy")
	# plotArbiAngles()
	# plotLabelledSet()
	plotProbs()

	pass
