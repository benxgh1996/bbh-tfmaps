import matplotlib.pyplot as plt
import utils
from utilsWaveform import *
import otter
import otter.bootstrap as bt
import os

path = os.path
BBH_DIR = path.dirname(path.abspath(__file__))
# The directory that contains Georgia Tech waveforms.
WAVEFORM_DIR = path.join(BBH_DIR, "..", "lvcnr-lfs",
						 "GeorgiaTech")
inputWaveName = "GT0453"
motherFreq = 0.4
maxScale = 512
waveList = [inputWaveName]

# Creating a web report
report = otter.Otter("otterHtmls/{}_mf={:.2f}.html".
					 format(inputWaveName, motherFreq),
					 author="Gonghan Xu",
					 title="Time-frequency maps for "
						   "{} with mother frequency {:.2f}".
					 format(inputWaveName, motherFreq))
with report:
	# Putting the plots onto the web page.
	for waveName in waveList:
		plotNum = 0
		wavePath = path.join(WAVEFORM_DIR, waveName+".h5")
		for iota in np.linspace(pi/4, pi, 4, endpoint=False):
			for phi in np.linspace(0, 2 * pi, 4, endpoint=False):
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
				plt.show()
				# Writing the downsampled time_freq image
				# Creating a row with three columns.
				row = bt.Row(3)
				# Putting the figure to the middle cell.
				row[1].width = 6
				row[1] + fig
				report + row

