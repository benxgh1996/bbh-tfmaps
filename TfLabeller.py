import Tkinter as tk
import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkMessageBox as msg
from utilsWaveform import *
from TfInstance import TfInstance
import os
import inspect as ins
from labelFunctions import *
from copy import copy


MOTHER_FREQ = 0.5
MAX_SCALE = 512
# FIG_WIDTH, FIG_HEIGHT = 7.6, 5.8
FIG_WIDTH, FIG_HEIGHT = 6.8, 5.2
# Default figure size
# FIG_WIDTH, FIG_HEIGHT = 6.4, 4.8
# Define useful paths.
# The directory for this file.
path = os.path
BBH_DIR = path.dirname(path.abspath(__file__))
# The directory that contains Georgia Tech waveforms.
WAVEFORM_DIR = path.join(BBH_DIR, "..", "lvcnr-lfs", "GeorgiaTech")
DATA_PATH = "tfInstances.npy"
# The width for Tk entries.
ENTRY_WIDTH = 8


class TfLabeller(object):
	# submitAction is a callback function to be invoked
	# on submitting a user label.
	# It has a signature of submitAction(chirpNum, chirpTimes).
	# nextAction is a function that will be invoked on skipping
	# the current plot.
	# @param waveform (str): An absolute path for the HDF5
	#	waveform file. This absolute path can contain "..".
	def __init__(self, waveform, iotaNum, phiNum, iotaStart,
				 iotaEnd, phiStart, phiEnd):
		assert path.isfile(waveform)
		assert isinstance(iotaNum, int)
		assert isinstance(phiNum, int)
		self.win = tk.Tk()
		self.win.title("TfLabeller")
		# Waveform related attributes
		# self.waveForm is the absolute path to the waveform file.
		self.waveform = waveform
		# inputWaveName will only look like "GT0577".
		self.waveName = utils.get_wvname(waveform)
		# The number of iota angles that we are interested in.
		self.iotaNum = iotaNum
		# The number of phi angles that we are interested in.
		self.phiNum = phiNum
		self.totalPlotNum = self.iotaNum * self.phiNum
		self.currIotaNum = 0
		self.currPhiNum = 0
		self.iota = float(iotaStart)
		self.phi = float(phiStart)
		self.iotaList = np.linspace(iotaStart, iotaEnd, iotaNum,
									dtype=float)
		self.phiList = np.linspace(phiStart, phiEnd, phiNum,
								   dtype=float)
		# When displaying plotNum in a plot, remember to
		# add 1 to it.
		self.plotNum = 0
		# self.hasSubmitted records whether the user has
		# successfully submitted any label into the labelSet.
		# hasSubmitted will be set to False again right after
		# we successfully save all labels to disk.
		self.hasSubmitted = False

		# Initiate labelSet and oldLabelNum.
		# labelSet is a set of historically labelled TfMaps
		# that are to be loaded from a npy file.
		# We should not mute any instances of labelSet since
		# strictly-speaking we should only use set on immutable
		# objects.
		try:
			self.labelSet = set(np.load(DATA_PATH))
		except IOError, e:
			if e.errno == 2:
				# Handles if data set does not exist.
				self.labelSet = set()
			else:
				raise e
		# Creating a dictionary that is useful to look up
		# whether a mock TfInstance object has double chirp.
		self.labelDict = {}
		for instance in self.labelSet:
			self.labelDict[instance] = instance.hasDoubleChirp

		# oldLabelNum is meant to keep a record of
		# the number of existing labels before the current GUI
		# session. This will be helpful for debugging by assertion
		# later on.
		# Also, note that oldLabelNum is updated only after
		# loading the existing labels from disk. It will remain
		# constant afterwards.
		self.oldLabelNum = len(self.labelSet)
		self.oldLabelSet = copy(self.labelSet)
		# Checking that we have loaded the labelSet with correct
		# instance types.
		if self.oldLabelNum > 0:
			assert isinstance(iter(self.labelSet).next(), TfInstance)

		# justLabelSet is a set that holds the successfully
		# submitted labels in the current GUI session.
		# It consists of 3-tuples of (inputWaveName, iotaNum, phiNum).
		self.justLabelSet = set()
		# Instantiate all the widgets on the window.
		self.createWidgets()

	def addJustLabel(self):
		self.justLabelSet.add((self.waveName,
							   self.currIotaNum,
							   self.currPhiNum))

	def _destroyWindow(self):
		self.win.quit()
		self.win.destroy()

	# Called by the initiator to initiate all the widgets in
	# the labeller GUI.
	def createWidgets(self):
		# Handling the window
		self.win.withdraw()
		self.win.protocol('WM_DELETE_WINDOW', self._destroyWindow)

		# The action frame.
		self.actionFrame = ttk.Frame(self.win)
		self.actionFrame.grid(row=0, column=0)
		# Creats the previous button that loads the previous plot.
		self.prevButton = ttk.Button(self.actionFrame,
									 text="Previous",
									 command=self.prevAction)
		self.prevButton.grid(row=0, column=0)
		# The next button will proceed the window to the next plot
		# without saving any labelling for the current plot.
		self.nextButton = ttk.Button(self.actionFrame,
									 text="Next",
									 command=self.nextAction)
		self.nextButton.grid(row=0, column=1)
		# Adding a reload button that can reload the current plot.
		self.reloadButton = ttk.Button(self.actionFrame,
									   text="Reload plot",
									   command=self.reloadCanvas)
		self.reloadButton.grid(row=0, column=2)
		# The submit button
		self.submitButton = ttk.Button(self.actionFrame,
									   text="Submit",
									   command=self.trySubmit)
		self.submitButton.grid(row=0, column=3)
		# The save button
		self.saveButton = ttk.Button(self.actionFrame,
									 text="Save all labels",
									 command=self.saveLabels)
		self.saveButton.grid(row=0, column=4)

		# The count frame
		self.countFrame = ttk.Frame(self.win)
		self.countFrame.grid(row=1, column=0)
		# The check button that an user can select whether
		# the time-frequency map has double chirp.
		self.hasDoubleChirp = tk.IntVar()
		self.doubleChirpButton = tk.Checkbutton(self.countFrame,
												text="Double chirp",
												variable=self.hasDoubleChirp,
												bg="gray93")
		self.doubleChirpButton.deselect()
		self.doubleChirpButton.grid(row=0, column=0)
		# The force submit check button.
		# This button will allow the user to force submit his/her
		# current label, even if the current tfmap has been
		# labelled before.
		# Caution again that the attribute forceSubmit is not int!
		self.forceSubmit = tk.IntVar()
		self.forceButton = tk.Checkbutton(self.countFrame,
										  text="Force submit",
										  variable=self.forceSubmit,
										  bg="gray93")
		# Deselect the force submit button on creation.
		self.forceButton.deselect()
		self.forceButton.grid(row=0, column=1)


		# The record frame
		self.recordFrame = ttk.Frame(self.win)
		self.recordFrame.grid(row=2, column=0)
		self.pastButton = tk.Checkbutton(self.recordFrame,
										 text="Historical label",
										 state=tk.DISABLED)
		self.pastButton.grid(row=0, column=0)
		self.justButton = tk.Checkbutton(self.recordFrame,
										 text="Submitted label",
										 state=tk.DISABLED)
		self.justButton.grid(row=0, column=1)

		# Handling the figure
		# Creating a frame to hold the canvas.
		self.figFrame = ttk.Frame(self.win)
		self.figFrame.grid(row=3, column=0)
		# self.figFrame.pack(side=tk.BOTTOM, fill=tk.X)
		self.fig = Figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
		self.axis = self.fig.add_subplot(111)
		self.canvas = FigureCanvasTkAgg(self.fig, master=self.figFrame)
		# self.canvas.get_tk_widget().grid(row=0, column=0)
		self.canvas.get_tk_widget().pack()
		# Initiate the first plot on creation of the GUI.
		self.replot()

		# # Scrollbar
		# self.scrollBar = ttk.Scrollbar(self.win, orient=tk.VERTICAL)
		# self.scrollBar.grid(row=0, rowspan=4, column=1)

	# Save all the labels that are created on this labeller
	# to disk.
	def saveLabels(self):
		if not self.hasSubmitted:
			msg.showwarning(message="You have not submitted any "
									"new labels yet.")
		else:
			assert len(self.labelSet) >= self.oldLabelNum
			# Notice that we DO want to keep the attribute
			# self.labelSet substantial even after we save all the
			# labelled data. This means that we can keep doing our
			# labelling after saving some previously labelled data.
			savePath = DATA_PATH
			np.save(savePath, list(self.labelSet))
			# Setting this to False can prevent the GUI from
			# unnecessary label savings if there are no new labels
			# submitted between two saveLabels commands.
			self.hasSubmitted = False
			msg.showinfo(message="You have saved all the labelled "
								 "data.")
			print "You have saved all the labels."

	# Reload the current canvas. This is useful on
	# scaling the GUI window by mouse dragging.
	def reloadCanvas(self):
		# Deselect the force submit button
		self.forceButton.deselect()
		self.canvas.draw()

	# Confirmation message after the user hits the submit button.
	def trySubmit(self):
		if bool(self.hasDoubleChirp.get()):
			shortMsg = "Has"
		else:
			shortMsg = "Zero"
		# Rendering the final confirmation before submitting
		# the user's labelling.
		submitMsg = shortMsg + " double chirp."
		submitAns = msg.askyesnocancel(message=submitMsg)
		if submitAns:
			submitSuccess = self.submitAction()
			# Proceeding to the next plot only if we have
			# submitted our current label successfully.
			if submitSuccess:
				self.nextAction()

	# After the user confirms submitting the current label
	# on the message box, do this. Essentially, this function
	# will add the current label to the existing label set.
	# @return A boolean, with True indicating the current
	#	label has been successfully submitted, and False otherwise.
	def submitAction(self):
		forceSubmit = bool(self.forceSubmit.get())

		curInstance = TfInstance(self.waveName, self.iota, self.phi,
								 MOTHER_FREQ,
								 bool(self.hasDoubleChirp.get()))

		# Check whether the current label is already in the labelled
		# set.
		if curInstance not in self.labelSet:
			self.labelSet.add(curInstance)
			assert curInstance not in self.labelDict
			self.labelDict[curInstance] = curInstance.hasDoubleChirp
			self.hasSubmitted = True
			self.addJustLabel()
			submitSuccess = True
		else:
			# If not forced to save label, then return False
			# to indicate that the current label is already in the
			# labelled set.
			# Therefore, this function will only return False if
			# the current label is a duplicate and the user did not
			# choose to force-submit, if the forceSubmit argument
			# is supplied.
			if not forceSubmit:
				submitSuccess = False
			# Replace the "same" old label to a new label, if forced.
			else:
				self.labelSet.remove(curInstance)
				self.labelSet.add(curInstance)
				del self.labelDict[curInstance]
				self.labelDict[curInstance] = curInstance.hasDoubleChirp
				self.hasSubmitted = True
				self.addJustLabel()
				submitSuccess = True

		if not submitSuccess:
			msg.showwarning(message="The current label duplicates "
									"an existing label. You might "
									"want to choose force submit.")
			print "label NOT submitted due to duplication."
		# Now submitSuccess must be True.
		elif forceSubmit:
			print "Label force-submitted."
		else:
			print "Label successfully submitted."

		# Command line indication of current submit.
		iotaStr = utils.ang_to_str(self.iotaList[self.currIotaNum])
		phiStr = utils.ang_to_str(self.phiList[self.currPhiNum])
		print "{}. {}, iota: {}, phi: {}".format(self.plotNum+1, self.waveName,
												 iotaStr, phiStr)
		print
		return submitSuccess


	# Replots the canvas given the updated iota and phi
	# numbers.
	def replot(self):
		# First, deselect the force submit button.
		self.forceButton.deselect()
		self.pastButton.deselect()
		self.justButton.deselect()
		self.doubleChirpButton.deselect()
		iota = self.iotaList[self.currIotaNum]
		phi = self.phiList[self.currPhiNum]
		# Updating the current iota and phi values whenever
		# we replot.
		self.iota = float(iota)
		self.phi = float(phi)

		# Select the pastButton if applicable.
		tfMock = TfInstance(self.waveName, self.iota,
							self.phi, MOTHER_FREQ)
		if tfMock in self.oldLabelSet:
			self.pastButton.select()
		# Select the justButton if applicable.
		if (self.waveName, self.currIotaNum, self.currPhiNum) in \
				self.justLabelSet:
			self.justButton.select()
		# Notice that self.labelSet is always substantially created
		# in the initiator.
		curPlotDoubleChirp = None
		if tfMock in self.labelSet:
			curPlotDoubleChirp = self.labelDict[tfMock]

		iotaStr = utils.ang_to_str(iota)
		phiStr = utils.ang_to_str(phi)
		# The the waveform-related data
		wf_data = gen_waveform(self.waveform, iota, phi)
		tf_data = tf_decompose(wf_data['hp'],
							   wf_data["sample_times"],
							   MOTHER_FREQ, MAX_SCALE)
		wplane = tf_data["wplane"]
		wfreqs = tf_data["wfreqs"]
		sample_times = wf_data["sample_times"]
		# Selecting the data for plotting.
		wplane_sel, freqs_sel, times_sel \
			= utils.select_wplane(wplane,
								  wfreqs,
								  sample_times,
								  mid_t=0,
								  xnum=500,
								  ynum=350,
								  left_t_window=-0.05,
								  right_t_window=0.03,
								  freq_window=500)
		# Plot the time-frequency map.
		self.axis.clear()
		self.axis.pcolormesh(times_sel, freqs_sel, wplane_sel,
							 cmap="gray")
		self.axis.set_xlabel("time (s)")
		self.axis.set_ylabel("frequency (Hz)")
		title = "{}, {}/{}*{}, iota: {}, phi: {}"\
			.format(self.waveName, self.plotNum+1,
					self.iotaNum, self.phiNum, iotaStr, phiStr)
		if curPlotDoubleChirp is not None:
			if curPlotDoubleChirp:
				title += "\nDouble Chirp: Yes"
			else:
				title += "\nDouble Chirp: No"
		self.axis.set_title(title)
		self.canvas.draw()

	# Loads the previous plot.
	def prevAction(self):
		self.plotNum = (self.plotNum - 1) % self.totalPlotNum
		# If currPhiNum down-exceeds its lower limit, then decrease
		# the currIotaNum.
		if self.currPhiNum == 0:
			self.currIotaNum = (self.currIotaNum - 1) % self.iotaNum
		self.currPhiNum = (self.currPhiNum - 1) % self.phiNum
		self.replot()

	# Loads the next plot.
	def nextAction(self):
		self.plotNum = (self.plotNum + 1) % self.totalPlotNum
		# If currPhiNum exceeds its upper limit, then increase
		# the currIotaNum.
		if self.currPhiNum == self.phiNum - 1:
			self.currIotaNum = (self.currIotaNum + 1) % self.iotaNum
		self.currPhiNum = (self.currPhiNum + 1) % self.phiNum
		self.replot()

	def main(self):
		self.win.update()
		self.win.deiconify()
		self.win.mainloop()


if __name__ == "__main__":
	waveform = "GT0453.h5"
	waveform = path.join(WAVEFORM_DIR, waveform)
	# It's fine to use int here because I have explicitly specified
	# the dtype for linspace in the initiator to be float.
	iotaStart, iotaEnd = 0, pi
	phiStart, phiEnd = 0, 2*pi
	iotaNum = 10
	phiNum = 10
	labelGui = TfLabeller(waveform, iotaNum, phiNum,
						  iotaStart, iotaEnd, phiStart, phiEnd)
	labelGui.main()
