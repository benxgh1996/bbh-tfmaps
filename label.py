import Tkinter as tk
import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkMessageBox as msg
from utilsWaveform import *

MOTHER_FREQ = 0.5
MAX_SCALE = 512
FIG_WIDTH, FIG_HEIGHT = 7.6, 5.8
# Default figure size
# FIG_WIDTH, FIG_HEIGHT = 6.4, 4.8


class ProjLabeller():
	# submitAction is a callback function to be invoked
	# on submitting a user label.
	# It has a signature of submitAction(chirpNum, chirpTimes).
	# nextAction is a function that will be invoked on skipping
	# the current plot.
	def __init__(self, waveform, iotaNum, phiNum):
		self.win = tk.Tk()
		self.win.title("Project Labeller")
		# Waveform related attributes
		self.waveform = waveform
		self.waveName = utils.get_wvname(waveform)
		# The number of iota angles that we are interested in.
		self.iotaNum = iotaNum
		# The number of phi angles that we are interested in.
		self.phiNum = phiNum
		self.totalPlotNum = self.iotaNum * self.phiNum
		self.currIotaNum = 0
		self.currPhiNum = 0
		self.iotaList = np.linspace(0, pi, iotaNum, endpoint=False)
		self.phiList = np.linspace(0, 2*pi, phiNum, endpoint=False)
		# When displaying plotNum in a plot, remember to
		# add 1 to it.
		self.plotNum = 0
		# Instantiate all the widgets on the window.
		self.createWidgets()

	def _destroyWindow(self):
		self.win.quit()
		self.win.destroy()

	def createWidgets(self):
		# Handling the window
		self.win.withdraw()
		self.win.protocol('WM_DELETE_WINDOW', self._destroyWindow)

		# The action frame.
		self.actionFrame = ttk.Frame(self.win)
		self.actionFrame.grid(row=0, column=0)
		# Creats the previous button that loads the previous plot.
		self.prevButton = ttk.Button(self.actionFrame,
									 text="Prev",
									 command=self.prevAction)
		self.prevButton.grid(row=0, column=0)
		# The next button will proceed to window to the next plot
		# without saving any labelling for the current plot.
		self.nextButton = ttk.Button(self.actionFrame,
									 text="Next",
									 command=self.nextAction)
		self.nextButton.grid(row=0, column=1)
		# Adding a reload button that can reload the current plot.
		self.reloadButton = ttk.Button(self.actionFrame,
									   text="Reload",
									   command=self.reloadCanvas)
		self.reloadButton.grid(row=0, column=2)
		# The clear button
		self.clearButton = ttk.Button(self.actionFrame,
									  text="Clear",
									  command=self.clearTimeFrame)
		self.clearButton.grid(row=0, column=3)
		# The submit button
		self.submitButton = ttk.Button(self.actionFrame,
									   text="Submit",
									   command=self.trySubmit)
		self.submitButton.grid(row=0, column=4)

		# The count frame
		self.countFrame = ttk.Frame(self.win)
		self.countFrame.grid(row=1, column=0)
		# Creating chirp counter
		# The chirp count indicator
		self.countLabel = ttk.Label(self.countFrame,
									text="Number of chirps")
		self.countLabel.grid(row=0, column=0)
		# The chirp count entry
		self.chirpNum = tk.IntVar()
		self.countEntry = ttk.Entry(self.countFrame,
									textvariable=self.chirpNum)
		self.countEntry.grid(row=0, column=1)
		# The chirp count confirmation button
		self.countButton = ttk.Button(self.countFrame,
									  text="Confirm",
									  command=self.confirmCount)
		self.countButton.grid(row=0, column=2)

		# The time frame.
		self.timeFrame = ttk.Frame(self.win)
		self.timeFrame.grid(row=2, column=0)
		# Creating the timeWidgets here so that this attribute
		# always exist when we try to destroy the existing
		# timeWidgets.
		# A list that stores the time widgets.
		# Each row of widgets is stored as a dict element.
		self.timeWidgets = []

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
		self.replot()

	# Reload the current canvas. This is useful on
	# scaling the GUI window by mouse dragging.
	def reloadCanvas(self):
		self.canvas.draw()

	# Rendering the chirp times input entries on confirmation
	# of chirp count number.
	def confirmCount(self):
		# Reinitialize the time frame only if entering
		# a different chirp number.
		if self.chirpNum.get() == len(self.timeWidgets):
			return
		else:
			self.destroyTimeWidgets()
		# Reinitialize the time widgets list so that
		# we will not have more time widgets stored in the list
		# than there actually are due to appending.
		self.timeWidgets = []
		chirpNum = self.chirpNum.get()
		assert type(chirpNum) == int
		MAX_ROW_ENTRIES = 4
		# The current colNum of the to-be-created widget.
		colNum = 0
		# This loop will create all the widgets that help
		# to record the time positions of chirps.
		for i in range(chirpNum):
			# widgetDict is a dictionary that holds all the widgets
			# and the input time information for a chirp.
			widgetDict = {}
			# The current row number for this entry.
			rowNum = i / MAX_ROW_ENTRIES
			# The indicator label
			widgetDict["chirpLabel"] = ttk.Label(self.timeFrame,
												 text="C{} (sec):".
												 format(i+1))
			widgetDict["chirpLabel"].grid(row=rowNum, column=colNum)
			colNum += 1
			# The variable that holds the input time.
			widgetDict["chirpTime"] = tk.DoubleVar()
			# The entry where the user inputs the chirp time.
			widgetDict["timeEntry"] = ttk.Entry(self.timeFrame,
												textvariable=
												widgetDict["chirpTime"])
			widgetDict["timeEntry"].grid(row=rowNum, column=colNum)
			colNum = (colNum + 1) % (MAX_ROW_ENTRIES * 2)
			self.timeWidgets.append(widgetDict)

	# Final confirmation before submitting user label.
	def trySubmit(self):
		chirpNum = self.chirpNum.get()
		assert chirpNum >= 0
		# Making sure the timeWidgets list stores the right number
		# of chirps.
		assert len(self.timeWidgets) == chirpNum
		chirpTimes = []
		for i in range(chirpNum):
			chirpTimes.append(self.timeWidgets[i]["chirpTime"].get())
			# Making sure the chirp times are in ascending order.
			if i > 0:
				assert chirpTimes[i] >= chirpTimes[i-1]

		# Rendering the final confirmation before submitting
		# the user's labelling.
		submitMsg = "Are you sure you want to submit your labelling?\n"\
					"Number of chirps: {}\n".format(chirpNum)
		for i in range(chirpNum):
			submitMsg += "Chirp {}: {} sec\n".format(i+1, chirpTimes[i])
		submitAns = msg.askyesnocancel("Submitting your labelling",
									   submitMsg)
		if submitAns:
			self.submitAction(chirpNum, chirpTimes)
			# Proceeding to the next plot.
			self.nextAction()

	# Clearing all the entries of the current window.
	# This will diminish all the time entries.
	def clearTimeFrame(self):
		# Clear the chirp count entry
		self.countEntry.delete(0, tk.END)
		self.destroyTimeWidgets()

	# This function destroys all the currently present
	# time-related widgets.
	def destroyTimeWidgets(self):
		self.timeFrame.destroy()
		self.timeFrame = ttk.Frame(self.win)
		self.timeFrame.grid(row=2, column=0)

	def submitAction(self, chirpNum, chirpTimes):
		print "{}th".format(self.plotNum+1)
		iotaStr = utils.ang_to_str(self.iotaList[self.currIotaNum])
		phiStr = utils.ang_to_str(self.phiList[self.currPhiNum])
		print "{}, iota: {}, phi: {}".format(self.waveName, iotaStr,
											 phiStr)
		print "chirpNum: {}".format(chirpNum)
		print "chirpTimes: ",
		for time in chirpTimes:
			print "{:.2f}, ".format(time),
		print
		print

	# Replots the canvas given the updated iota and phi
	# numbers.
	def replot(self):
		iota = self.iotaList[self.currIotaNum]
		phi = self.phiList[self.currPhiNum]
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
								  ynum=500,
								  left_t_window=-0.05,
								  right_t_window=0.05,
								  freq_window=500)
		# Plot the time-frequency map.
		self.axis.clear()
		self.axis.pcolormesh(times_sel, freqs_sel, wplane_sel,
							 cmap="gray")
		self.axis.set_xlabel("time (s)")
		self.axis.set_ylabel("frequency (Hz)")
		self.axis.set_title("{} {}th, iota: {}, phi: {}".
							format(self.waveName, self.plotNum+1,
								   iotaStr, phiStr))
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
	waveform = "GT0577.h5"
	iotaNum = 4
	phiNum = 4
	labelHelper = ProjLabeller(waveform, iotaNum, phiNum)
	labelHelper.main()