from tfmap import TfMap
import numpy as np

# This function will add the current label, if possible,
# 	into a set of all historically existing labels.
# 	This function is invoked after the user clicks the
#	submit button and confirms his/her submission after
# 	seeing the pop-up confirmation message.
# @Return: A boolean. True means that the current
# 	label is added (possibly forced) to the labelled set.
#	False means that the current label is not added to
# 	the labelled set due to duplication.
def submitLabel(labeller, forceSubmit=False):
	# Making sure that labeller GUI has a substantial
	# attribute that holds an instance of all the historically
	# labelled time-frequency maps.
	# Now we want to add the new labelled data to the labelSet.
	currLabel = TfMap(labeller.waveName, labeller.iota, labeller.phi,
					 labeller.timeArr, labeller.freqArr,
					 labeller.intensityArr, labeller.chirpNum.get(),
					  labeller.chirpTimes)
	# Check whether the current label is already in the labelled
	# set.
	if currLabel not in labeller.labelSet:
		labeller.labelSet.add(currLabel)
		labeller.hasSubmitted = True
		labeller.addJustLabel()
		return True
	else:
		# If not forced to save label, then return False
		# to indicate that the current label is already in the
		# labelled set.
		# Therefore, this function will only return False if
		# the current label is a duplicate and the user did not
		# choose to force-submit, if the forceSubmit argument
		# is supplied.
		if not forceSubmit:
			return False
		# Replace the "same" old label to a new label, if forced.
		else:
			labeller.labelSet.remove(currLabel)
			labeller.labelSet.add(currLabel)
			labeller.hasSubmitted = True
			labeller.addJustLabel()
			return True
