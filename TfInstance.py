import numpy as np
from utils import *


# Represents a labelled time-frequency map.
class TfInstance(object):
	# @param iota (float): In radian.
	# @param phi (float): In radian.
	def __init__(self, waveformName, iota, phi, motherFreq,
				 hasDoubleChirp=None,
				 spin1x=None, spin1y=None, spin1z=None,
				 spin2x=None, spin2y=None, spin2z=None,
				 finalSpin=None, finalMass=None,
				 timeArr=None, freqArr=None, ampArr=None,
				 strongCheck=True):
		if strongCheck:
			assert waveformName.find(".") == waveformName.find("/") == -1
			assert waveformName[:2] == "GT"
			assert type(iota) == float
			assert type(phi) == float
		self.waveformName = waveformName
		self.iota = iota
		self.phi = phi
		self.motherFreq = motherFreq
		self.hasDoubleChirp = hasDoubleChirp
		self.spin1x = spin1x
		self.spin1y = spin1y
		self.spin1z = spin1z
		self.spin2x = spin2x
		self.spin2y = spin2y
		self.spin2z = spin2z
		self.finalSpin = finalSpin
		self.finalMass = finalMass
		self.timeArr = timeArr
		self.freqArr = freqArr
		# ampArr is a 2-D array of floating point values.
		# When filled with actual data, it will have increasing
		# row indices corresponding to increasing frequency values.
		# It will also have increasing column indices corresponding to
		# increasing time values.
		self.ampArr = ampArr

	# Factory for re-wrapping a legacy TfInstance object. The use
	# case for this is primarily for converting a TfInstance object
	# from a numpy-pickled (as .npy) object array. The pickled TfInstance
	# object may lack some attributes of the more recently updated
	# TfInstance class. This factory method does not do deep copy.
	# @param tfIns (TfInstance)
	# @return newIns (TfInstance)
	@staticmethod
	def factory(tfIns):
		newIns = TfInstance(None, None, None, None, strongCheck=False)
		newIns.waveformName = tfIns.waveformName
		newIns.iota = tfIns.iota
		newIns.phi = tfIns.phi
		newIns.motherFreq = tfIns.motherFreq
		newIns.hasDoubleChirp = tfIns.hasDoubleChirp
		newIns.spin1x = tfIns.spin1x
		newIns.spin1y = tfIns.spin1y
		newIns.spin1z = tfIns.spin1z
		newIns.spin2x = tfIns.spin2x
		newIns.spin2y = tfIns.spin2y
		newIns.spin2z = tfIns.spin2z
		newIns.finalSpin = tfIns.finalSpin
		newIns.finalMass = tfIns.finalMass
		newIns.timeArr = tfIns.timeArr if hasattr(tfIns, "timeArr") else None
		newIns.freqArr = tfIns.freqArr if hasattr(tfIns, "freqArr") else None
		newIns.ampArr = tfIns.ampArr if hasattr(tfIns, "ampArr") else None
		return newIns

	# This method is designed for comparing the iota angles.
	# This is because comparing floats can be tricky due to
	# precision problems. For example, 1.0 + 2.0 != 3.0.
	def iotaRound(self, ndigits=6):
		return round(self.iota, ndigits=ndigits)

	def phiRound(self, ndigits=6):
		return round(self.phi, ndigits=ndigits)

	def motherFreqRound(self, ndigits=6):
		return round(self.motherFreq, ndigits=ndigits)

	# Returns whether or not this object is light-weight, i.e. whether
	# or not this object contains array-like, non-parameter data.
	def isLight(self):
		return (self.timeArr is None) and (self.freqArr is None) \
				and (self.ampArr is None)

	# The __eq__, __ne__, and __hash__ methods are rewritten
	# to make TfInstance objects capable of working with sets.
	def __eq__(self, other):
		if isinstance(other, TfInstance):
			return self.waveformName == other.waveformName and \
				   self.iotaRound() == other.iotaRound() and \
				   self.phiRound() == other.phiRound() and \
				   self.motherFreqRound() == other.motherFreqRound()
		else:
			return False

	def __ne__(self, other):
		return not self.__eq__(other)

	# Rewritten to make TfInstance objects sortable.
	def __cmp__(self, other):
		if cmp(self.waveformName, other.waveformName) != 0:
			return cmp(self.waveformName, other.waveformName)
		elif cmp(self.motherFreqRound(), other.motherFreqRound()) != 0:
			return cmp(self.motherFreqRound(), other.motherFreqRound())
		elif cmp(self.iotaRound(), other.iotaRound()) != 0:
			return cmp(self.iotaRound(), other.iotaRound())
		else:
			return cmp(self.phiRound(), other.phiRound())

	def __hash__(self):
		# Returning the hash of a 4-tuple.
		return hash((self.waveformName, self.iotaRound(),
					 self.phiRound(), self.motherFreqRound()))

	def __str__(self):
		return "{}, iota: {}, phi: {}, motherFreq: {:.2f}, {}."\
			.format(self.waveformName, ang_to_str(self.iota),
					ang_to_str(self.phi), self.motherFreq,
					self.hasDoubleChirp)

	def __repr__(self):
		return self.__str__()

