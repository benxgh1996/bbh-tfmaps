import numpy as np
from utils import *


# Represents a labelled time-frequency map.
class TfInstance(object):
	def __init__(self, waveformName, iota, phi, motherFreq,
				 hasDoubleChirp=None,
				 spin1x=None, spin1y=None, spin1z=None,
				 spin2x=None, spin2y=None, spin2z=None,
				 finalSpin=None, finalMass=None):
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

	# This method is designed for comparing the iota angles.
	# This is because comparing floats can be tricky due to
	# precision problems. For example, 1.0 + 2.0 != 3.0.
	def iotaRound(self, ndigits=6):
		return round(self.iota, ndigits=ndigits)

	def phiRound(self, ndigits=6):
		return round(self.phi, ndigits=ndigits)

	def motherFreqRound(self, ndigits=6):
		return round(self.motherFreq, ndigits=ndigits)

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

