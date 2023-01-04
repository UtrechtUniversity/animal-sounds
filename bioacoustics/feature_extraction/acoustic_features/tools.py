# -*-coding:Utf-8 -*

# Copyright: Marielle MALFANTE - GIPSA-Lab -
# Univ. Grenoble Alpes, CNRS, Grenoble INP, GIPSA-lab, 38000 Grenoble, France
# (04/2018)
#
# marielle.malfante@gipsa-lab.fr (@gmail.com)
#
# This software is a computer program whose purpose is to automatically
# processing time series (automatic classification, detection). The architecture
# is based on machine learning tools.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL

import json
import numpy as np
from sympy.ntheory import factorint
from scipy.signal import butter, lfilter, filtfilt
from .featuresFunctions import energy, energy_u
from math import sqrt


def bestFFTlength(n):
    """
    Computation can be super long for signal of length with a bad factorint.
    Compute fft on bestFFTlength points in this case.
    """
    while max(factorint(n)) >= 100:
        n -= 1
    return n


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Butter bandpass filter design
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Butter filtering
    """
    if not lowcut or not highcut:
        return data
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def extract_features(config, signals, features, fs, meta_data):
    """
    Function used to extract features outside of a recording environment
    """
    # (nData,_) = np.shape(signals)
    nData = np.shape(signals)[0]
    allFeatures = np.zeros((nData,), dtype=object)
    for i in range(np.shape(signals)[0]):
        signature = signals[i]
        # ... preprocessing

        # if config.preprocessing['energy_norm']:
        E = energy(signature, arg_dict={"E_u": energy_u(signature)})

        # print(meta_data[i])
        # print('E',E)

        # print(type(E))
        #
        # print('signature',signature)
        #
        # print('type signature',type(signature))
        if E == 0:
            signature = np.zeros(len(signature))
        else:
            signature = np.array(signature) / sqrt(E)

        # ... features extraction
        features.compute(signature, fs)
        allFeatures[i] = features.featuresValues
    # Get proper shape (nData,nFeatures) instead of (nData,)
    allFeatures = np.vstack(allFeatures)
    return allFeatures
