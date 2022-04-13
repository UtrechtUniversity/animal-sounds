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

from os.path import isfile
from svm_features.featuresFunctions import *
from svm_features.tools import bestFFTlength
import json

class FeatureVector:
    """
    Feature vector object to compute and store features values for a given signal
    A feature vector object can be computed many times for various signals.
    It is a pattern.
    The basic idea is to create a feature vector object as a 'pattern' of features to be computed on a dataset.
    Only one feature vector is create, and it is computed as many times as there are data to analyze.
    - pathToFeatureConfig: path to configuration file. Why do we need this path if the config object is passed as argument in __init__ ???
    - domains: list of domains in which the features are to be cocmputed (as str). I.E. ['time', 'frequeny', 'cepstral']. Is automatically read from onfig object.
    - n_domains: number of domains in whih the features are omputed (len of domains)
    - n_features: number of features to compute IN ONE DOMAIN. /!\ /!\ /!\ /!\ /!\ (value set when _readFeaturesFunctions is called)
    - featuresFunctions: functions needed to compute the features, read from config file when _readFeaturesFunctions is called
    - intermValues: needed intermediate values, factorized to optimize computation times.
    - featuresValues: value of all features, is of size (n_domains*n_features, ), init full of zeros in compute,
    takes actual values when _computation is called in compute.
    - featuresRef: ref of all features, is of size (n_domains*n_features, )
    - featuresOptArguments: opt arguments for each feature function, is of size (n_domains*n_features, ), each element is None or a dict
    - featuresComputationDomains: feature computation domain for each feature, is of size (n_domains*n_features, )
    - _verbatim: how chatty do you want your computer to be?
    - _readFeaturesFunctions: reads the feature vector "pattern" from config files, loads all the function, etc (proper init of the object needed before calling compute)
    """

    def __init__(self,config,verbatim=0):
        """
        Initialization method
        """
        self.configFeatures = config.features #config.general['project_root'] + config.features['path_to_config']
        self.domains = config.domain.split(' ')
        self.n_domains = len(self.domains)
        self.n_features = None # /!\ In ONE domain
        self.featuresFunctions = None
        self.intermValues = None
        self.featuresValues = None
        self.featuresRef = None
        self.featuresOptArguments = None
        self.featuresComputationDomains = None



        # Read features functions
        self._readFeaturesFunctions()

    def compute(self,signal,fs):
        """
        Compute features from signal, according to configuration information given at __init__
        /!\ signal is already in the proper bandwidth
        """

        # Get signals for all domains
        signals = np.zeros((self.n_domains,),dtype=object)
        for i,domain in enumerate(self.domains):
            if domain == 'time':
                signals[i] = signal
            elif domain == 'spectral':
                signals[i] = np.absolute(np.fft.fft(signal, bestFFTlength(len(signal))))
            elif domain == 'cepstral':
                signals[i] = np.absolute(np.fft.fft(np.absolute(np.fft.fft(signal, bestFFTlength(len(signal))))))
            else:
                print('computation domain should be time, spectral or cepstral')
                return
        # Define variables: featuresValues
        self.featuresValues = np.zeros((self.n_features*self.n_domains,),dtype=float)
        # Proceed to actual computation
        self._computation(signals,fs)

    def _readFeaturesFunctions(self):
        """
        Get functions needed to compute features from json file.
        NB: Functions are stored in external file, check README for more info
        """
        # Read features config file
        # configFeatures = json.load(open(self.pathToFeatureConfig,'rb'))

        # Set right dimensions depending on number of features and number of domains
        self.n_features = len(self.configFeatures.keys())
        self.featuresRef = np.zeros((self.n_domains*self.n_features,),dtype=object)
        #self.Values will be defined later, during computation
        self.featuresFunctions = np.zeros((self.n_domains*self.n_features,),dtype=object)
        self.featuresOptArguments = np.zeros((self.n_domains*self.n_features,),dtype=object)
        self.featuresComputationDomains = np.zeros((self.n_domains*self.n_features,),dtype=object)

        # Get all featuresFunctions, featuresRef, featuresOptArguments from configFeatures
        # Also set featuresComputationDomains
        # -----> First find them for no specific domain (-> Unique)
        featuresFunctionsUnique = np.zeros((self.n_features,),dtype=object)
        featuresOptArgumentsUnique = np.zeros((self.n_features,),dtype=object)
        featuresRefUnique = np.zeros((self.n_features,),dtype=object)
        for i,i_feature in enumerate(sorted(list(self.configFeatures.keys()))):
            featuresFunctionsUnique[i] = eval(self.configFeatures[i_feature]["function"])
            featuresOptArgumentsUnique[i] = eval(self.configFeatures[i_feature]["function_opt_arg"])
            featuresRefUnique[i] = str(i_feature)
        # -----> Then extend to all domains
        for i,domain in enumerate(self.domains):
            self.featuresFunctions[i*self.n_features:(i+1)*self.n_features] = featuresFunctionsUnique
            self.featuresOptArguments[i*self.n_features:(i+1)*self.n_features] = featuresOptArgumentsUnique
            self.featuresRef[i*self.n_features:(i+1)*self.n_features] = [domain[0]+f for f in featuresRefUnique]
            self.featuresComputationDomains[i*self.n_features:(i+1)*self.n_features] = [domain[0].upper()]*self.n_features

    def _intermComputation(self,signal_in_domain,fs):
        """
        Proceed to interm computation in one domain. Interm values are needed
        before computation (computation factorization)
        - signal_in_domain : shape (length,) contains signal in the needed domain
        - fs: sampling frequency
        """
        # /!\ intermValues for ONE domain. Need to recompute for each domain.
        self.intermValues = dict()

        # Interm values computation.
        self.intermValues['fs'] = fs
        self.intermValues['u'] = np.linspace(0, (len(signal_in_domain) - 1)*(1/fs), len(signal_in_domain))
        self.intermValues['E_u'] = energy_u(signal_in_domain, self.intermValues)
        self.intermValues['E'] = energy(signal_in_domain, self.intermValues)
        self.intermValues['u_bar'] = u_mean(signal_in_domain, self.intermValues)
        self.intermValues['RMS_u'] = RMS_u(signal_in_domain, self.intermValues)

    def _computation(self,signals,fs):
        """
        Compute featuresValues associated to self FeatureVector pattern
        - signals : shape (n_domains,) contains signals in all the domains
        - self.featuresValues is defined and set in this function
        """
        for i,domain in enumerate(self.domains):
            # Compute needed interm values (needed for computation)
            self._intermComputation(signals[i],fs)
            # Compute each feature value
            for j in range(self.n_features): #i*n_features + j -est features.py
                # If there already is a dictionnary of optional arguments, copy
                # and update it with interm values. Then compute feature.
                if self.featuresOptArguments[i*self.n_features + j]:
                    new_dictionary = self.intermValues.copy()
                    new_dictionary.update(self.featuresOptArguments[i*self.n_features + j])
                    self.featuresValues[i*self.n_features + j] = self.featuresFunctions[i*self.n_features + j](signals[i],new_dictionary)
                # Otherwise directly compute feature value.
                else:
                    self.featuresValues[i*self.n_features + j] = self.featuresFunctions[i*self.n_features + j](signals[i],self.intermValues)
