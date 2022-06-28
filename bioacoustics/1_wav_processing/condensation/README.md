# Condensation

## Introduction

To speed up the annotation process of the large amount of recordings we have created a script that tries to detect and gather audio fragments that stand out with respect to the background noise of the jungle. These fragments _might_ contain Chimpanzee vocalizations. By applying this script we were able to reduce the workload of annotators: instead of listening to many hours of recordings we could present a drastically 'condensed' version of the data in order to speed up the annotation process.

The detection comprises a Short-time Fourier transform to produce a power distribution in the time-frequency domain. Depending on the properties of the expected primate vocalizations we discard redundant frequency bands. From the remaining bands we collect small time-intervals in which the registered signal loudness exceeds a species specific threshold, or in which the local cumulative power distribution deviates from a global counterpart. This collection represents a set of timestamps where we expect to hear disruptions in the ambient noise. The time-intervals are used to extract the corresponding signal fragments from our raw data. These fragments are bundled into a new audio file which a resulting high density of vocalizations that can be annotated more efficiently.

## Usage

In this folder you find 2 Python files, a shell script and a folder containing one WAV file for testing:

* The extractor.py module contains a class that analyses WAV data as described above;
* The condensate.py script applies the Extractor class and facilitates
