# Condensation

Extracting anomalous fragments from a WAV file.

## Description

This folder 

In order to speed up annotation of raw audio data we 

To speed up the annotation of the sanctuary data we ‘condensate’ the data with an energy/change based automatic vocalization detection. The detection comprises a Short-time Fourier transform to produce a power distribution in the time-frequency domain. Depending on the properties of the expected primate vocalizations we discard redundant frequency bands. From the remaining bands we collect small time-intervals in which the registered signal loudness exceeds a species specific threshold, or in which the local cumulative power distribution deviates from a global counterpart. This collection represents a set of timestamps where we expect to hear disruptions in the ambient noise. The time-intervals are used to extract the corresponding signal fragments from our raw data. These fragments are bundled into a new audio file which a resulting high density of vocalizations that can be annotated more efficiently.

