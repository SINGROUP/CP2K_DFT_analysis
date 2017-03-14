#!/usr/bin/python
#
# Compute total and l-projected DOS from CP2K dosfile. Apply Gaussian
# broadening instead of sharp histogram if desired.
#
# NB: Assumes CP2K output DOS files named *dosfile* are in the
# directory where you run the script.
#
# NB: Assumes a non-spin-polarized calculation.
#
# NB: Shifts the energy scale so that E_Fermi = 0.0 eV.
#
# Eero Holmstrom, 2016
#

import numpy as np
import sys
import os

# define some variables
lChannels = ('s', 'p', 'd', 'f')
totalDOSComputed = False

# usage
if len(sys.argv) < 4:
    print("Usage: %s [E1] [E2] [dE] [sigma (in eV)]" % (sys.argv[0]))
    exit(1)

# parse input arguments
doGaussian = False
E1 = float(sys.argv[1])
E2 = float(sys.argv[2])
dE = float(sys.argv[3])
if len(sys.argv) >= 5:
    sigma = float(sys.argv[4])
    doGaussian = True
    print("\nRead in the following parameters:\n\nE1 = %f\nE2 = %f\ndE = %f\nsigma = %f eV" % (E1, E2, dE, sigma))
else:
    print("\nRead in the following parameters:\n\nE1 = %f\nE2 = %f\ndE = %f" % (E1, E2, dE))

# find dos files from directory
dircontents = os.listdir('.')
dosfiles = [file for file in dircontents if 'dosfile' in file]
dosfiles.sort()
print("\nFound the following dos files:\n")
for file in dosfiles:
    print("%s" % file)

# let the user know which method of computing the DOS you'll use
if doGaussian is True:
    print("\nUsing Gaussian broadening to compute the DOS.")
else:
    print("\nUsing direct histogram approach to compute the DOS.")

# go through each dos file. for each one, go through each angular
# momentum channel.

for file in dosfiles:

    print("\nNow processing file %s...\n" % file)
    
    # read in entire file
    thisFile = open(file, "r")
    data = thisFile.readlines()
    nlines = len(data)
    print("Read in a total of %d lines." % nlines)
    
    # find chemical species of atom

    if 'list' in file:
        chemType = 'list'
    else:
        chemType = data[0].split(" ")[6]
        print("Found chemical type %s." % chemType)

    # find Fermi level in eV
    EFermi = np.float_(data[0].split(" ")[-2])*27.211
    print("Found Fermi energy of %f eV." % EFermi)

    # find number of angular momentum channels
    nlchannels = len(filter(None, data[1].strip("").strip("\n").split(" "))) - 5
    print("Found %d angular momentum channels.\n" % nlchannels)

    # read energy vs. DOS data for all components into a single matrix
    # of eV vs. DOS data
    dosData = []
    for i in range(2, nlines):
        dosData.append(filter(None, data[i].strip(" ").strip("\n").split(" "))[1:])
    dosData = np.float_(np.array(dosData))
    # convert energy axis from Hartree to eV
    dosData[:,0] = 27.211*dosData[:,0]

    #
    # compute total DOS
    #
    if totalDOSComputed is False:
        
        print("Computing total DOS...")

        # do either gaussian smearing or traditional histogram approach
        if doGaussian is True:

            thisDOSEnergy = np.linspace(E1, E2, int(E2-E1)/dE)
            thisDOSHist = np.zeros([len(thisDOSEnergy), 1])
            j = 0
            for thisE in thisDOSEnergy:
                for eps in dosData[:, 0]:
                    thisDOSHist[j] = thisDOSHist[j] + 1.0/(np.sqrt(2*np.pi)*sigma)*np.exp(-(thisE-eps)**2 / (2*sigma**2))
                j = j + 1
            thisDOSEnergy = thisDOSEnergy[np.newaxis].T - EFermi
            thisDOS = np.hstack((thisDOSEnergy, thisDOSHist))
            print("Done. Saving to file pydos_total.dat.\n")
            np.savetxt('pydos_total.dat', thisDOS)

        else:

            thisDOSHist, thisDOSEnergy = np.histogram(dosData[:, 0], range=(E1,E2), bins=int((E2-E1)/dE))
            thisDOSEnergy = (thisDOSEnergy[0:-1] + thisDOSEnergy[1:])/2.0
            thisDOSEnergy = thisDOSEnergy - EFermi
            thisDOS = np.vstack((thisDOSEnergy, thisDOSHist)).T
            print("Done. Saving to file pydos_total.dat.\n")
            np.savetxt('pydos_total.dat', thisDOS)

    totalDOSComputed = True

    #
    # compute projected DOS
    #
    for i in range(0, nlchannels):

        # do gaussian smearing or traditional histogram approach
        if doGaussian is True:

            print("Now processing channel %s..." % lChannels[i])
            thisDOSEnergy = np.linspace(E1, E2, int(E2-E1)/dE)
            thisDOSHist = np.zeros([len(thisDOSEnergy), 1])
            j = 0
            for thisE in thisDOSEnergy:
                for (eps, weight) in zip(dosData[:, 0], dosData[:, 2+i]):
                    thisDOSHist[j] = thisDOSHist[j] + 1.0/(np.sqrt(2*np.pi)*sigma)*np.exp(-(thisE-eps)**2 / (2*sigma**2))*weight
                j = j + 1
        else:

            print("Now processing channel %s..." % lChannels[i])
            thisDOSHist, thisDOSEnergy = np.histogram(dosData[:, 0], weights=dosData[:, 2+i], range=(E1,E2), bins=int((E2-E1)/dE))
            thisDOSHist = thisDOSHist[np.newaxis].T
            thisDOSEnergy = (thisDOSEnergy[0:-1] + thisDOSEnergy[1:])/2.0

        # combine results into single array
        thisDOSEnergy = thisDOSEnergy[np.newaxis].T - EFermi
        thisDOS = np.hstack((thisDOSEnergy, thisDOSHist))
        # save to aptly named file
        thisOutputFile = 'pydos_' + chemType + '_' + lChannels[i] + '.dat'
        print("Done. Saving to file %s." % thisOutputFile)
        np.savetxt(thisOutputFile, thisDOS)
    
    print("\nDone with file %s." % file)

print("\nAll done. Exiting.\n")
exit(0)
