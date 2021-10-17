"""
Author: Miranda Herman
Created: 2020-10-28
Last Modified: 2021-05-11
Description: Calculates the 6-D log likelihood map for a series of atmospheric
models cross-correlated with planetary emission spectra. Parameters are log VMR, 
day-night contrast, peak phase offset, scaled line contrast, orbital velocity, 
and systemic velocity. 
NOTE: Because this computes the full likelihood map, not MCMC chains, this file 
is very computationally expensive to run when the full parameter grid is used, 
and the output can be multiple Gigabytes. Either run the file on a server that 
can handle this or reduce the ranges and/or stepsizes for the parameter arrays.
"""

from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Gaussian1DKernel
import argparse
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt
import pandas as pd
import time 
import os


def planck(wavelength,temp):
	""" 
	Calculates the Planck function for a given temperature over a
	given wavelength range.
	"""
	c1 = 1.1911e-12
	c2 = 1.439
	y = 1e4/wavelength
	a = c1*(y**5.)
	tmp =  c2*y/temp
	b = np.exp(tmp) - 1.
	bbsor = a/b
	return bbsor


def remove_env(wave, spec, px):
	"""
	Subtracts the lower envelope from a model spectrum by finding 
	the minimum value in the given stepsize, then interpolating.
	"""
	low_wave, low_spec = [], []
	for i in range(len(spec)/px - 1):
		idx = np.nanargmin(spec[i*px:(i+1)*px])
		low_spec.append(spec[idx+i*px])
		low_wave.append(wave[idx+i*px])
	interp = interp1d(low_wave, low_spec, fill_value='extrapolate')
	envelope = interp(wave)
	corrected = spec - envelope
	return corrected


def butterworth(x, order, freq, filt_type='highpass'):
	"""
	Applies a high-pass Butterworth filter, with a given order and 
	cut-off frequency, to the given model.
	"""
	butterfilt = butter(order, freq, btype=filt_type, output='sos')
	x_filtered = sosfiltfilt(butterfilt, x)
	return x_filtered


def wavegrid(wavemin,wavemax,res):
	"""
	Creates a wavelength array evenly spaced in resolution.
	"""
	c=299792458.
	dx=np.log(1.+1./res)
	x=np.arange(np.log(wavemin),np.log(wavemax),dx)
	wavelength=np.exp(x)
	#waveno=1e4/wavelength
	return wavelength #,waveno


def correlate(wave,spec,stdev,vgrid,minwave,maxwave,model_interp):
	"""
	Calculates the cross-correlation map for a given spectral order,
	along with the other two terms of the log likelihood equation:
	the spectra squared, and the base model squared.
	"""
	cmap = np.empty((len(spec),len(vgrid)))
	lnL_term1 = np.empty(len(spec))
	lnL_term2 = np.empty((len(spec),len(vgrid)))

	# Isolate wavelength range and scale data 
	w_idx = (wave[0,:] >= minwave) & (wave[0,:] <= maxwave)

	for frame in range(len(spec)):
		fixspec = spec[frame,w_idx] - np.nanmean(spec[frame,w_idx])
		fixspec /= stdev[frame,w_idx]

		# Calculate data term for log likelihood
		lnL_term1[frame] = np.nansum(fixspec**2)

		for i, vel in enumerate(vgrid):
			# Shift model to desired velocity and scale
			redshift = 1. - vel / 3e5
			shift_wave = wave[0,w_idx] * redshift
			mspec_shifted = model_interp(shift_wave)
			mspec_weighted = mspec_shifted - np.nanmean(mspec_shifted)
			mspec_weighted /= stdev[frame,w_idx]

			# Perform cross-correlation
			corr_top = np.nansum(mspec_weighted * fixspec)
			#corr_bot = np.sqrt(np.nansum(mspec_weighted**2) * np.nansum(fixspec**2))
			cmap[frame,i] = corr_top #/ corr_bot

			# Calculate model term for log likelihood
			lnL_term2[frame,i] = np.nansum(mspec_weighted**2)

	return cmap, lnL_term1, lnL_term2


def submed(cmap):
	"""
	Subtracts the median along the velocity axis from the 
	cross-correlation map.
	"""
	mdn = np.nanmedian(cmap,axis=1)
	sub = cmap - mdn[:,np.newaxis]
	return sub


def phasefold(Kps, vgrid, vsys, cmap, phase, barycor_vect, DumpKpVsys=False):
	"""
	Shifts the cross-correlation map to planet's rest frame and 
	creates the Kp-Vsys map.
	"""
	fmap = np.empty((len(Kps), len(vsys)))
	KTVmap = np.zeros((len(Kps), len(cmap), len(vsys)))

	for i, Kp in enumerate(Kps):
		fullmap = np.empty((len(cmap),len(vsys)))
		for frame in range(len(phase)):            

			# Shift to planet's orbital velocity
			vp = Kp * np.sin(2.*np.pi*phase[frame])
            
			### Check that the definition of vp is correct for my phases by reversing sign with -Kp
			###vp = -Kp * np.sin(2.*np.pi*phase[frame])
            
            
			### vshift = vgrid - vp
			## Attempting to add barycentric correction 
			vshift = vgrid - vp + barycor_vect[frame]
        

			shift = interp1d(vshift, cmap[frame,:], bounds_error=False)
			shifted_map = shift(vsys)
			fullmap[frame,:] = shifted_map

		KTVmap[i] = fullmap
		fmap[i,:] = np.nansum(fullmap, axis=0)
        
# 	if DumpKpVsys:
# 		np.save('DumpKpVsys5.npy',fmap)
        
	
        
    
# 	print('End of phasefold')
        
	return fmap, KTVmap


def chi2(cmap, merr, serr, alpha, Kps, vgrid, vsys, phase, barycor_vect):
    """
    Calculates the chi squared portion of the lnL from the 
    previously computed cross-correlation map and other base 
    terms, for a given set of scaled line contrast values.
    """
    X2 = np.zeros((len(alpha), len(Kps), len(vsys)))	# (alpha, Kps, Vsys)

    # Shift merr and cmap to the planet's velocity, so their axes are (Kp, time, Vsys)
    _, term2_shift = phasefold(Kps, vgrid, vsys, merr, phase, barycor_vect)
    _, term3_shift = phasefold(Kps, vgrid, vsys, cmap, phase, barycor_vect, DumpKpVsys=True)

    # Calculate the log likelihood for each value of alpha	
    for i,a in enumerate(alpha):
        X2_KTV = serr[np.newaxis,:,np.newaxis] + a**2 * term2_shift - 2 * a * term3_shift
    
        # Sum the log likelihood in time
        X2[i] = np.nansum(X2_KTV, axis=1)
        
    #print('End of chi2')


    return X2


def brightvar(phase, offset_deg, contrast):
	"""
	Computes the brightness variation for a given set of day-night 
	contrast and peak phase offset values over a given phase range.
	"""
	offset = offset_deg / 360.
	# Equation: Ap = 1 - C * cos^2 (pi * (phi - theta))
	A_p = 1. - contrast[:,np.newaxis,np.newaxis] * \
		np.cos(np.pi*(phase[np.newaxis,np.newaxis,:] - \
		offset[np.newaxis,:,np.newaxis]))**2
	return A_p


##########################
    
def GetAllResiduals(TargetForCrossCor,night,arm,arm_subpart,ModelScalingFactor,PCA_algorithm,UpToComponents,SysremComponentsRemovedVect):    
    
    if arm == 'vis':
        numorders = 61
    
    if arm == 'nir':
        numorders = 28
    
    ListOfResiduals = []
    
    ### First load the supporting files 
    #mjdLoadDirectory = '../CrossCorrelationDataAndProcessing/%s/ProcessedData/%s/%s/%s'%(DataOrigin,TargetForCrossCor,night,arm)
    mjdLoadDirectory = '%s/%s/ProcessedData/%s/%s/%s'%(FirstPartOfLoadPath,DataOrigin,TargetForCrossCor,night,arm)


    planetrv = np.loadtxt('%s/radv.txt'%(mjdLoadDirectory))
    phase =  np.loadtxt('%s/phase.txt'%(mjdLoadDirectory))
    baryrv = np.loadtxt('%s/BarycentricRVcorrection_kms.txt'%(mjdLoadDirectory))
    mjd = np.loadtxt('%s/mjd.txt'%(mjdLoadDirectory))    
    
    AncillaryData = (phase,planetrv,baryrv,mjd)
    
    for OrderIndex in range(numorders):  

        SysremComponentsRemoved = int(SysremComponentsRemovedVect[OrderIndex])
    
        print('Loading order index %d of %d'%(OrderIndex,numorders))
        
        ####
        
        #LoadDirectory = '../CrossCorrelationDataAndProcessing/%s/ProcessedData/%s/%s/%s/Part%s/ModelScalingFactor%0.5e/ModelShift_%.3e_kms/Order%d'%(DataOrigin,TargetForCrossCor,night,arm,arm_subpart,ModelScalingFactor,ModelShift_kms,OrderIndex)
        LoadDirectory = '%s/%s/ProcessedData/%s/%s/%s/Part%s/ModelScalingFactor%0.5e/ModelShift_%.3e_kms/Order%d'%(FirstPartOfLoadPath,DataOrigin,TargetForCrossCor,night,arm,arm_subpart,ModelScalingFactor,ModelShift_kms,OrderIndex)

        
        ### For the residual file     
        #DataFile = '%s/Flux8a_TelluricFrame_sysremedTo%d_ColWeighted_var.fits'%(LoadDirectory,UpToComponents)       
        DataFile = '%s/Flux7_TelluricFrame_sysremedTo%d.fits'%(LoadDirectory,UpToComponents)       

           
        #DataFileWaveArray = '%s/CroppedWave.fits'%(LoadDirectory)
        #DataFileWaveArray = '%s/Wave_PostSysrem_ShiftedForBaryCorrection.fits'%(LoadDirectory)
        DataFileWaveArray = '%s/Wave4_ToSysrem.fits'%(LoadDirectory)
        
        
        residualsAllSysremsFile = fits.open(DataFile)
        
        residuals = residualsAllSysremsFile[SysremComponentsRemoved].data  
       
        #### If loading Flux7_TelluricFrame_sysremedX.fits, need to subtract the median from each spectrum to get them centered around 0 since that is not done yet 
        MedPerSpec = np.median(residuals,axis=1)
        residuals -= MedPerSpec[:,np.newaxis]         
        
        ## Just in case some nans have slipped through
        residuals[np.isnan(residuals)] = 0.0
    
        ##pyfits.writeto('QuickOutput.fits',residuals,overwrite='True')      
    
        wave = fits.getdata(DataFileWaveArray)
        
        ListOfResiduals.append((wave,residuals))
        
        
    AllOutput = (AncillaryData,ListOfResiduals)        
        
    return AllOutput


###############################################################################


# parser = argparse.ArgumentParser(description="Likelihood Mapping of High-resolution Spectra")
# parser.add_argument("-nights", nargs="*", help="MJD nights", type=str)
# parser.add_argument("-d", '--datapath', default="./", help="path to data")
# parser.add_argument("-m", '--modelpath', default="./", help="path to models")
# parser.add_argument("-o", '--outpath', default="./", help="path for output")
# parser.add_argument("-ext", '--extension', default=".fits", help="output file name extension")
# args = parser.parse_args()

# nights = args.nights
# data_path = args.datapath
# model_path = args.modelpath
# out_path = args.outpath
# ext = args.extension
    
FirstPartOfLoadPath = '../CrossCorrelationDataAndProcessing'
#FirstPartOfLoadPath = 'F:'

DataOrigin = 'KELT-9b_CARMENES_emission_data'

# Define parameter arrays
#vmrs = np.arange(-5., -2.1, 0.1)
#vmrs = np.arange(-5., -2.1, 0.1) #### using VMR as FeH in KELT-9b models 
    
#vmrs = np.arange([-1.0,0.0,1.0]) #### using VMR as FeH in KELT-9b models 
    
#vmrs = ['-1.0','+0.0','+1.0'] #### using VMR as FeH in KELT-9b models 
    
#vmrs = ['+0.0'] #### using VMR as FeH in KELT-9b models 
    
# Species = 'Fe'
# vmrs = [-1.0, 0.0, 1.0, 1.7, 2.0, 2.3]
# fcs = [0.25, 0.5, 0.75, 1.0]
# CtoOs = [0.35, 0.55, 0.7, 0.75, 1.0, 1.5] ### Note that 0.55 is solar 

#### First attempt to try with 4,3,2 to help distinguish which CC dim is which in LnL output 
# Species = 'Fe'
# vmrs = [-1.0, 0.0, 1.0, 1.7]
# fcs = [0.25, 0.5, 0.75]
# CtoOs = [0.55, 0.7] ### Note that 0.55 is solar 
    
###### Trying one vrm per processor:
# Species = 'Fe'
# vmrs = [-1.0, 0.0, 1.0, 1.7, 2.0, 2.3]
# #vmrs = [2.3]
# # fcs = [0.25, 0.5, 0.75, 1.0]
# # CtoOs = [0.35, 0.55, 0.7, 0.75, 1.0, 1.5] ### Note that 0.55 is solar 

# fcs = [0.5]
# CtoOs = [0.55] ### Note that 0.55 is solar 

# alpha = np.arange(0.5, 5., 0.1)

# ### Set offset and contrast to lenght 1 to make it faster and help judge file size 

# #offset = np.arange(-60.0,60.0, 2.)
# #offset = np.array([0.0])
# offset = np.arange(-180.0,200,20.0)

# #contrast = np.arange(0.,1.1, 0.1)
# #contrast = np.arange(0.,1.2, 0.2)
# contrast = np.arange(0.,1.25, 0.25)

# #contrast = np.array([0.0])

# vgrid = np.arange(-600.0,601.0,1.0)
# Vsys = np.arange(-150., 151., 1)
# Kps = np.arange(240.0-25,240.0+25+1,1)

############## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
### For making a 3.2 GB file with offsets +/- 180 in steps of 20
### file name: 618_offsetPm180.fits
# Species = 'Fe'
# vmrs = [-1.0, 0.0, 1.0, 1.7, 2.0, 2.3]
# fcs = [0.5]
# CtoOs = [0.55] ### Note that 0.55 is solar 
# alpha = np.arange(0.5, 5., 0.1)
# offset = np.arange(-180.0,200,20.0)
# contrast = np.arange(0.,1.25, 0.25)
# vgrid = np.arange(-600.0,601.0,1.0)
# Vsys = np.arange(-150., 151., 1)
# Kps = np.arange(240.0-25,240.0+25+1,1)

### For making files with only Goyal dimensions (0.795 GB)
# Species = 'Fe'
# #vmrs = [-1.0, 0.0, 1.0, 1.7, 2.0, 2.3]
# vmrs = [2.3]

# fcs = [0.25, 0.5, 0.75, 1.0]
# CtoOs = [0.35, 0.55, 0.7, 0.75, 1.0, 1.5] ### Note that 0.55 is solar 

# alpha = np.arange(0.5, 5., 0.1)

# offset = np.array([0.0])
# contrast = np.array([0.0])

# vgrid = np.arange(-600.0,601.0,1.0)
# Vsys = np.arange(-150., 151., 1)
# Kps = np.arange(240.0-25,240.0+25+1,1)


########################

### Values for making a LnL file with a different extent in each dimension 
### to make it easier to check that the dimensions are working out 


### For making files with only Goyal dimensions (0.795 GB)
# Species = 'Fe'
# vmrs = [-1.0, 0.0, 1.0, 1.7, 2.0, 2.3]
# #vmrs = [2.3]

# fcs = [0.25, 0.5, 0.75, 1.0]
# #CtoOs = [0.35, 0.55, 0.7, 0.75, 1.0, 1.5] ### Note that 0.55 is solar 
# CtoOs = [0.35, 0.55, 0.75, 1.0, 1.5]

# #alpha = np.arange(0.5, 5., 0.1)
# alpha = np.arange(0.5, 5.5, 0.5)

# offset = np.array([-30.0,0.0,30.0])
# contrast = np.array([0.0, 1.0])

# vgrid = np.arange(-600.0,601.0,1.0)
# Vsys = np.arange(-150., 151., 1)
# Kps = np.arange(240.0-25,240.0+25+1,1)



### 
### Miranda's original values 
# alpha = np.arange(0.5, 5., 0.1)
# vgrid = np.arange(-600.,601.5, 1.5)
# Vsys = np.arange(-150., 150., 0.5)
# Kps = np.arange(175.,275., 0.5)
# offset = np.arange(-30.,60., 1.)
# contrast = np.arange(0.,1.1, 0.1)


### For the phasefolding:
###    vgrid - Kp needs to have a large overlap with Vsys 
### because it is interpolated onto  


# alpha = np.array([0.5,1.0,2.0])
# offset = np.array([-30.0,0.0,30.0])
# contrast = np.array([0.0,0.5,1.0])


#Kps = np.arange(90,391,5.0)

#####################

# Species = 'Fe'
# vmrs = [-1.0, 0.0, 1.0, 1.7, 2.0, 2.3]
# #vmrs = [2.3]

# fcs = [0.25, 0.5, 0.75, 1.0]
# CtoOs = [0.35, 0.55, 0.7, 0.75, 1.0, 1.5] ### Note that 0.55 is solar 

# #alpha = np.arange(0.5, 5., 0.1)
# #alpha = np.arange(0.1, 1.1, 0.1)
# #alpha = np.arange(0.5, 1.5, 0.1)
# alpha = np.arange(0.1, 5.0, 0.2)

# #offset = np.array([-90.0, -45.0, 0.0, 45.0 ,90.0])
# offset = np.arange(-180.0,180+45,45)

# contrast = np.array([0.0,0.25,0.5,0.75,1.0])

# vgrid = np.arange(-600.0,601.0,1.0)
# Vsys = np.arange(-150., 151., 1)

# Kps = np.arange(240.0-25,240.0+25+1,1)

### #####
### Revised grid with low zero alpha

Species = 'Fe'
vmrs = [-1.0, 0.0, 1.0, 1.7, 2.0, 2.3]
#vmrs = [2.3]

fcs = [0.25, 0.5, 0.75, 1.0]
CtoOs = [0.35, 0.55, 0.7, 0.75, 1.0, 1.5] ### Note that 0.55 is solar 

#alpha = np.arange(0.5, 5., 0.1)
#alpha = np.arange(0.1, 1.1, 0.1)
#alpha = np.arange(0.5, 1.5, 0.1)
#alpha = np.arange(0.1, 5.0, 0.2)
alpha = np.arange(0.0, 4.505, 0.05)

#offset = np.array([-90.0, -45.0, 0.0, 45.0 ,90.0])
#offset = np.arange(-180.0,180+45,45)
#offset = np.arange(-180.0,180,30)
offset = np.arange(-180.0,210,30)

contrast = np.array([0.0,0.25,0.5,0.75,1.0])

vgrid = np.arange(-600.0,601.0,1.0)
#Vsys = np.arange(-20-30, -20+30, 1)
Vsys = np.arange(-20-28, -20+28, 1)

Kps = np.arange(240.0-25,240.0+25+1,1)









#lnL = np.zeros((len(vmrs),len(fcs),len(CtoOs),len(contrast), len(offset), len(alpha), len(Kps), len(Vsys)))


lnL = np.zeros((len(vmrs),len(fcs),len(CtoOs),len(contrast), len(offset), len(alpha), len(Kps), len(Vsys)))
####nights = ['20190528All']
#nights = ['20190528P2']
nights = ['20180618All']
night = nights[0]

out_path = '%s/%s/LLOutput/%s/%s/'%(FirstPartOfLoadPath,DataOrigin,night,Species)

# lnL = np.memmap('%s/%s_memmapped.dat'%(out_path,night), dtype=np.float64,
#               mode='w+', shape=(len(vmrs),len(fcs),len(CtoOs),len(contrast), len(offset), len(alpha), len(Kps), len(Vsys)))






# # Specify number of SYSREM iterations used on spectra for each MJD night
# iters = {'56550': 5, '56561': 4, '56904': 4, '56915': 6, '56966': 6}

# # Specify Butterworth filter cut-off frequency for each night
# bfreq = {'56550': 0.035, '56561': 0.04, '56904': 0.03, '56915': 0.025, '56966': 0.055}

#nights = ['20180609All','20180618All','20190528All','20190604All']



ModelShift_kms = 0.0

TargetForCrossCor = 'NoModel'
ModelScalingFactor = 0.0

PCA_algorithm = 'sysremed'
#UpToComponents = 15
UpToComponents = 10



#nights = ['20180618All']



SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig6.csv')

# OrdersToExcludeList = [52, 53, 54, 56, 57, 58, 59, 60, 61, 
#                         69, 70, 71, 73, 78, 79, 80, 81, 
#                         82, 83, 84, 91, 97, 98, 99, 100,
#                         101, 106, 107, 108, 109, 110]


if night == '20190528P2':    
    OrdersToExcludeList = [52,53,54,56,57,58,59,60,69,70,71,72,73,78,79,80,
                            81,82,83]

    # ### To only inlcude good orders from VIS arm 
    # ListOfNIROrders = list(np.arange(61,89))
    
    # OrdersToExcludeList = [52,53,54,56,57,58,59,60]+ListOfNIROrders
    
    
    
if night == '20180618All':    
    OrdersToExcludeList = [38,52,53,54,56,57,58,59,60,62,69,70,71,72,73,78,79,80,
                           81,82,83,84,85,88]




for night in nights:
    
    # for ArmSubpartIndex in range(len(ArmSubpartList)):
    
    #     arm = ArmSubpartList[ArmSubpartIndex][0]
    #     arm_subpart = ArmSubpartList[ArmSubpartIndex][1]
        
    #     n_orders = 61  ### only for VIS at the moment. If using NIR as well it will be 28
    
    VisANumberOfSysremIterationsVect = SysremItConfig_df['%s_%s_%s'%(night,'vis','A')].values  
    NirANumberOfSysremIterationsVect = SysremItConfig_df['%s_%s_%s'%(night,'nir','A')].values  
    ##NirBNumberOfSysremIterationsVect = SysremItConfig_df['%s_%s_%s'%(night,'nir','B')].values  



        
        
	# Read in data
	#specPerOrder = np.load(data_path+night+'_spectra.npy')[iters[night]-1] - 1.		# (orders, frames, pixels)
    VISAInfoAndResdiaulsPerOrder = GetAllResiduals(TargetForCrossCor,night,'vis','A',ModelScalingFactor,PCA_algorithm,UpToComponents,VisANumberOfSysremIterationsVect)
    NIRAInfoAndResdiaulsPerOrder = GetAllResiduals(TargetForCrossCor,night,'nir','A',ModelScalingFactor,PCA_algorithm,UpToComponents,NirANumberOfSysremIterationsVect)
    ###NIRBInfoAndResdiaulsPerOrder = GetAllResiduals(TargetForCrossCor,night,'nir','B',ModelScalingFactor,PCA_algorithm,UpToComponents,NirBNumberOfSysremIterationsVect)
     
    phase = VISAInfoAndResdiaulsPerOrder[0][0]		# (frames)
    barycor_vect = VISAInfoAndResdiaulsPerOrder[0][2]

    
# 	wave = np.load(data_path+night+'_wavelength.npy')				# (orders, frames, pixels)
# 	phase = np.load(data_path+night+'_phase.npy')					# (frames)
    
    wave = []
    spec = [] 
    
    for i in range(len(VISAInfoAndResdiaulsPerOrder[1])):
        wave.append(VISAInfoAndResdiaulsPerOrder[1][i][0])
        spec.append(VISAInfoAndResdiaulsPerOrder[1][i][1])
        
    for i in range(len(NIRAInfoAndResdiaulsPerOrder[1])):
        wave.append(NIRAInfoAndResdiaulsPerOrder[1][i][0])
        spec.append(NIRAInfoAndResdiaulsPerOrder[1][i][1])
        
    # for i in range(len(NIRBInfoAndResdiaulsPerOrder[1])):
    #     wave.append(NIRBInfoAndResdiaulsPerOrder[1][i][0])
    #     spec.append(NIRBInfoAndResdiaulsPerOrder[1][i][1])
        

	
# 	# Only include phases below 0.41 and above 0.59, to avoid stellar Fe signal
# 	p_ind = np.where((phase < 0.41) & (phase > -0.41))[0]
# 	phase = phase[p_ind]
# 	spec = spec[:,p_ind,:]
# 	wave = wave[:,p_ind,:]
		
	# Determine size of arrays
# 	n_orders = spec.shape[0]

    n_frames = spec[0].shape[0]
    ##n_pix = spec.shape[2]
    StartTime = time.time()    
    
    
    for FcIndex, fc in enumerate(fcs):
        for v, vmr in enumerate(vmrs):
            for CtoOIndex, CtoO in enumerate(CtoOs):           
                
                if vmr < 0:
                    ModelDescription = '%.2f_%.1f_%.2f'%(fc, vmr, CtoO)
                
                if vmr >= 0:
                    ModelDescription = '%.2f_+%.1f_%.2f'%(fc, vmr, CtoO)
                
                # Get dayside model
                #hdu = fits.open(model_path+'model_wasp33b_FeI_logvmr%.1f.fits' % (vmr))		# (wavelength, spectrum)
                #model = hdu[0].data
		
                # Interpolate model to wavelength grid with consistent resolution
                #m_wave = wavegrid(model[0,0], model[0,-1], 3e5)
                #wv_interp = interp1d(model[0],model[1], kind='linear', fill_value=0, bounds_error=False)
                #m_spec = wv_interp(m_wave)

                # Convolve model with 1D Gaussian kernel, then filter
                #FWHM_inst = {'CFHT': 4.48, 'Subaru': 1.8}
                #mspec_conv = convolve(m_spec, Gaussian1DKernel(stddev=FWHM_inst['CFHT']/2.35))
                #mspec_day = remove_env(m_wave,mspec_conv, 250) 
                #mspec_bf = butterworth(mspec_conv, 1, bfreq[night])
                
                # Create interpolator to put model onto data's wavelength grid
                ### put in here a flattend model with wavelength in A since the data is in A                 
                

                model = np.load('%s/%s/ModelSpectraLLFromLoop/ContSub_K9b_%s_%s_Vrot6.63_CarmenesRes_pRT_flux_per_Hz.npy.npy'%(FirstPartOfLoadPath,DataOrigin,Species,ModelDescription))
                                
                m_wave = model[:,0]*1e4
                mspec_bf = model[:,1]
                                
                filt_interp = interp1d(m_wave, mspec_bf, kind='linear', fill_value=0.,bounds_error=False)
        

                # Create variables/arrays for lnL components
                N = 0.
                cmap_osum = np.zeros((n_frames, len(vgrid)))
                merr_osum = np.zeros((n_frames, len(vgrid)))
                serr_osum = np.zeros((n_frames))
                
                NumOrders = len(spec)
                cmap_PerOrder = np.zeros((n_frames, len(vgrid),NumOrders))
                merr_PerOrder = np.zeros((n_frames, len(vgrid),NumOrders))
                serr_PerOrder = np.zeros((n_frames,NumOrders))
                
        
        
                #NumVisOrders = 10
                #NumVisOrders = 1
        
                ccStartTime = time.time()
                ##### Perform cross-correlation for orders redward of 600 nm, and sum together
                ##for i,o in enumerate(np.arange(24,37)): 
                
                ListOfOrdersToDo = []
                
                ListOfAllOrders = np.arange(len(spec))
                
                for i in ListOfAllOrders:
                    if i not in OrdersToExcludeList:
                        ListOfOrdersToDo.append(i)

      
                for i,o in enumerate(ListOfOrdersToDo):  
                    
                    print('Doing FcIndex index %d of %d: order index %d of %d'%(FcIndex,len(fcs),i,len(ListOfOrdersToDo)))
                    print('Doing vmr index %d of %d: order index %d of %d'%(v,len(vmrs),i,len(ListOfOrdersToDo)))
                    print('Doing CtoO index %d of %d: order index %d of %d'%(FcIndex,len(CtoOs),i,len(ListOfOrdersToDo)))
                    
                    # Calculate time- and wavelength-dependent uncertainties
                    tsigma = np.nanstd(spec[o], axis=0)
                    wsigma = np.nanstd(spec[o], axis=1)
                    sigma = np.outer(wsigma, tsigma)
                    #sigma /= np.nanstd(spec[o,:,:])
                    sigma /= np.nanstd(spec[o])
                    sigma[((sigma < 0.0005) | np.isnan(sigma))] = 1e20
        
                    # Calculate number of data points in spectra
                    # minwave, maxwave = np.nanmin(wave[o,:,:]), np.nanmax(wave[o,:,:])
                    # minwidx, maxwidx = np.nanargmin(wave[o,0,:]), np.nanargmax(wave[o,0,:])
                    # N += len(wave[o,0,minwidx:maxwidx]) * len(phase)
                    
                    WavePerOrder = wave[o]
                    SpecPerOrder = spec[o]
                        

            
                    minwave, maxwave = np.nanmin(WavePerOrder), np.nanmax(WavePerOrder)
                    minwidx, maxwidx = np.nanargmin(WavePerOrder), np.nanargmax(WavePerOrder)
                    N += len(WavePerOrder[0,minwidx:maxwidx]) * len(phase)
        
                    # Perform cross-correlation
                    #cmap0, serr, merr = correlate(wave[o,:,:], spec[o,:,:], sigma, vgrid, minwave, maxwave, filt_interp)
                    
                    
                    SummedCCFOutputPath = '%s/%s/LLOutput/SummedCCFs/%s/%s/%s'%(FirstPartOfLoadPath,DataOrigin,night,Species,ModelDescription)
                    
                    if not os.path.exists('%s/cmap_osum.npy'%(SummedCCFOutputPath)):                      
                    
                        cmap0, serr, merr = correlate(WavePerOrder, SpecPerOrder, sigma, vgrid, minwave, maxwave, filt_interp)
    
                        cmap = submed(cmap0)
                
                        cmap_osum +=  cmap
                        merr_osum +=  merr
                        serr_osum += serr
                
                if not os.path.exists('%s/cmap_osum.npy'%(SummedCCFOutputPath)):         
                    os.makedirs(SummedCCFOutputPath)
                
                    np.save('%s/cmap_osum'%(SummedCCFOutputPath),cmap_osum)
                    np.save('%s/merr_osum'%(SummedCCFOutputPath),merr_osum)
                    np.save('%s/serr_osum'%(SummedCCFOutputPath),serr_osum)
                    
                else: 
                    print('Loading: %s'%(SummedCCFOutputPath))
                    
                    cmap_osum = np.load('%s/cmap_osum.npy'%(SummedCCFOutputPath))
                    merr_osum = np.load('%s/merr_osum.npy'%(SummedCCFOutputPath))
                    serr_osum = np.load('%s/serr_osum.npy'%(SummedCCFOutputPath))
                
            
                ccEndTime = time.time()
                print('Cross-correlations took %.2f mins'%((ccEndTime-ccStartTime)/60))
        
                # Compute brightness variation for given contrasts and offsets
                variation = brightvar(phase, offset, contrast)
		
                # Apply brightness variation to lnL terms
                lnL_term1 = serr_osum
                lnL_term2 = merr_osum[np.newaxis,np.newaxis,:,:] * variation[:,:,:,np.newaxis]**2
                lnL_term3 = cmap_osum[np.newaxis,np.newaxis,:,:] * variation[:,:,:,np.newaxis]

                chi2StartTime = time.time()
                print('Starting chi2 calc')
                # Calculate lnL for given VMR
                for i in range(len(contrast)):
                    for j in range(len(offset)):
                        X2 = chi2(lnL_term3[i,j], lnL_term2[i,j], lnL_term1, alpha, Kps, vgrid, Vsys, phase, barycor_vect)
                        
                        if N == 0.0:
                            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            print('About to divide by N = 0')
                            
                       
                        lnL[v,FcIndex,CtoOIndex,i,j] += -N/2. * np.log(X2 / N)
                chi2EndTime = time.time()
                print('Chi 2 took %.2f hr'%((chi2EndTime-chi2StartTime)/3600))

# Find highest likelihood values
maximum = np.nanmax(lnL)
maxes = np.where(lnL == maximum)
fidx = maxes[0][0]

fcindex = maxes[1][0]
CtoOIndex = maxes[2][0]

cidx = maxes[3][0]
oidx = maxes[4][0]
aidx = maxes[5][0]
kidx = maxes[6][0]
vidx = maxes[7][0]

# Print highest likelihood values
print('Location of highest likelihood:')
#print('logVMR = %.1f' % (vmrs[fidx]))
print('FeH = %s' % (vmrs[fidx]))

print('Fc = %s' % (fcs[fcindex]))
print('C/O = %s' % (CtoOs[CtoOIndex]))

print('C = %.1f' % (contrast[cidx]))
print('off = %.1f' % (offset[oidx]))
print('a = %.1f' % (alpha[aidx]))
print('Kp = %.1f' % (Kps[kidx]))
print('Vsys = %.1f' % (Vsys[vidx]))

# Write lnL to fits file




if not os.path.exists(out_path):
    os.makedirs(out_path)

# #SaveFileName = 'lnL_KELT-9b_Fe_V3.fits'
# #SaveFileName = 'Night618_SameOrders528_lnL_KELT-9b_Fe_6FeH.fits'
# #SaveFileName = '618_OffsetPm60_lnL_KELT-9b_Fe_6FeH.fits'

#SaveFileName = 'Trying3DimsOfModels1VMRPerProc.fits'
    
#SaveFileName = '528_offsetPm180.fits'
    
#SaveFileName = '618_offsetPm180.fits'
 
#SaveFileName = '528P2_making_only_vis_summed_CCFs.fits'

#SaveFileName = '618All_making_vis_and_nir_summed_CCFs.fits'
    
#SaveFileName = '528P2_checking_dims.fits'

#SaveFileName = '618_Goyal_dimension.fits'

#SaveFileName = '528P2_CombNirV6.fits'
SaveFileName = '618All_CombNirV6.fits'



hdu2 = fits.PrimaryHDU(lnL)
hdu2.writeto(out_path+SaveFileName, overwrite=True)


#np.save('%s/%s/LLOutput/CompareSizeV3Fits.npy'%(FirstPartOfLoadPath,DataOrigin),lnL)

# lnL.flush()




EndTime = time.time()    

TimeTaken_hr = (EndTime - StartTime)/3600

print('Total time taken: %.2f hr'%(TimeTaken_hr))
