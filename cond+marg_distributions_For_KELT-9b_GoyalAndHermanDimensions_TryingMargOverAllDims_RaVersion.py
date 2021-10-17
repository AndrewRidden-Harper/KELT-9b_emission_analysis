"""
Author: Miranda Herman
Created: 2020-11-20
Last Modified: 2021-05-11
Description: Uses the log likelihood output of logL.py to calculate the 
conditional and marginalized likelihood distributions (mostly useful for 
creating plots). Also calculates the constrained value, uncertainty, and 
significance for each parameter. This calculation is separate from logL.py 
so that the likelihood maps from multiple datasets can be combined before 
computing the parameter constraints.
"""

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt 
from matplotlib.gridspec import  GridSpec
import os 
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


import matplotlib as mpl
mpl.use('Agg')

import matplotlib 
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 

def AddSubplotWithColorbar(FigureObject,AxisObject,CondArrayToPlot):
    
    im = AxisObject.imshow(CondArrayToPlot,aspect='auto',origin='lower',interpolation='none')
    
    divider = make_axes_locatable(AxisObject)
    
    cax = divider.append_axes('right', size='5%', pad=0.05)

    fig.colorbar(im, cax=cax, orientation='vertical')
    
    return None


def significance(param,margL):
	"""
	Calculates the median, 1-sigma uncertainties, and significance
	for a given parameter based on its marginalized likelihood
	distribution.
	"""
	cdf = np.cumsum(margL)
	cdf_norm = cdf/cdf[-1]
	interp_cdf = interp1d(cdf_norm, param)
	low1sig, x_med, up1sig = interp_cdf(np.array([0.5-0.68*0.5, 0.5, 0.5+0.68*0.5]))
	uncert_low, uncert_up = x_med-low1sig, up1sig-x_med
	snr_low, snr_up = x_med/(x_med-low1sig), x_med/(up1sig-x_med)
	
	if snr_low > snr_up:
		return x_med, uncert_low, uncert_up, snr_up
	else:
		return x_med, uncert_low, uncert_up, snr_low


def lowerlim(param,margL):
	"""
	Calculates the 1-sigma lower limit for a bounded parameter
	based on its marginalized likelihood distribution.
	"""
	cdf = np.cumsum(margL)
	cdf_norm = cdf/cdf[-1]
	interp_cdf = interp1d(cdf_norm, param)
	lowlim = interp_cdf(np.array([0.68]))
	return lowlim


###############################################################################

# Open lnL file and compute likelihood
# path = './'
# lnL = fits.open(path+'lnL_wasp33b_FeI.fits')[0].data
    
##path = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput'
#lnL = fits.open('%s/lnL_KELT-9b_FeV1.fits'%(path))[0].data

#lnL = fits.open('%s/lnL_KELT-9b_Fe_TryingToSeeKpVys.fits'%(path))[0].data

#lnL = fits.open('%s/lnL_KELT-9b_Fe_V3.fits'%(path))[0].data

# #night = '20190528All'
# night = '20180618All'    

# SpeciesName = 'Fe'
# path = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/%s/%s'%(night,SpeciesName)





### Night 528 
#lnL = fits.open('%s/lnL_KELT-9b_Fe_6FeH.fits'%(path))[0].data

#lnL = fits.open('%s/Night618_SameOrders528_lnL_KELT-9b_Fe_6FeH.fits'%(path))[0].data

### Attempting to combine both 618 and 528
#lnL = fits.open('%s/Night618_SameOrders528_lnL_KELT-9b_Fe_6FeH.fits'%(path))[0].data + fits.open('%s/lnL_KELT-9b_Fe_6FeH.fits'%(path))[0].data


#lnL = fits.open('%s/528_OffsetPm60_lnL_KELT-9b_Fe_6FeH.fits'%(path))[0].data 

#lnL = fits.open('%s/618_OffsetPm60_lnL_KELT-9b_Fe_6FeH.fits'%(path))[0].data 

#lnL = fits.open('%s/528_OffsetPm60_lnL_KELT-9b_Fe_6FeH.fits'%(path))[0].data 
#lnL += fits.open('%s/618_OffsetPm60_lnL_KELT-9b_Fe_6FeH.fits'%(path))[0].data 


#lnL = fits.open('%s/Trying3DimsOfModels.fits'%(path))[0].data 
#lnL = fits.open('Trying3DimsOfModels.fits')[0].data 
#lnL = fits.open('528_offsetPm180.fits')[0].data 
#lnL = fits.open('%s/528_offsetPm180.fits'%(path))[0].data 

#lnL = fits.open('%s/618_offsetPm180.fits'%(path))[0].data 

################ Load 618 and 528 for Miranda dims 

# SpeciesName = 'Fe'    

# night = '20190528All'
# path = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/%s/%s'%(night,SpeciesName)

# lnL = fits.open('%s/528_offsetPm180.fits'%(path))[0].data 

# night = '20180618All'
# path = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/%s/%s'%(night,SpeciesName)

# lnL += fits.open('%s/618_offsetPm180.fits'%(path))[0].data 
# #lnL = fits.open('%s/618_offsetPm180.fits'%(path))[0].data 
    
### Load file for only Goyal dims    

# SpeciesName = 'Fe'    

# night = '20190528All'
# path = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/%s/%s'%(night,SpeciesName)

# lnL = fits.open('%s/528_Goyal_dimension.fits'%(path))[0].data 

# SpeciesName = 'Fe'    

# night = '20180618All'
# path = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/%s/%s'%(night,SpeciesName)

# lnL = fits.open('%s/618_Goyal_dimension.fits'%(path))[0].data 
# #lnL += fits.open('%s/618_Goyal_dimension.fits'%(path))[0].data 





#####################

# SpeciesName = 'Fe'    

# night = '20190528P2'

# lnL = fits.open('528P2_checking_dims.fits')[0].data 

# 528P2_CombNir.fits


##############################

SpeciesName = 'Fe'    

night = '20180618All'
path = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/%s/%s'%(night,SpeciesName)
#lnL = fits.open('%s/618All_CombNirV4.fits'%(path))[0].data 
#lnL = fits.open('%s/618All_CombNirV5.fits'%(path))[0].data 
lnL = fits.open('%s/618All_CombNirV6.fits'%(path))[0].data 
# lnL = fits.open('%s/618All_CombNirV4_MaxSub.fits'%(path))[0].data 
#lnL = fits.open('%s/618All_CombNirV4.fits'%(path))[0].data #- (-865769.37271484)

# path = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/%s/%s/GridV6'%(night,SpeciesName)
# if not os.path.exists(path):
#     os.makedirs(path)



night = '20190528P2'
path = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/%s/%s'%(night,SpeciesName)
#lnL += fits.open('%s/528P2_CombNirV4.fits'%(path))[0].data 
lnL += fits.open('%s/528P2_CombNirV6.fits'%(path))[0].data 
#lnL += (fits.open('%s/528P2_CombNirV4.fits'%(path))[0].data - (-636854.21327769))
#lnL += fits.open('%s/528P2_CombNirV4_MaxSub.fits'%(path))[0].data

# path = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/%s/%s/GridV6'%(night,SpeciesName)
# if not os.path.exists(path):
#     os.makedirs(path)


# lnL = fits.open('%s/528P2_CombNirV4.fits'%(path))[0].data 

# lnL -= np.max(lnL)
# fits.writeto('%s/618All_CombNirV4_MaxSub.fits'%(path),lnL)

# raise Exception

path = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/%s/%s/V6'%('618_and_528',SpeciesName)
if not os.path.exists(path):
    os.makedirs(path)


print('Loaded LnL')

maximum = np.nanmax(lnL)

maxes = np.where(lnL == maximum)

lnL -= maximum
L = np.exp(lnL)


print('Shape of L:')
print(L.shape)
print()


f = open("%s/maxes.txt"%(path), "a")
for i in range(len(maxes)):
    f.write('%s'%(maxes[i]))
f.close()

print('maxes')
print(maxes)

fidx = maxes[0][0]

Ridx = maxes[1][0]
Tidx = maxes[2][0]

cidx = maxes[3][0]
oidx = maxes[4][0]
aidx = maxes[5][0]
kidx = maxes[6][0]
vidx = maxes[7][0]

# print('kidx')
# print(kidx)

## lnL = np.zeros((len(vmrs),len(fcs),len(CtoOs),len(contrast), len(offset), len(alpha), len(Kps), len(Vsys)))


### Create conditional distribution maps
famap = np.sum(lnL,axis=(1,2,3,4,6,7)) #famap = lnL[:,Ridx,Tidx,cidx,oidx,:,kidx,vidx]
fcmap = np.sum(lnL,axis=(1,2,4,5,6,7)) #fcmap = lnL[:,Ridx,Tidx,:,oidx,aidx,kidx,vidx]
fomap = np.sum(lnL,axis=(1,2,3,5,6,7)) #fomap = lnL[:,Ridx,Tidx,cidx,:,aidx,kidx,vidx]

fkmap = np.sum(lnL,axis=(1,2,3,4,5,7)) #fkmap = lnL[:,Ridx,Tidx,cidx,oidx,aidx,:,vidx] ###
fvmap = np.sum(lnL,axis=(1,2,3,4,5,6))  #fvmap = lnL[:,Ridx,Tidx,cidx,oidx,aidx,kidx,:]
comap = np.sum(lnL,axis=(0,1,2,5,6,7)) #comap = lnL[fidx,Ridx,Tidx,:,:,aidx,kidx,vidx]

acmap = np.sum(lnL,axis=(0,1,2,4,6,7)) ## acmap = lnL[fidx,Ridx,Tidx,:,oidx,:,kidx,vidx]
aomap = np.sum(lnL,axis=(0,1,2,3,6,7)) #aomap = lnL[fidx,Ridx,Tidx,cidx,:,:,kidx,vidx]
akmap = np.sum(lnL,axis=(0,1,2,3,4,7)) #akmap = lnL[fidx,Ridx,Tidx,cidx,oidx,:,:,vidx] ###

avmap = np.sum(lnL,axis=(0,1,2,3,4,6))  #avmap = lnL[fidx,Ridx,Tidx,cidx,oidx,:,kidx,:]
kcmap = np.sum(lnL,axis=(0,1,2,4,5,7)) #kcmap = lnL[fidx,Ridx,Tidx,:,oidx,aidx,:,vidx] ####
komap = np.sum(lnL,axis=(0,1,2,3,5,7)) #komap = lnL[fidx,Ridx,Tidx,cidx,:,aidx,:,vidx] ####

kvmap = np.sum(lnL,axis=(0,1,2,3,4,5)) #kvmap = lnL[fidx,Ridx,Tidx,cidx,oidx,aidx,:,:] ####
vcmap = np.sum(lnL,axis=(0,1,2,4,5,6)) #vcmap = lnL[fidx,Ridx,Tidx,:,oidx,aidx,kidx,:]
vomap = np.sum(lnL,axis=(0,1,2,3,5,6)) #vomap = lnL[fidx,Ridx,Tidx,cidx,:,aidx,kidx,:]

###############

Rkmap = np.sum(lnL,axis=(0,2,3,4,5,7)) # Rkmap = lnL[fidx,:,Tidx,cidx,oidx,aidx,:,vidx] ###
Rvmap = np.sum(lnL,axis=(0,2,3,4,5,6)) #Rvmap = lnL[fidx,:,Tidx,cidx,oidx,aidx,kidx,:] ###
Ramap = np.sum(lnL,axis=(0,2,3,4,6,7)) #Ramap = lnL[fidx,:,Tidx,cidx,oidx,:,kidx,vidx] ###


Rcmap = np.sum(lnL,axis=(0,2,4,5,6,7)) # Rcmap = lnL[fidx,:,Tidx,:,oidx,aidx,kidx,vidx] ###
Romap = np.sum(lnL,axis=(0,2,3,5,6,7)) # Romap = lnL[fidx,:,Tidx,cidx,:,aidx,kidx,vidx] ###

Rfmap = np.sum(lnL,axis=(2,3,4,5,6,7)) # Rfmap = lnL[:,:,Tidx,cidx,oidx,aidx,kidx,vidx] ###

################

Tkmap = np.sum(lnL,axis=(0,1,3,4,5,7)) # Tkmap = lnL[fidx,Ridx,:,cidx,oidx,aidx,:,vidx]
Tvmap = np.sum(lnL,axis=(0,1,3,4,5,6)) # Tvmap = lnL[fidx,Ridx,:,cidx,oidx,aidx,kidx,:]
Tamap = np.sum(lnL,axis=(0,1,3,4,6,7)) # Tamap = lnL[fidx,Ridx,:,cidx,oidx,:,kidx,vidx]


Tcmap = np.sum(lnL,axis=(0,1,4,5,6,7)) # Tcmap = lnL[fidx,Ridx,:,:,oidx,aidx,kidx,vidx]
Tomap = np.sum(lnL,axis=(0,1,3,5,6,7)) # Tomap = lnL[fidx,Ridx,:,cidx,:,aidx,kidx,vidx]
Tfmap = np.sum(lnL,axis=(1,3,4,5,6,7)) # Tfmap = lnL[:,Ridx,:,cidx,oidx,aidx,kidx,vidx]
TRmap = np.sum(lnL,axis=(0,3,4,5,6,7)) # TRmap = lnL[fidx,:,:,cidx,oidx,aidx,kidx,vidx]


#################################
### conditional probabilities in liklihoood (L)
### instead of log-liklihood (lnL)

famapL = np.sum(L,axis=(1,2,3,4,6,7)) #famap = lnL[:,Ridx,Tidx,cidx,oidx,:,kidx,vidx]
fcmapL = np.sum(L,axis=(1,2,4,5,6,7)) #fcmap = lnL[:,Ridx,Tidx,:,oidx,aidx,kidx,vidx]
fomapL = np.sum(L,axis=(1,2,3,5,6,7)) #fomap = lnL[:,Ridx,Tidx,cidx,:,aidx,kidx,vidx]

fkmapL = np.sum(L,axis=(1,2,3,4,5,7)) #fkmap = lnL[:,Ridx,Tidx,cidx,oidx,aidx,:,vidx] ###
fvmapL = np.sum(L,axis=(1,2,3,4,5,6))  #fvmap = lnL[:,Ridx,Tidx,cidx,oidx,aidx,kidx,:]
comapL = np.sum(L,axis=(0,1,2,5,6,7)) #comap = lnL[fidx,Ridx,Tidx,:,:,aidx,kidx,vidx]

acmapL = np.sum(L,axis=(0,1,2,4,6,7)) ## acmap = lnL[fidx,Ridx,Tidx,:,oidx,:,kidx,vidx]
aomapL = np.sum(L,axis=(0,1,2,3,6,7)) #aomap = lnL[fidx,Ridx,Tidx,cidx,:,:,kidx,vidx]
akmapL = np.sum(L,axis=(0,1,2,3,4,7)) #akmap = lnL[fidx,Ridx,Tidx,cidx,oidx,:,:,vidx] ###

avmapL = np.sum(L,axis=(0,1,2,3,4,6))  #avmap = lnL[fidx,Ridx,Tidx,cidx,oidx,:,kidx,:]
kcmapL = np.sum(L,axis=(0,1,2,4,5,7)) #kcmap = lnL[fidx,Ridx,Tidx,:,oidx,aidx,:,vidx] ####
komapL = np.sum(L,axis=(0,1,2,3,5,7)) #komap = lnL[fidx,Ridx,Tidx,cidx,:,aidx,:,vidx] ####

kvmapL = np.sum(L,axis=(0,1,2,3,4,5)) #kvmap = lnL[fidx,Ridx,Tidx,cidx,oidx,aidx,:,:] ####
vcmapL = np.sum(L,axis=(0,1,2,4,5,6)) #vcmap = lnL[fidx,Ridx,Tidx,:,oidx,aidx,kidx,:]
vomapL = np.sum(L,axis=(0,1,2,3,5,6)) #vomap = lnL[fidx,Ridx,Tidx,cidx,:,aidx,kidx,:]

###############

RkmapL = np.sum(L,axis=(0,2,3,4,5,7)) # Rkmap = lnL[fidx,:,Tidx,cidx,oidx,aidx,:,vidx] ###
RvmapL = np.sum(L,axis=(0,2,3,4,5,6)) #Rvmap = lnL[fidx,:,Tidx,cidx,oidx,aidx,kidx,:] ###
RamapL = np.sum(L,axis=(0,2,3,4,6,7)) #Ramap = lnL[fidx,:,Tidx,cidx,oidx,:,kidx,vidx] ###


RcmapL = np.sum(L,axis=(0,2,4,5,6,7)) # Rcmap = lnL[fidx,:,Tidx,:,oidx,aidx,kidx,vidx] ###
RomapL = np.sum(L,axis=(0,2,3,5,6,7)) # Romap = lnL[fidx,:,Tidx,cidx,:,aidx,kidx,vidx] ###

RfmapL = np.sum(L,axis=(2,3,4,5,6,7)) # Rfmap = lnL[:,:,Tidx,cidx,oidx,aidx,kidx,vidx] ###

################

TkmapL = np.sum(L,axis=(0,1,3,4,5,7)) # Tkmap = lnL[fidx,Ridx,:,cidx,oidx,aidx,:,vidx]
TvmapL = np.sum(L,axis=(0,1,3,4,5,6)) # Tvmap = lnL[fidx,Ridx,:,cidx,oidx,aidx,kidx,:]
TamapL = np.sum(L,axis=(0,1,3,4,6,7)) # Tamap = lnL[fidx,Ridx,:,cidx,oidx,:,kidx,vidx]


TcmapL = np.sum(L,axis=(0,1,4,5,6,7)) # Tcmap = lnL[fidx,Ridx,:,:,oidx,aidx,kidx,vidx]
TomapL = np.sum(L,axis=(0,1,3,5,6,7)) # Tomap = lnL[fidx,Ridx,:,cidx,:,aidx,kidx,vidx]
TfmapL = np.sum(L,axis=(1,3,4,5,6,7)) # Tfmap = lnL[:,Ridx,:,cidx,oidx,aidx,kidx,vidx]
TRmapL = np.sum(L,axis=(0,3,4,5,6,7)) # TRmap = lnL[fidx,:,:,cidx,oidx,aidx,kidx,vidx]


############
###################





# plt.imshow(kvmap,aspect='auto',interpolation='none',origin='lower')
# plt.colorbar()
# plt.savefig('Test_fa.pdf')



# Compute marginalized distributions
# margf = np.nansum(L[:,Ridx,Tidx,cidx,oidx,aidx,:,vidx],axis=1)

# margR = np.nansum(L[fidx,:,Tidx,cidx,oidx,aidx,:,vidx],axis=1)
# margT = np.nansum(L[fidx,Ridx,:,cidx,oidx,aidx,:,vidx],axis=1)

# margc = np.nansum(L[fidx,Ridx,Tidx,:,oidx,aidx,:,vidx],axis=1)
# margo = np.nansum(L[fidx,Ridx,Tidx,cidx,:,aidx,:,vidx],axis=1)
# marga = np.nansum(L[fidx,Ridx,Tidx,cidx,oidx,:,:,vidx],axis=1)
# margk = np.nansum(L[fidx,Ridx,Tidx,cidx,oidx,aidx,:,:],axis=1)
# margv = np.nansum(L[fidx,Ridx,Tidx,cidx,oidx,aidx,:,:],axis=0)

margf = np.nansum(L,axis=(1,2,3,4,5,6,7))
margR = np.nansum(L,axis=(0,2,3,4,5,6,7))
margT = np.nansum(L,axis=(0,1,3,4,5,6,7))
margc = np.nansum(L,axis=(0,1,2,4,5,6,7))
margo = np.nansum(L,axis=(0,1,2,3,5,6,7))
marga = np.nansum(L,axis=(0,1,2,3,4,6,7))
margk = np.nansum(L,axis=(0,1,2,3,4,5,7))
margv = np.nansum(L,axis=(0,1,2,3,4,5,6))


### Manually calculated marginalized distributions
# margf2 = np.nansum(famapL,axis=1) + \
#           np.nansum(fcmapL,axis=1) + \
#           np.nansum(fomapL,axis=1) + \
#           np.nansum(fkmapL,axis=1) + \
#           np.nansum(fvmapL,axis=1) + \
#           np.nansum(RfmapL,axis=1) + \
#           np.nansum(TfmapL,axis=1)

# ### test multiplying liklihoods 
# margf5 = np.nansum(famapL,axis=1) * \
#          np.nansum(fcmapL,axis=1) * \
#          np.nansum(fomapL,axis=1) * \
#          np.nansum(fkmapL,axis=1) * \
#          np.nansum(fvmapL,axis=1) * \
#          np.nansum(RfmapL,axis=1) * \
#          np.nansum(TfmapL,axis=1)
         
# margf3a = np.nansum(famap,axis=1) + \
#          np.nansum(fcmap,axis=1) + \
#          np.nansum(fomap,axis=1) + \
#          np.nansum(fkmap,axis=1) + \
#          np.nansum(fvmap,axis=1) + \
#          np.nansum(Rfmap,axis=1) + \
#          np.nansum(Tfmap,axis=1)
             
# margf3 = np.exp(margf3a - maximum)  #### 

# margf4 = np.nansum(lnL[:,Ridx,Tidx,cidx,oidx,aidx,:,vidx],axis=1)

del L 
del lnL
            
cond_output_dir = '%s/cond'%(path)
if not os.path.exists(cond_output_dir):
    os.makedirs(cond_output_dir)



# OutputForPlot = 'LnLNeededForCornerPlot'

# if not os.path.exists(OutputForPlot):
#     os.makedirs(OutputForPlot)

### log-likelihood 2D plots 
# np.save('%s/famap.npy'%(cond_output_dir),famap)
# np.save('%s/fcmap.npy'%(cond_output_dir),fcmap)
# np.save('%s/fomap.npy'%(cond_output_dir),fomap)

# np.save('%s/fkmap.npy'%(cond_output_dir),fkmap)
# np.save('%s/fvmap.npy'%(cond_output_dir),fvmap)
# np.save('%s/comap.npy'%(cond_output_dir),comap)

# np.save('%s/acmap.npy'%(cond_output_dir),acmap)
# np.save('%s/aomap.npy'%(cond_output_dir),aomap)
# np.save('%s/akmap.npy'%(cond_output_dir),akmap)

# np.save('%s/avmap.npy'%(cond_output_dir),avmap)
# np.save('%s/kcmap.npy'%(cond_output_dir),kcmap)
# np.save('%s/komap.npy'%(cond_output_dir),komap)

# np.save('%s/kvmap.npy'%(cond_output_dir),kvmap)
# np.save('%s/vcmap.npy'%(cond_output_dir),vcmap)
# np.save('%s/vomap.npy'%(cond_output_dir),vomap)

# np.save('%s/Rkmap.npy'%(cond_output_dir),Rkmap)
# np.save('%s/Rvmap.npy'%(cond_output_dir),Rvmap)
# np.save('%s/Ramap.npy'%(cond_output_dir),Ramap)
# np.save('%s/Rcmap.npy'%(cond_output_dir),Rcmap)
# np.save('%s/Romap.npy'%(cond_output_dir),Romap)
# np.save('%s/Rfmap.npy'%(cond_output_dir),Rfmap)

# np.save('%s/Tkmap.npy'%(cond_output_dir),Tkmap)
# np.save('%s/Tvmap.npy'%(cond_output_dir),Tvmap)
# np.save('%s/Tamap.npy'%(cond_output_dir),Tamap)
# np.save('%s/Tcmap.npy'%(cond_output_dir),Tcmap)
# np.save('%s/Tomap.npy'%(cond_output_dir),Tomap)
# np.save('%s/Tfmap.npy'%(cond_output_dir),Tfmap)
# np.save('%s/TRmap.npy'%(cond_output_dir),TRmap)
    
    
    
### 2D likelihood plots 
np.save('%s/famapL.npy'%(cond_output_dir),famapL)
np.save('%s/fcmapL.npy'%(cond_output_dir),fcmapL)
np.save('%s/fomapL.npy'%(cond_output_dir),fomapL)

np.save('%s/fkmapL.npy'%(cond_output_dir),fkmapL)
np.save('%s/fvmapL.npy'%(cond_output_dir),fvmapL)
np.save('%s/comapL.npy'%(cond_output_dir),comapL)

np.save('%s/acmapL.npy'%(cond_output_dir),acmapL)
np.save('%s/aomapL.npy'%(cond_output_dir),aomapL)
np.save('%s/akmapL.npy'%(cond_output_dir),akmapL)

np.save('%s/avmapL.npy'%(cond_output_dir),avmapL)
np.save('%s/kcmapL.npy'%(cond_output_dir),kcmapL)
np.save('%s/komapL.npy'%(cond_output_dir),komapL)

np.save('%s/kvmapL.npy'%(cond_output_dir),kvmapL)
np.save('%s/vcmapL.npy'%(cond_output_dir),vcmapL)
np.save('%s/vomapL.npy'%(cond_output_dir),vomapL)

np.save('%s/RkmapL.npy'%(cond_output_dir),RkmapL)
np.save('%s/RvmapL.npy'%(cond_output_dir),RvmapL)
np.save('%s/RamapL.npy'%(cond_output_dir),RamapL)
np.save('%s/RcmapL.npy'%(cond_output_dir),RcmapL)
np.save('%s/RomapL.npy'%(cond_output_dir),RomapL)
np.save('%s/RfmapL.npy'%(cond_output_dir),RfmapL)

np.save('%s/TkmapL.npy'%(cond_output_dir),TkmapL)
np.save('%s/TvmapL.npy'%(cond_output_dir),TvmapL)
np.save('%s/TamapL.npy'%(cond_output_dir),TamapL)
np.save('%s/TcmapL.npy'%(cond_output_dir),TcmapL)
np.save('%s/TomapL.npy'%(cond_output_dir),TomapL)
np.save('%s/TfmapL.npy'%(cond_output_dir),TfmapL)
np.save('%s/TRmapL.npy'%(cond_output_dir),TRmapL)

    

np.save('%s/maximum.npy'%(cond_output_dir),np.array([maximum]))



marg_output_dir = '%s/margs'%(path)
if not os.path.exists(marg_output_dir):
    os.makedirs(marg_output_dir)
    
    

np.save('%s/margf.npy'%(marg_output_dir),margf)
np.save('%s/margc.npy'%(marg_output_dir),margc)
np.save('%s/margo.npy'%(marg_output_dir),margo)
np.save('%s/marga.npy'%(marg_output_dir),marga)
np.save('%s/margk.npy'%(marg_output_dir),margk)
np.save('%s/margv.npy'%(marg_output_dir),margv)

np.save('%s/margR.npy'%(marg_output_dir),margR)
np.save('%s/margT.npy'%(marg_output_dir),margT)






#margf528dump = np.savetxt('margf618dump.txt',margf)
#margf528dump = np.savetxt('618_and_528_margf_dump.txt',margf)

print('len(margf)')
print(len(margf))

print('Margf:')
print(margf)




# margcPlotHalfWidth = 5
# margoPlotHalfWidth = 5
# margaPlotHalfWidth = 5
# margkPlotHalfWidth = 20
# margvPlotHalfWidth = 5




# plt.figure()
# plt.plot(margv)
# plt.savefig('margv.pdf')

#### First attempt to try with 4,3,2 to help distinguish which CC dim is which in LnL output 
# Species = 'Fe'
# vmrs = [-1.0, 0.0, 1.0, 1.7]
# fcs = [0.25, 0.5, 0.75]
# CtoOs = [0.55, 0.7] ### Note that 0.55 is solar 

# alpha = np.arange(0.5, 5., 0.1)

# ### Set offset and contrast to lenght 1 to make it faster and help judge file size 

# #offset = np.arange(-60.0,60.0, 2.)
# offset = np.array([0.0])

# #contrast = np.arange(0.,1.1, 0.1)
# contrast = np.array([0.5])


####### For the plots with Herman dimensions (replacing VMR with metallicity) to see offset up to pm 180 degrees:

# Species = 'Fe'
# vmrs = [-1.0, 0.0, 1.0, 1.7, 2.0, 2.3]

# fcs = [0.5]
# CtoOs = [0.55] ### Note that 0.55 is solar 

# offset = np.arange(-180.0,200,20.0)
# contrast = np.arange(0.,1.25, 0.25)


# alpha = np.arange(0.5, 5., 0.1)
# vgrid = np.arange(-600.0,601.0,1.0)
# Vsys = np.arange(-150., 151., 1)
# Kps = np.arange(240.0-25,240.0+25+1,1)


# ### For making files with only Goyal dimensions (0.795 GB)
# Species = 'Fe'
# vmrs = [-1.0, 0.0, 1.0, 1.7, 2.0, 2.3]
# fcs = [0.25, 0.5, 0.75, 1.0]
# CtoOs = [0.35, 0.55, 0.7, 0.75, 1.0, 1.5] ### Note that 0.55 is solar 

# alpha = np.arange(0.5, 5., 0.1)

# offset = np.array([0.0])
# contrast = np.array([0.0])

# vgrid = np.arange(-600.0,601.0,1.0)
# Vsys = np.arange(-150., 151., 1)
# Kps = np.arange(240.0-25,240.0+25+1,1)

### Values for making a LnL file with a different extent in each dimension 
### to make it easier to check that the dimensions are working out 

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

############################3

# Species = 'Fe'
# vmrs = [-1.0, 0.0, 1.0, 1.7, 2.0, 2.3]
# #vmrs = [2.3]

# fcs = [0.25, 0.5, 0.75, 1.0]
# CtoOs = [0.35, 0.55, 0.7, 0.75, 1.0, 1.5] ### Note that 0.55 is solar 

# #alpha = np.arange(0.5, 5., 0.1)
# alpha = np.arange(0.1, 1.1, 0.1)

# offset = np.array([-60.0,-30.0,0.0,30.0,60.0])
# contrast = np.array([0.0,0.25,0.5,0.75,1.0])

# vgrid = np.arange(-600.0,601.0,1.0)
# Vsys = np.arange(-150., 151., 1)
# Kps = np.arange(240.0-25,240.0+25+1,1)

#########################

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

###########################################
############################################

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
Vsys = np.arange(-20-28, -20+28, 1)

Kps = np.arange(240.0-25,240.0+25+1,1)



assert len(margf) == len(vmrs)
assert len(margR) == len(fcs)
assert len(margT) == len(CtoOs)

assert len(margc) == len(contrast)
assert len(margo) == len(offset)
assert len(marga) == len(alpha)

assert len(margk) == len(Kps)
assert len(margv) == len(Vsys)




# #########################################
# #### Corner plot with no imshow extent (showing dimensions)
# #### of the Goyal model dimensions and the Herman offset and contrast dimensions 

ncols = 8
nrows = 8
# grid = GridSpec(nrows, ncols,
#                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)


grid = GridSpec(nrows, ncols,
                left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.8, hspace=0.8)


fig = plt.figure(0)
fig.clf()

# Add axes which can span multiple grid boxes
ax1 = fig.add_subplot(grid[0, 0])

ax2 = fig.add_subplot(grid[1, 0])
ax3 = fig.add_subplot(grid[1, 1])

ax4 = fig.add_subplot(grid[2, 0])
ax5 = fig.add_subplot(grid[2, 1])
ax6 = fig.add_subplot(grid[2, 2])

ax7 = fig.add_subplot(grid[3, 0])
ax8 = fig.add_subplot(grid[3, 1])
ax9 = fig.add_subplot(grid[3, 2])
ax10 = fig.add_subplot(grid[3, 3])

ax11 = fig.add_subplot(grid[4, 0])
ax12 = fig.add_subplot(grid[4, 1])
ax13 = fig.add_subplot(grid[4, 2])
ax14 = fig.add_subplot(grid[4, 3])
ax15 = fig.add_subplot(grid[4, 4])

ax16 = fig.add_subplot(grid[5, 0])
ax17 = fig.add_subplot(grid[5, 1])
ax18 = fig.add_subplot(grid[5, 2])
ax19 = fig.add_subplot(grid[5, 3])
ax20 = fig.add_subplot(grid[5, 4])
ax21 = fig.add_subplot(grid[5, 5])

ax22 = fig.add_subplot(grid[6, 0])
ax23 = fig.add_subplot(grid[6, 1])
ax24 = fig.add_subplot(grid[6, 2])
ax25 = fig.add_subplot(grid[6, 3])
ax26 = fig.add_subplot(grid[6, 4])
ax27 = fig.add_subplot(grid[6, 5])
ax28 = fig.add_subplot(grid[6, 6])

ax29 = fig.add_subplot(grid[7, 0])
ax30 = fig.add_subplot(grid[7, 1])
ax31 = fig.add_subplot(grid[7, 2])
ax32 = fig.add_subplot(grid[7, 3])
ax33 = fig.add_subplot(grid[7, 4])
ax34 = fig.add_subplot(grid[7, 5])
ax35 = fig.add_subplot(grid[7, 6])
ax36 = fig.add_subplot(grid[7, 7])


ax1.step(np.arange(len(margk)),margk,where='mid')



ax2.imshow(kvmap.T,aspect='auto',origin='lower')
ax3.step(np.arange(len(margv)),margv,where='mid')



ax4.imshow(akmap,aspect='auto',origin='lower')
ax5.imshow(avmap,aspect='auto',origin='lower')
ax6.step(np.arange(len(marga)),marga,where='mid')


ax7.imshow(kcmap,aspect='auto',origin='lower')
ax8.imshow(vcmap,aspect='auto',origin='lower')
ax9.imshow(acmap,aspect='auto',origin='lower')
ax10.step(np.arange(len(margc)),margc,where='mid')



ax11.imshow(komap,aspect='auto',origin='lower')
ax12.imshow(vomap,aspect='auto',origin='lower')
ax13.imshow(aomap,aspect='auto',origin='lower')
ax14.imshow(comap.T,aspect='auto',origin='lower')
ax15.step(np.arange(len(margo)),margo,where='mid')

ax16.imshow(fkmap,aspect='auto',origin='lower')
ax17.imshow(fvmap,aspect='auto',origin='lower')
ax18.imshow(famap,aspect='auto',origin='lower')
ax19.imshow(fcmap,aspect='auto',origin='lower')
ax20.imshow(fomap,aspect='auto',origin='lower')
ax21.step(np.arange(len(margf)),margf,where='mid')

####

im22 = ax22.imshow(Rkmap,aspect='auto',origin='lower')


divider = make_axes_locatable(ax22)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im22, cax=cax, orientation='vertical')


ax23.imshow(Rvmap,aspect='auto',origin='lower')
ax24.imshow(Ramap,aspect='auto',origin='lower')
ax25.imshow(Rcmap,aspect='auto',origin='lower')
ax26.imshow(Romap,aspect='auto',origin='lower')
ax27.imshow(Rfmap.T,aspect='auto',origin='lower')
ax28.step(np.arange(len(margR)),margR,where='mid')

ax29.imshow(Tkmap,aspect='auto',origin='lower')
ax30.imshow(Tvmap,aspect='auto',origin='lower')
ax31.imshow(Tamap,aspect='auto',origin='lower')
ax32.imshow(Tcmap,aspect='auto',origin='lower')
ax33.imshow(Tomap,aspect='auto',origin='lower')
ax34.imshow(Tfmap.T,aspect='auto',origin='lower')
ax35.imshow(TRmap.T,aspect='auto',origin='lower')
ax36.step(np.arange(len(margT)),margT,where='mid')


ax2.set_ylabel(r'v$_{sys}$ (km s$^{-1}$)')
ax4.set_ylabel(r'$\alpha$')
ax7.set_ylabel(r'C')
ax11.set_ylabel(r'$\theta$ ($^\circ$)')
#ax16.set_ylabel(r'$\log_{10}$ (metallicity)')
ax16.set_ylabel(r'met')
ax22.set_ylabel(r'F$_c$')
ax29.set_ylabel(r'C/O')


ax29.set_xlabel(r'K$_p$ (km s$^{-1}$)')
ax30.set_xlabel(r'v$_{sys}$ (km s$^{-1}$)')
ax31.set_xlabel(r'$\alpha$')
ax32.set_xlabel(r'C')
ax33.set_xlabel(r'$\theta$ ($^\circ$)')
#ax34.set_xlabel(r'$\log_{10}$ (metallicity)')
ax34.set_xlabel(r'met')
ax35.set_xlabel(r'F$_c$')
ax36.set_xlabel(r'C/O')



#ax2.axes.xaxis.set_visible(False)
#ax.axes.yaxis.set_visible(False)

#fig.savefig('618_Fe_6FeH.pdf')
#fig.savefig('test_corner.pdf')

#fig.savefig('618_OffsetPm60.pdf')
#fig.savefig('528_OffsetPm60.pdf')
fig.savefig('%s/Corner_plot_dims_no_colorbar.pdf'%(path))
plt.close()

##########################################################
##################################################

# #########################################
# #### Corner plot WITH imshow extent (showing dimensions)
# #### of the Goyal model dimensions and the Herman offset and contrast dimensions 


##############################


SMALL_SIZE = 5
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#############################

ncols = 8
nrows = 8
# grid = GridSpec(nrows, ncols,
#                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)


grid = GridSpec(nrows, ncols,
                left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.2, hspace=0.2)


fig = plt.figure()
fig.clf()

# Add axes which can span multiple grid boxes
ax1 = fig.add_subplot(grid[0, 0])

ax2 = fig.add_subplot(grid[1, 0])
ax3 = fig.add_subplot(grid[1, 1])

ax4 = fig.add_subplot(grid[2, 0])
ax5 = fig.add_subplot(grid[2, 1])
ax6 = fig.add_subplot(grid[2, 2])

ax7 = fig.add_subplot(grid[3, 0])
ax8 = fig.add_subplot(grid[3, 1])
ax9 = fig.add_subplot(grid[3, 2])
ax10 = fig.add_subplot(grid[3, 3])

ax11 = fig.add_subplot(grid[4, 0])
ax12 = fig.add_subplot(grid[4, 1])
ax13 = fig.add_subplot(grid[4, 2])
ax14 = fig.add_subplot(grid[4, 3])
ax15 = fig.add_subplot(grid[4, 4])

ax16 = fig.add_subplot(grid[5, 0])
ax17 = fig.add_subplot(grid[5, 1])
ax18 = fig.add_subplot(grid[5, 2])
ax19 = fig.add_subplot(grid[5, 3])
ax20 = fig.add_subplot(grid[5, 4])
ax21 = fig.add_subplot(grid[5, 5])

ax22 = fig.add_subplot(grid[6, 0])
ax23 = fig.add_subplot(grid[6, 1])
ax24 = fig.add_subplot(grid[6, 2])
ax25 = fig.add_subplot(grid[6, 3])
ax26 = fig.add_subplot(grid[6, 4])
ax27 = fig.add_subplot(grid[6, 5])
ax28 = fig.add_subplot(grid[6, 6])

ax29 = fig.add_subplot(grid[7, 0])
ax30 = fig.add_subplot(grid[7, 1])
ax31 = fig.add_subplot(grid[7, 2])
ax32 = fig.add_subplot(grid[7, 3])
ax33 = fig.add_subplot(grid[7, 4])
ax34 = fig.add_subplot(grid[7, 5])
ax35 = fig.add_subplot(grid[7, 6])
ax36 = fig.add_subplot(grid[7, 7])


ax1.step(np.arange(len(margk)),margk,where='mid')



ax2.imshow(kvmap.T,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],Vsys[0],Vsys[-1]])
ax3.step(np.arange(len(margv)),margv,where='mid')



ax4.imshow(akmap,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],alpha[0],alpha[-1]])
ax5.imshow(avmap,aspect='auto',origin='lower')
ax6.step(np.arange(len(marga)),marga,where='mid')


ax7.imshow(kcmap,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],contrast[0],contrast[-1]])
ax8.imshow(vcmap,aspect='auto',origin='lower')
ax9.imshow(acmap,aspect='auto',origin='lower')
ax10.step(np.arange(len(margc)),margc,where='mid')



ax11.imshow(komap,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],offset[0],offset[-1]])
ax12.imshow(vomap,aspect='auto',origin='lower')
ax13.imshow(aomap,aspect='auto',origin='lower')
ax14.imshow(comap.T,aspect='auto',origin='lower')
ax15.step(np.arange(len(margo)),margo,where='mid')

ax16.imshow(fkmap,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],vmrs[0],vmrs[-1]])
ax17.imshow(fvmap,aspect='auto',origin='lower')
ax18.imshow(famap,aspect='auto',origin='lower')
ax19.imshow(fcmap,aspect='auto',origin='lower')
ax20.imshow(fomap,aspect='auto',origin='lower')
ax21.step(np.arange(len(margf)),margf,where='mid')

####

ax22.imshow(Rkmap,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],fcs[0],fcs[-1]])
ax23.imshow(Rvmap,aspect='auto',origin='lower')
ax24.imshow(Ramap,aspect='auto',origin='lower')
ax25.imshow(Rcmap,aspect='auto',origin='lower')
ax26.imshow(Romap,aspect='auto',origin='lower')
ax27.imshow(Rfmap.T,aspect='auto',origin='lower')
ax28.step(np.arange(len(margR)),margR,where='mid')

ax29.imshow(Tkmap,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],CtoOs[0],CtoOs[-1]])
ax30.imshow(Tvmap,aspect='auto',origin='lower',extent=[Vsys[0],Vsys[-1],CtoOs[0],CtoOs[-1]])
ax31.imshow(Tamap,aspect='auto',origin='lower',extent=[alpha[0],alpha[-1],CtoOs[0],CtoOs[-1]])
ax32.imshow(Tcmap,aspect='auto',origin='lower',extent=[contrast[0],contrast[-1],CtoOs[0],CtoOs[-1]])
ax33.imshow(Tomap,aspect='auto',origin='lower',extent=[offset[0],offset[-1],CtoOs[0],CtoOs[-1]])
ax34.imshow(Tfmap.T,aspect='auto',origin='lower',extent=[vmrs[0],vmrs[-1],CtoOs[0],CtoOs[-1]])
ax35.imshow(TRmap.T,aspect='auto',origin='lower',extent=[fcs[0],fcs[-1],CtoOs[0],CtoOs[-1]])
ax36.step(CtoOs,margT,where='mid')


ax2.set_ylabel(r'v$_{sys}$ (km s$^{-1}$)')
ax4.set_ylabel(r'$\alpha$')
ax7.set_ylabel(r'C')
ax11.set_ylabel(r'$\theta$ ($^\circ$)')
#ax16.set_ylabel(r'$\log_{10}$ (metallicity)')
ax16.set_ylabel(r'met')
ax22.set_ylabel(r'F$_c$')
ax29.set_ylabel(r'C/O')


ax29.set_xlabel(r'K$_p$ (km s$^{-1}$)')
ax30.set_xlabel(r'v$_{sys}$ (km s$^{-1}$)')
ax31.set_xlabel(r'$\alpha$')
ax32.set_xlabel(r'C')
ax33.set_xlabel(r'$\theta$ ($^\circ$)')
#ax34.set_xlabel(r'$\log_{10}$ (metallicity)')
ax34.set_xlabel(r'met')
ax35.set_xlabel(r'F$_c$')
ax36.set_xlabel(r'C/O')

# ###########

ax1.axes.xaxis.set_visible(False)
ax1.axes.yaxis.set_visible(False)

ax2.axes.xaxis.set_visible(False)


ax3.axes.xaxis.set_visible(False)
ax3.axes.yaxis.set_visible(False)

ax4.axes.xaxis.set_visible(False)

ax5.axes.xaxis.set_visible(False)
ax5.axes.yaxis.set_visible(False)

ax6.axes.xaxis.set_visible(False)
ax6.axes.yaxis.set_visible(False)

ax7.axes.xaxis.set_visible(False)

ax8.axes.xaxis.set_visible(False)
ax8.axes.yaxis.set_visible(False)

ax9.axes.xaxis.set_visible(False)
ax9.axes.yaxis.set_visible(False)

ax10.axes.xaxis.set_visible(False)
ax10.axes.yaxis.set_visible(False)

ax11.axes.xaxis.set_visible(False)

ax12.axes.xaxis.set_visible(False)
ax12.axes.yaxis.set_visible(False)

ax13.axes.xaxis.set_visible(False)
ax13.axes.yaxis.set_visible(False)

ax14.axes.xaxis.set_visible(False)
ax14.axes.yaxis.set_visible(False)

ax15.axes.xaxis.set_visible(False)
ax15.axes.yaxis.set_visible(False)

ax16.axes.xaxis.set_visible(False)


ax17.axes.yaxis.set_visible(False)
ax17.axes.xaxis.set_visible(False)

ax18.axes.yaxis.set_visible(False)
ax18.axes.xaxis.set_visible(False)

ax19.axes.yaxis.set_visible(False)
ax19.axes.xaxis.set_visible(False)

ax20.axes.yaxis.set_visible(False)
ax20.axes.xaxis.set_visible(False)

ax21.axes.xaxis.set_visible(False)
ax21.axes.yaxis.set_visible(False)


ax22.axes.xaxis.set_visible(False)

ax23.axes.xaxis.set_visible(False)
ax23.axes.yaxis.set_visible(False)

ax24.axes.xaxis.set_visible(False)
ax24.axes.yaxis.set_visible(False)

ax25.axes.xaxis.set_visible(False)
ax25.axes.yaxis.set_visible(False)

ax26.axes.xaxis.set_visible(False)
ax26.axes.yaxis.set_visible(False)

ax27.axes.xaxis.set_visible(False)
ax27.axes.yaxis.set_visible(False)

ax28.axes.xaxis.set_visible(False)
ax28.axes.yaxis.set_visible(False)


ax30.axes.yaxis.set_visible(False)
ax31.axes.yaxis.set_visible(False)
ax32.axes.yaxis.set_visible(False)
ax33.axes.yaxis.set_visible(False)
ax34.axes.yaxis.set_visible(False)
ax35.axes.yaxis.set_visible(False)
ax36.axes.yaxis.set_visible(False)




#ax2.axes.xaxis.set_visible(False)
#ax.axes.yaxis.set_visible(False)

#fig.savefig('618_Fe_6FeH.pdf')
#fig.savefig('test_corner.pdf')

#fig.savefig('618_OffsetPm60.pdf')
#fig.savefig('528_OffsetPm60.pdf')
##fig.savefig('Corner_plot_Goyal_and_Herman_dims.pdf')

#fig.savefig('Goyal_and_Herman_dims_with_scale.pdf')

fig.savefig('%s/Corner_plot_Goyal_and_Herman_dims_With_Scale.pdf'%(path))
plt.close()

#########################################################################
###################################################################
######################################################################

# #########################################
# #### Corner plot with no imshow extent (showing dimensions) of lnL and colorbars 


ncols = 8
nrows = 8
# grid = GridSpec(nrows, ncols,
#                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)


grid = GridSpec(nrows, ncols,
                left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.6, hspace=0.4)


fig = plt.figure(figsize=(20,20))
fig.clf()

# Add axes which can span multiple grid boxes
ax1 = fig.add_subplot(grid[0, 0])

ax2 = fig.add_subplot(grid[1, 0])
ax3 = fig.add_subplot(grid[1, 1])

ax4 = fig.add_subplot(grid[2, 0])
ax5 = fig.add_subplot(grid[2, 1])
ax6 = fig.add_subplot(grid[2, 2])

ax7 = fig.add_subplot(grid[3, 0])
ax8 = fig.add_subplot(grid[3, 1])
ax9 = fig.add_subplot(grid[3, 2])
ax10 = fig.add_subplot(grid[3, 3])

ax11 = fig.add_subplot(grid[4, 0])
ax12 = fig.add_subplot(grid[4, 1])
ax13 = fig.add_subplot(grid[4, 2])
ax14 = fig.add_subplot(grid[4, 3])
ax15 = fig.add_subplot(grid[4, 4])

ax16 = fig.add_subplot(grid[5, 0])
ax17 = fig.add_subplot(grid[5, 1])
ax19 = fig.add_subplot(grid[5, 3])
ax20 = fig.add_subplot(grid[5, 4])
ax21 = fig.add_subplot(grid[5, 5])

ax22 = fig.add_subplot(grid[6, 0])
ax23 = fig.add_subplot(grid[6, 1])
ax24 = fig.add_subplot(grid[6, 2])
ax25 = fig.add_subplot(grid[6, 3])
ax26 = fig.add_subplot(grid[6, 4])
ax27 = fig.add_subplot(grid[6, 5])
ax28 = fig.add_subplot(grid[6, 6])

ax29 = fig.add_subplot(grid[7, 0])
ax30 = fig.add_subplot(grid[7, 1])
ax31 = fig.add_subplot(grid[7, 2])
ax32 = fig.add_subplot(grid[7, 3])
ax33 = fig.add_subplot(grid[7, 4])
ax34 = fig.add_subplot(grid[7, 5])
ax35 = fig.add_subplot(grid[7, 6])
ax36 = fig.add_subplot(grid[7, 7])


ax1.step(np.arange(len(margk)),margk,where='mid')

AddSubplotWithColorbar(fig,ax2,kvmap.T)
ax3.step(np.arange(len(margv)),margv,where='mid')


AddSubplotWithColorbar(fig,ax4,akmap)
AddSubplotWithColorbar(fig,ax5,avmap)
ax6.step(np.arange(len(marga)),marga,where='mid')


AddSubplotWithColorbar(fig,ax7,kcmap)
AddSubplotWithColorbar(fig,ax8,vcmap)
AddSubplotWithColorbar(fig,ax9,acmap)
ax10.step(np.arange(len(margc)),margc,where='mid')


AddSubplotWithColorbar(fig,ax11,komap)
AddSubplotWithColorbar(fig,ax12,vomap)
AddSubplotWithColorbar(fig,ax13,aomap)
AddSubplotWithColorbar(fig,ax14,comap.T)
ax15.step(np.arange(len(margo)),margo,where='mid')


AddSubplotWithColorbar(fig,ax16,fkmap)
AddSubplotWithColorbar(fig,ax17,fvmap)
AddSubplotWithColorbar(fig,ax18,famap)
AddSubplotWithColorbar(fig,ax19,fcmap)
AddSubplotWithColorbar(fig,ax20,fomap)
ax21.step(np.arange(len(margf)),margf,where='mid')

####



AddSubplotWithColorbar(fig,ax22,Rkmap)
AddSubplotWithColorbar(fig,ax23,Rvmap)
AddSubplotWithColorbar(fig,ax24,Ramap)    
AddSubplotWithColorbar(fig,ax25,Rcmap)
AddSubplotWithColorbar(fig,ax26,Romap)
AddSubplotWithColorbar(fig,ax27,Rfmap.T)
ax28.step(np.arange(len(margR)),margR,where='mid')


AddSubplotWithColorbar(fig,ax29,Tkmap)
AddSubplotWithColorbar(fig,ax30,Tvmap)
AddSubplotWithColorbar(fig,ax31,Tamap)
AddSubplotWithColorbar(fig,ax32,Tcmap)
AddSubplotWithColorbar(fig,ax33,Tomap)
AddSubplotWithColorbar(fig,ax34,Tfmap.T)
AddSubplotWithColorbar(fig,ax35,TRmap.T)
ax36.step(np.arange(len(margT)),margT,where='mid')


ax2.set_ylabel(r'v$_{sys}$ (km s$^{-1}$)')
ax4.set_ylabel(r'$\alpha$')
ax7.set_ylabel(r'C')
ax11.set_ylabel(r'$\theta$ ($^\circ$)')
#ax16.set_ylabel(r'$\log_{10}$ (metallicity)')
ax16.set_ylabel(r'met')
ax22.set_ylabel(r'F$_c$')
ax29.set_ylabel(r'C/O')


ax29.set_xlabel(r'K$_p$ (km s$^{-1}$)')
ax30.set_xlabel(r'v$_{sys}$ (km s$^{-1}$)')
ax31.set_xlabel(r'$\alpha$')
ax32.set_xlabel(r'C')
ax33.set_xlabel(r'$\theta$ ($^\circ$)')
#ax34.set_xlabel(r'$\log_{10}$ (metallicity)')
ax34.set_xlabel(r'met')
ax35.set_xlabel(r'F$_c$')
ax36.set_xlabel(r'C/O')



#ax2.axes.xaxis.set_visible(False)
#ax.axes.yaxis.set_visible(False)

#fig.savefig('618_Fe_6FeH.pdf')
#fig.savefig('test_corner.pdf')

#fig.savefig('618_OffsetPm60.pdf')
#fig.savefig('528_OffsetPm60.pdf')
fig.savefig('%s/Corner_plot_dims_lnL_colorbars.pdf'%(path))
plt.close()


#######################################
### 
# #########################################
# #### Corner plot with no imshow extent (showing dimensions) of L and colorbars 


ncols = 8
nrows = 8
# grid = GridSpec(nrows, ncols,
#                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)


grid = GridSpec(nrows, ncols,
                left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.6, hspace=0.4)


fig = plt.figure(figsize=(20,20))
fig.clf()

# Add axes which can span multiple grid boxes
ax1 = fig.add_subplot(grid[0, 0])

ax2 = fig.add_subplot(grid[1, 0])
ax3 = fig.add_subplot(grid[1, 1])

ax4 = fig.add_subplot(grid[2, 0])
ax5 = fig.add_subplot(grid[2, 1])
ax6 = fig.add_subplot(grid[2, 2])

ax7 = fig.add_subplot(grid[3, 0])
ax8 = fig.add_subplot(grid[3, 1])
ax9 = fig.add_subplot(grid[3, 2])
ax10 = fig.add_subplot(grid[3, 3])

ax11 = fig.add_subplot(grid[4, 0])
ax12 = fig.add_subplot(grid[4, 1])
ax13 = fig.add_subplot(grid[4, 2])
ax14 = fig.add_subplot(grid[4, 3])
ax15 = fig.add_subplot(grid[4, 4])

ax16 = fig.add_subplot(grid[5, 0])
ax17 = fig.add_subplot(grid[5, 1])
ax18 = fig.add_subplot(grid[5, 2])
ax19 = fig.add_subplot(grid[5, 3])
ax20 = fig.add_subplot(grid[5, 4])
ax21 = fig.add_subplot(grid[5, 5])

ax22 = fig.add_subplot(grid[6, 0])
ax23 = fig.add_subplot(grid[6, 1])
ax24 = fig.add_subplot(grid[6, 2])
ax25 = fig.add_subplot(grid[6, 3])
ax26 = fig.add_subplot(grid[6, 4])
ax27 = fig.add_subplot(grid[6, 5])
ax28 = fig.add_subplot(grid[6, 6])

ax29 = fig.add_subplot(grid[7, 0])
ax30 = fig.add_subplot(grid[7, 1])
ax31 = fig.add_subplot(grid[7, 2])
ax32 = fig.add_subplot(grid[7, 3])
ax33 = fig.add_subplot(grid[7, 4])
ax34 = fig.add_subplot(grid[7, 5])
ax35 = fig.add_subplot(grid[7, 6])
ax36 = fig.add_subplot(grid[7, 7])


ax1.step(np.arange(len(margk)),margk,where='mid')

AddSubplotWithColorbar(fig,ax2,kvmapL.T)
ax3.step(np.arange(len(margv)),margv,where='mid')


AddSubplotWithColorbar(fig,ax4,akmapL)
AddSubplotWithColorbar(fig,ax5,avmapL)
ax6.step(np.arange(len(marga)),marga,where='mid')


AddSubplotWithColorbar(fig,ax7,kcmapL)
AddSubplotWithColorbar(fig,ax8,vcmapL)
AddSubplotWithColorbar(fig,ax9,acmapL)
ax10.step(np.arange(len(margc)),margc,where='mid')


AddSubplotWithColorbar(fig,ax11,komapL)
AddSubplotWithColorbar(fig,ax12,vomapL)
AddSubplotWithColorbar(fig,ax13,aomapL)
AddSubplotWithColorbar(fig,ax14,comapL.T)
ax15.step(np.arange(len(margo)),margo,where='mid')


AddSubplotWithColorbar(fig,ax16,fkmapL)
AddSubplotWithColorbar(fig,ax17,fvmapL)
AddSubplotWithColorbar(fig,ax18,famapL)
AddSubplotWithColorbar(fig,ax19,fcmapL)
AddSubplotWithColorbar(fig,ax20,fomapL)
ax21.step(np.arange(len(margf)),margf,where='mid')

####



AddSubplotWithColorbar(fig,ax22,RkmapL)
AddSubplotWithColorbar(fig,ax23,RvmapL)
AddSubplotWithColorbar(fig,ax24,RamapL)    
AddSubplotWithColorbar(fig,ax25,RcmapL)
AddSubplotWithColorbar(fig,ax26,RomapL)
AddSubplotWithColorbar(fig,ax27,RfmapL.T)
ax28.step(np.arange(len(margR)),margR,where='mid')


AddSubplotWithColorbar(fig,ax29,TkmapL)
AddSubplotWithColorbar(fig,ax30,TvmapL)
AddSubplotWithColorbar(fig,ax31,TamapL)
AddSubplotWithColorbar(fig,ax32,TcmapL)
AddSubplotWithColorbar(fig,ax33,TomapL)
AddSubplotWithColorbar(fig,ax34,TfmapL.T)
AddSubplotWithColorbar(fig,ax35,TRmapL.T)
ax36.step(np.arange(len(margT)),margT,where='mid')


ax2.set_ylabel(r'v$_{sys}$ (km s$^{-1}$)')
ax4.set_ylabel(r'$\alpha$')
ax7.set_ylabel(r'C')
ax11.set_ylabel(r'$\theta$ ($^\circ$)')
#ax16.set_ylabel(r'$\log_{10}$ (metallicity)')
ax16.set_ylabel(r'[Fe/H]')
ax22.set_ylabel(r'F$_c$')
ax29.set_ylabel(r'C/O')


ax29.set_xlabel(r'K$_p$ (km s$^{-1}$)')
ax30.set_xlabel(r'v$_{sys}$ (km s$^{-1}$)')
ax31.set_xlabel(r'$\alpha$')
ax32.set_xlabel(r'C')
ax33.set_xlabel(r'$\theta$ ($^\circ$)')
#ax34.set_xlabel(r'$\log_{10}$ (metallicity)')
ax34.set_xlabel(r'[Fe/H]')
ax35.set_xlabel(r'F$_c$')
ax36.set_xlabel(r'C/O')


fig.savefig('%s/Corner_plot_dims_L_colorbars.pdf'%(path))
plt.close()

#########################################################

# #################################################
# ###############
# ######## Same as above but with L instead of lnL



# ncols = 8
# nrows = 8
# # grid = GridSpec(nrows, ncols,
# #                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)


# grid = GridSpec(nrows, ncols,
#                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.8, hspace=0.8)


# fig = plt.figure(0)
# fig.clf()

# # Add axes which can span multiple grid boxes
# ax1 = fig.add_subplot(grid[0, 0])

# ax2 = fig.add_subplot(grid[1, 0])
# ax3 = fig.add_subplot(grid[1, 1])

# ax4 = fig.add_subplot(grid[2, 0])
# ax5 = fig.add_subplot(grid[2, 1])
# ax6 = fig.add_subplot(grid[2, 2])

# ax7 = fig.add_subplot(grid[3, 0])
# ax8 = fig.add_subplot(grid[3, 1])
# ax9 = fig.add_subplot(grid[3, 2])
# ax10 = fig.add_subplot(grid[3, 3])

# ax11 = fig.add_subplot(grid[4, 0])
# ax12 = fig.add_subplot(grid[4, 1])
# ax13 = fig.add_subplot(grid[4, 2])
# ax14 = fig.add_subplot(grid[4, 3])
# ax15 = fig.add_subplot(grid[4, 4])

# ax16 = fig.add_subplot(grid[5, 0])
# ax17 = fig.add_subplot(grid[5, 1])
# ax18 = fig.add_subplot(grid[5, 2])
# ax19 = fig.add_subplot(grid[5, 3])
# ax20 = fig.add_subplot(grid[5, 4])
# ax21 = fig.add_subplot(grid[5, 5])

# ax22 = fig.add_subplot(grid[6, 0])
# ax23 = fig.add_subplot(grid[6, 1])
# ax24 = fig.add_subplot(grid[6, 2])
# ax25 = fig.add_subplot(grid[6, 3])
# ax26 = fig.add_subplot(grid[6, 4])
# ax27 = fig.add_subplot(grid[6, 5])
# ax28 = fig.add_subplot(grid[6, 6])

# ax29 = fig.add_subplot(grid[7, 0])
# ax30 = fig.add_subplot(grid[7, 1])
# ax31 = fig.add_subplot(grid[7, 2])
# ax32 = fig.add_subplot(grid[7, 3])
# ax33 = fig.add_subplot(grid[7, 4])
# ax34 = fig.add_subplot(grid[7, 5])
# ax35 = fig.add_subplot(grid[7, 6])
# ax36 = fig.add_subplot(grid[7, 7])


# ax1.step(np.arange(len(margk)),margk,where='mid')



# ax2.imshow(kvmapL.T,aspect='auto',origin='lower')
# ax3.step(np.arange(len(margv)),margv,where='mid')



# ax4.imshow(akmapL,aspect='auto',origin='lower')
# ax5.imshow(avmapL,aspect='auto',origin='lower')
# ax6.step(np.arange(len(marga)),marga,where='mid')


# ax7.imshow(kcmapL,aspect='auto',origin='lower')
# ax8.imshow(vcmapL,aspect='auto',origin='lower')
# ax9.imshow(acmap,aspect='auto',origin='lower')
# ax10.step(np.arange(len(margc)),margc,where='mid')



# ax11.imshow(komapL,aspect='auto',origin='lower')
# ax12.imshow(vomapL,aspect='auto',origin='lower')
# ax13.imshow(aomapL,aspect='auto',origin='lower')
# ax14.imshow(comapL.T,aspect='auto',origin='lower')
# ax15.step(np.arange(len(margo)),margo,where='mid')

# ax16.imshow(fkmapL,aspect='auto',origin='lower')
# ax17.imshow(fvmapL,aspect='auto',origin='lower')
# ax18.imshow(famapL,aspect='auto',origin='lower')
# ax19.imshow(fcmapL,aspect='auto',origin='lower')
# ax20.imshow(fomapL,aspect='auto',origin='lower')
# ax21.step(np.arange(len(margf)),margf,where='mid')

# ####

# ax22.imshow(RkmapL,aspect='auto',origin='lower')
# ax23.imshow(RvmapL,aspect='auto',origin='lower')
# ax24.imshow(RamapL,aspect='auto',origin='lower')
# ax25.imshow(RcmapL,aspect='auto',origin='lower')
# ax26.imshow(RomapL,aspect='auto',origin='lower')
# ax27.imshow(RfmapL.T,aspect='auto',origin='lower')
# ax28.step(np.arange(len(margR)),margR,where='mid')

# ax29.imshow(TkmapL,aspect='auto',origin='lower')
# ax30.imshow(TvmapL,aspect='auto',origin='lower')
# ax31.imshow(TamapL,aspect='auto',origin='lower')
# ax32.imshow(TcmapL,aspect='auto',origin='lower')
# ax33.imshow(TomapL,aspect='auto',origin='lower')
# ax34.imshow(TfmapL.T,aspect='auto',origin='lower')
# ax35.imshow(TRmapL.T,aspect='auto',origin='lower')
# ax36.step(np.arange(len(margT)),margT,where='mid')


# ax2.set_ylabel(r'v$_{sys}$ (km s$^{-1}$)')
# ax4.set_ylabel(r'$\alpha$')
# ax7.set_ylabel(r'C')
# ax11.set_ylabel(r'$\theta$ ($^\circ$)')
# #ax16.set_ylabel(r'$\log_{10}$ (metallicity)')
# ax16.set_ylabel(r'met')
# ax22.set_ylabel(r'F$_c$')
# ax29.set_ylabel(r'C/O')


# ax29.set_xlabel(r'K$_p$ (km s$^{-1}$)')
# ax30.set_xlabel(r'v$_{sys}$ (km s$^{-1}$)')
# ax31.set_xlabel(r'$\alpha$')
# ax32.set_xlabel(r'C')
# ax33.set_xlabel(r'$\theta$ ($^\circ$)')
# #ax34.set_xlabel(r'$\log_{10}$ (metallicity)')
# ax34.set_xlabel(r'met')
# ax35.set_xlabel(r'F$_c$')
# ax36.set_xlabel(r'C/O')



# #ax2.axes.xaxis.set_visible(False)
# #ax.axes.yaxis.set_visible(False)

# #fig.savefig('618_Fe_6FeH.pdf')
# #fig.savefig('test_corner.pdf')

# #fig.savefig('618_OffsetPm60.pdf')
# #fig.savefig('528_OffsetPm60.pdf')
# fig.savefig('%s/Corner_plot_Goyal_and_Herman_dims_L.pdf'%(path))

# #####################################################
# #############

# #################################################

# # #########################################
# # #### Corner plot showing the marginalized distributions of each 2D subplot 

# ncols = 8
# nrows = 8
# # grid = GridSpec(nrows, ncols,
# #                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)


# grid = GridSpec(nrows, ncols,
#                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.8, hspace=0.8)


# fig = plt.figure(0)
# fig.clf()

# # Add axes which can span multiple grid boxes
# ax1 = fig.add_subplot(grid[0, 0])

# ax2 = fig.add_subplot(grid[1, 0])
# ax3 = fig.add_subplot(grid[1, 1])

# ax4 = fig.add_subplot(grid[2, 0])
# ax5 = fig.add_subplot(grid[2, 1])
# ax6 = fig.add_subplot(grid[2, 2])

# ax7 = fig.add_subplot(grid[3, 0])
# ax8 = fig.add_subplot(grid[3, 1])
# ax9 = fig.add_subplot(grid[3, 2])
# ax10 = fig.add_subplot(grid[3, 3])

# ax11 = fig.add_subplot(grid[4, 0])
# ax12 = fig.add_subplot(grid[4, 1])
# ax13 = fig.add_subplot(grid[4, 2])
# ax14 = fig.add_subplot(grid[4, 3])
# ax15 = fig.add_subplot(grid[4, 4])

# ax16 = fig.add_subplot(grid[5, 0])
# ax17 = fig.add_subplot(grid[5, 1])
# ax18 = fig.add_subplot(grid[5, 2])
# ax19 = fig.add_subplot(grid[5, 3])
# ax20 = fig.add_subplot(grid[5, 4])
# ax21 = fig.add_subplot(grid[5, 5])

# ax22 = fig.add_subplot(grid[6, 0])
# ax23 = fig.add_subplot(grid[6, 1])
# ax24 = fig.add_subplot(grid[6, 2])
# ax25 = fig.add_subplot(grid[6, 3])
# ax26 = fig.add_subplot(grid[6, 4])
# ax27 = fig.add_subplot(grid[6, 5])
# ax28 = fig.add_subplot(grid[6, 6])

# ax29 = fig.add_subplot(grid[7, 0])
# ax30 = fig.add_subplot(grid[7, 1])
# ax31 = fig.add_subplot(grid[7, 2])
# ax32 = fig.add_subplot(grid[7, 3])
# ax33 = fig.add_subplot(grid[7, 4])
# ax34 = fig.add_subplot(grid[7, 5])
# ax35 = fig.add_subplot(grid[7, 6])
# ax36 = fig.add_subplot(grid[7, 7])


# ax1.step(np.arange(len(margk)),margk,where='mid')

# ax2.step(np.arange(len(np.sum(kvmapL,axis=1))),np.sum(kvmapL,axis=1),where='mid')


# ax3.step(np.arange(len(margv)),margv,where='mid')

# ax4.step(np.arange(len(np.sum(akmapL,axis=0))),np.sum(akmapL,axis=0),where='mid')

# ax5.step(np.arange(len(np.sum(avmapL,axis=0))),np.sum(avmapL,axis=0),where='mid')
# ax6.step(np.arange(len(marga)),marga,where='mid')


# ax7.step(np.arange(len(np.sum(kcmapL,axis=0))),np.sum(kcmapL,axis=0),where='mid')
# ax8.step(np.arange(len(np.sum(vcmapL,axis=0))),np.sum(vcmapL,axis=0),where='mid')
# ax9.step(np.arange(len(np.sum(acmap,axis=0))),np.sum(acmap,axis=0),where='mid')
# ax9.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
# ax10.step(np.arange(len(margc)),margc,where='mid')



# ax11.step(np.arange(len(np.sum(komapL,axis=0))),np.sum(komapL,axis=0),where='mid')
# ax12.step(np.arange(len(np.sum(vomapL,axis=0))),np.sum(vomapL,axis=0),where='mid')
# ax13.step(np.arange(len(np.sum(aomapL,axis=0))),np.sum(aomapL,axis=0),where='mid')
# ax14.step(np.arange(len(np.sum(comapL,axis=0))),np.sum(comapL,axis=0),where='mid')
# ax15.step(np.arange(len(margo)),margo,where='mid')

# ax16.step(np.arange(len(np.sum(fkmapL,axis=0))),np.sum(fkmapL,axis=0),where='mid')
# ax17.step(np.arange(len(np.sum(fvmapL,axis=0))),np.sum(fvmapL,axis=0),where='mid')
# ax18.step(np.arange(len(np.sum(famapL,axis=0))),np.sum(famapL,axis=0),where='mid')
# ax19.step(np.arange(len(np.sum(fcmapL,axis=0))),np.sum(fcmapL,axis=0),where='mid')
# ax20.step(np.arange(len(np.sum(fomapL,axis=0))),np.sum(fomapL,axis=0),where='mid')
# ax21.step(np.arange(len(margf)),margf,where='mid')

# ####

# ax22.step(np.arange(len(np.sum(RkmapL,axis=0))),np.sum(RkmapL,axis=0),where='mid')
# ax23.step(np.arange(len(np.sum(RvmapL,axis=0))),np.sum(RvmapL,axis=0),where='mid')
# ax24.step(np.arange(len(np.sum(RamapL,axis=0))),np.sum(RamapL,axis=0),where='mid')
# ax25.step(np.arange(len(np.sum(RcmapL,axis=0))),np.sum(RcmapL,axis=0),where='mid')
# ax26.step(np.arange(len(np.sum(RomapL,axis=0))),np.sum(RomapL,axis=0),where='mid')
# ax27.step(np.arange(len(np.sum(RfmapL,axis=1))),np.sum(RfmapL,axis=1),where='mid')
# ax28.step(np.arange(len(margR)),margR,where='mid')

# ax29.step(np.arange(len(np.sum(TkmapL,axis=0))),np.sum(TkmapL,axis=0),where='mid')
# ax30.step(np.arange(len(np.sum(TvmapL,axis=0))),np.sum(TvmapL,axis=0),where='mid')
# ax31.step(np.arange(len(np.sum(TamapL,axis=0))),np.sum(TamapL,axis=0),where='mid')
# ax32.step(np.arange(len(np.sum(TcmapL,axis=0))),np.sum(TcmapL,axis=0),where='mid')
# ax33.step(np.arange(len(np.sum(TomapL,axis=0))),np.sum(TomapL,axis=0),where='mid')
# ax34.step(np.arange(len(np.sum(TfmapL,axis=1))),np.sum(TfmapL,axis=1),where='mid')
# ax35.step(np.arange(len(np.sum(TRmapL,axis=1))),np.sum(TRmapL,axis=1),where='mid')
# ax36.step(np.arange(len(margT)),margT,where='mid')


# ax2.set_ylabel(r'v$_{sys}$ (km s$^{-1}$)')
# ax4.set_ylabel(r'$\alpha$')
# ax7.set_ylabel(r'C')
# ax11.set_ylabel(r'$\theta$ ($^\circ$)')
# #ax16.set_ylabel(r'$\log_{10}$ (metallicity)')
# ax16.set_ylabel(r'met')
# ax22.set_ylabel(r'F$_c$')
# ax29.set_ylabel(r'C/O')


# ax29.set_xlabel(r'K$_p$ (km s$^{-1}$)')
# ax30.set_xlabel(r'v$_{sys}$ (km s$^{-1}$)')
# ax31.set_xlabel(r'$\alpha$')
# ax32.set_xlabel(r'C')
# ax33.set_xlabel(r'$\theta$ ($^\circ$)')
# #ax34.set_xlabel(r'$\log_{10}$ (metallicity)')
# ax34.set_xlabel(r'met')
# ax35.set_xlabel(r'F$_c$')
# ax36.set_xlabel(r'C/O')



# #ax2.axes.xaxis.set_visible(False)
# #ax.axes.yaxis.set_visible(False)

# #fig.savefig('618_Fe_6FeH.pdf')
# #fig.savefig('test_corner.pdf')

# #fig.savefig('618_OffsetPm60.pdf')
# #fig.savefig('528_OffsetPm60.pdf')
# fig.savefig('%s/Corner_plot_Goyal_and_Herman_dims_marginalized.pdf'%(path))



# ############################################################################
# ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ########################################
# ### Corner plot with no imshow extent (showing dimensions)
# ### For only the Goyal dimensions 

# ncols = 6
# nrows = 6
# # grid = GridSpec(nrows, ncols,
# #                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)


# grid = GridSpec(nrows, ncols,
#                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.8, hspace=0.8)


# fig = plt.figure(0)
# fig.clf()

# # Add axes which can span multiple grid boxes
# ax1 = fig.add_subplot(grid[0, 0])

# ax2 = fig.add_subplot(grid[1, 0])
# ax3 = fig.add_subplot(grid[1, 1])

# ax4 = fig.add_subplot(grid[2, 0])
# ax5 = fig.add_subplot(grid[2, 1])
# ax6 = fig.add_subplot(grid[2, 2])

# ax7 = fig.add_subplot(grid[3, 0])
# ax8 = fig.add_subplot(grid[3, 1])
# ax9 = fig.add_subplot(grid[3, 2])
# ax10 = fig.add_subplot(grid[3, 3])

# ax11 = fig.add_subplot(grid[4, 0])
# ax12 = fig.add_subplot(grid[4, 1])
# ax13 = fig.add_subplot(grid[4, 2])
# ax14 = fig.add_subplot(grid[4, 3])
# ax15 = fig.add_subplot(grid[4, 4])

# ax16 = fig.add_subplot(grid[5, 0])
# ax17 = fig.add_subplot(grid[5, 1])
# ax18 = fig.add_subplot(grid[5, 2])
# ax19 = fig.add_subplot(grid[5, 3])
# ax20 = fig.add_subplot(grid[5, 4])
# ax21 = fig.add_subplot(grid[5, 5])


# ax1.step(np.arange(len(margk)),margk,where='mid')



# ax2.imshow(kvmap.T,aspect='auto',origin='lower')
# ax3.step(np.arange(len(margv)),margv,where='mid')



# ax4.imshow(akmap,aspect='auto',origin='lower')
# ax5.imshow(avmap,aspect='auto',origin='lower')
# ax6.step(np.arange(len(marga)),marga,where='mid')


# ax7.imshow(Rkmap,aspect='auto',origin='lower')
# ax8.imshow(Rvmap,aspect='auto',origin='lower')
# ax9.imshow(Ramap,aspect='auto',origin='lower')
# ax10.step(np.arange(len(margR)),margR,where='mid')



# ax11.imshow(Tkmap,aspect='auto',origin='lower')
# ax12.imshow(Tvmap,aspect='auto',origin='lower')
# ax13.imshow(Tamap,aspect='auto',origin='lower')
# ax14.imshow(TRmap.T,aspect='auto',origin='lower')
# ax15.step(np.arange(len(margT)),margT,where='mid')

# ax16.imshow(fkmap,aspect='auto',origin='lower')
# ax17.imshow(fvmap,aspect='auto',origin='lower')
# ax18.imshow(famap,aspect='auto',origin='lower')
# ax19.imshow(Rfmap,aspect='auto',origin='lower')
# ax20.imshow(Tfmap,aspect='auto',origin='lower')
# ax21.step(np.arange(len(margf)),margf,where='mid')

# ax2.set_ylabel(r'v$_{sys}$ (km s$^{-1}$)')
# ax4.set_ylabel(r'$\alpha$')
# ax7.set_ylabel(r'F$_C$')
# ax11.set_ylabel(r'C/O')
# ax16.set_ylabel(r'$\log_{10}$ Fe/H')


# ax16.set_xlabel(r'K$_p$ (km s$^{-1}$)')
# ax17.set_xlabel(r'v$_{sys}$ (km s$^{-1}$)')
# ax18.set_xlabel(r'$\alpha$')
# ax19.set_xlabel(r'F$_C$')
# ax20.set_xlabel(r'C/O')
# ax21.set_xlabel(r'$\log_{10}$ Fe/H')

# fig.savefig('618_and_528_Corner_plot_Only_Goyal_dims.pdf')

#######################################################


############################################################################
### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
########################################
### Corner plot using imshow extent to show axis scale 
## For only the Goyal dimensions 
# ncols = 6
# nrows = 6

# grid = GridSpec(nrows, ncols,
#                 left=0.15, bottom=0.15, right=0.94, top=0.94, wspace=0.10, hspace=0.10)


# fig = plt.figure(0)
# fig.clf()

# # Add axes which can span multiple grid boxes
# ax1 = fig.add_subplot(grid[0, 0])

# ax2 = fig.add_subplot(grid[1, 0])
# ax3 = fig.add_subplot(grid[1, 1])

# ax4 = fig.add_subplot(grid[2, 0])
# ax5 = fig.add_subplot(grid[2, 1])
# ax6 = fig.add_subplot(grid[2, 2])

# ax7 = fig.add_subplot(grid[3, 0])
# ax8 = fig.add_subplot(grid[3, 1])
# ax9 = fig.add_subplot(grid[3, 2])
# ax10 = fig.add_subplot(grid[3, 3])

# ax11 = fig.add_subplot(grid[4, 0])
# ax12 = fig.add_subplot(grid[4, 1])
# ax13 = fig.add_subplot(grid[4, 2])
# ax14 = fig.add_subplot(grid[4, 3])
# ax15 = fig.add_subplot(grid[4, 4])

# ax16 = fig.add_subplot(grid[5, 0])
# ax17 = fig.add_subplot(grid[5, 1])
# ax18 = fig.add_subplot(grid[5, 2])
# ax19 = fig.add_subplot(grid[5, 3])
# ax20 = fig.add_subplot(grid[5, 4])
# ax21 = fig.add_subplot(grid[5, 5])


# ax1.step(np.arange(len(margk)),margk,where='mid')



# ax2.imshow(kvmap.T,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],Vsys[0],Vsys[-1]])
# ax3.step(np.arange(len(margv)),margv,where='mid')



# ax4.imshow(akmap,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],alpha[0],alpha[-1]])
# ax5.imshow(avmap,aspect='auto',origin='lower')
# ax6.step(np.arange(len(marga)),marga,where='mid')


# ax7.imshow(Rkmap,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],fcs[0],fcs[-1]])
# ax8.imshow(Rvmap,aspect='auto',origin='lower')
# ax9.imshow(Ramap,aspect='auto',origin='lower')
# ax10.step(np.arange(len(margR)),margR,where='mid')



# ax11.imshow(Tkmap,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],CtoOs[0],CtoOs[-1]])
# ax12.imshow(Tvmap,aspect='auto',origin='lower')
# ax13.imshow(Tamap,aspect='auto',origin='lower')
# ax14.imshow(TRmap.T,aspect='auto',origin='lower')
# ax15.step(np.arange(len(margT)),margT,where='mid')

# ax16.imshow(fkmap,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],vmrs[0],vmrs[-1]])
# ax17.imshow(fvmap,aspect='auto',origin='lower',extent=[Vsys[0],Vsys[-1],vmrs[0],vmrs[-1]])
# ax18.imshow(famap,aspect='auto',origin='lower',extent=[alpha[0],alpha[-1],vmrs[0],vmrs[-1]])
# ax19.imshow(Rfmap,aspect='auto',origin='lower',extent=[fcs[0],fcs[-1],vmrs[0],vmrs[-1]])
# ax20.imshow(Tfmap,aspect='auto',origin='lower',extent=[CtoOs[0],CtoOs[-1],vmrs[0],vmrs[-1]])
# ax21.step(np.arange(len(margf)),margf,where='mid')

# ax2.set_ylabel(r'v$_{sys}$ (km s$^{-1}$)',labelpad=3)
# ax4.set_ylabel(r'$\alpha$',labelpad=20)
# ax7.set_ylabel(r'F$_C$',labelpad=12)
# ax11.set_ylabel(r'C/O',labelpad=12)
# ax16.set_ylabel(r'$\log_{10}$ Fe/H',labelpad=18)


# ax16.set_xlabel(r'K$_p$ (km s$^{-1}$)')
# ax17.set_xlabel(r'v$_{sys}$ (km s$^{-1}$)')
# ax18.set_xlabel(r'$\alpha$')
# ax19.set_xlabel(r'F$_C$')
# ax20.set_xlabel(r'C/O')
# ax21.set_xlabel(r'$\log_{10}$ Fe/H')

# ###########

# ax1.axes.xaxis.set_visible(False)
# ax1.axes.yaxis.set_visible(False)

# ax2.axes.xaxis.set_visible(False)


# ax3.axes.xaxis.set_visible(False)
# ax3.axes.yaxis.set_visible(False)

# ax4.axes.xaxis.set_visible(False)

# ax5.axes.xaxis.set_visible(False)
# ax5.axes.yaxis.set_visible(False)

# ax6.axes.xaxis.set_visible(False)
# ax6.axes.yaxis.set_visible(False)

# ax7.axes.xaxis.set_visible(False)

# ax8.axes.xaxis.set_visible(False)
# ax8.axes.yaxis.set_visible(False)

# ax9.axes.xaxis.set_visible(False)
# ax9.axes.yaxis.set_visible(False)

# ax10.axes.xaxis.set_visible(False)
# ax10.axes.yaxis.set_visible(False)

# ax11.axes.xaxis.set_visible(False)

# ax12.axes.xaxis.set_visible(False)
# ax12.axes.yaxis.set_visible(False)

# ax13.axes.xaxis.set_visible(False)
# ax13.axes.yaxis.set_visible(False)

# ax14.axes.xaxis.set_visible(False)
# ax14.axes.yaxis.set_visible(False)

# ax15.axes.xaxis.set_visible(False)
# ax15.axes.yaxis.set_visible(False)


# ax17.axes.yaxis.set_visible(False)
# ax18.axes.yaxis.set_visible(False)
# ax19.axes.yaxis.set_visible(False)
# ax20.axes.yaxis.set_visible(False)
# ax19.axes.yaxis.set_visible(False)

# ax21.axes.xaxis.set_visible(False)
# ax21.axes.yaxis.set_visible(False)

# #fig.suptitle('Night 20190528')
# fig.suptitle('Night 20180618')
# #fig.suptitle('Nights 20180618 and 20190528')

# #fig.savefig('618_and_528_Corner_plot_Only_Goyal_dims_WithScale.pdf')
# fig.savefig('618_Corner_plot_Only_Goyal_dims_WithScale.pdf')
# #fig.savefig('528_Corner_plot_Only_Goyal_dims_WithScale.pdf')



# ###################################################3

# #### Corner plot with only Miranda's dimensions (with metalicity replacing her VMR) 
# #### These plots have no extent to show the matrix dimensions 

# ncols = 6
# nrows = 6
# # grid = GridSpec(nrows, ncols,
# #                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)


# grid = GridSpec(nrows, ncols,
#                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.8, hspace=0.8)


# fig = plt.figure(0)
# fig.clf()

# # Add axes which can span multiple grid boxes
# ax1 = fig.add_subplot(grid[0, 0])

# ax2 = fig.add_subplot(grid[1, 0])
# ax3 = fig.add_subplot(grid[1, 1])

# ax4 = fig.add_subplot(grid[2, 0])
# ax5 = fig.add_subplot(grid[2, 1])
# ax6 = fig.add_subplot(grid[2, 2])

# ax7 = fig.add_subplot(grid[3, 0])
# ax8 = fig.add_subplot(grid[3, 1])
# ax9 = fig.add_subplot(grid[3, 2])
# ax10 = fig.add_subplot(grid[3, 3])

# ax11 = fig.add_subplot(grid[4, 0])
# ax12 = fig.add_subplot(grid[4, 1])
# ax13 = fig.add_subplot(grid[4, 2])
# ax14 = fig.add_subplot(grid[4, 3])
# ax15 = fig.add_subplot(grid[4, 4])

# ax16 = fig.add_subplot(grid[5, 0])
# ax17 = fig.add_subplot(grid[5, 1])
# ax18 = fig.add_subplot(grid[5, 2])
# ax19 = fig.add_subplot(grid[5, 3])
# ax20 = fig.add_subplot(grid[5, 4])
# ax21 = fig.add_subplot(grid[5, 5])


# ax1.step(np.arange(len(margk)),margk,where='mid')



# ax2.imshow(kvmap.T,aspect='auto',origin='lower')
# ax3.step(np.arange(len(margv)),margv,where='mid')



# ax4.imshow(akmap,aspect='auto',origin='lower')
# ax5.imshow(avmap,aspect='auto',origin='lower')
# ax6.step(np.arange(len(marga)),marga,where='mid')


# ax7.imshow(kcmap,aspect='auto',origin='lower')
# ax8.imshow(vcmap,aspect='auto',origin='lower')
# ax9.imshow(acmap,aspect='auto',origin='lower')
# ax10.step(np.arange(len(margc)),margc,where='mid')



# ax11.imshow(komap,aspect='auto',origin='lower')
# ax12.imshow(vomap,aspect='auto',origin='lower')
# ax13.imshow(aomap,aspect='auto',origin='lower')
# ax14.imshow(comap.T,aspect='auto',origin='lower')
# ax15.step(np.arange(len(margo)),margo,where='mid')

# ax16.imshow(fkmap,aspect='auto',origin='lower')
# ax17.imshow(fvmap,aspect='auto',origin='lower')
# ax18.imshow(famap,aspect='auto',origin='lower')
# ax19.imshow(fcmap,aspect='auto',origin='lower')
# ax20.imshow(fomap,aspect='auto',origin='lower')
# ax21.step(np.arange(len(margf)),margf,where='mid')

# ax2.set_ylabel(r'v$_{sys}$ (km s$^{-1}$)')
# ax4.set_ylabel(r'$\alpha$')
# ax7.set_ylabel(r'C')
# ax11.set_ylabel(r'$\theta$ ($^\circ$)')
# ax16.set_ylabel(r'$\log_{10}$ Fe/H')


# ax16.set_xlabel(r'K$_p$ (km s$^{-1}$)')
# ax17.set_xlabel(r'v$_{sys}$ (km s$^{-1}$)')
# ax18.set_xlabel(r'$\alpha$')
# ax19.set_xlabel(r'C')
# ax20.set_xlabel(r'$\theta$ ($^\circ$)')
# ax21.set_xlabel(r'$\log_{10}$ Fe/H')



# #ax2.axes.xaxis.set_visible(False)
# #ax.axes.yaxis.set_visible(False)

# #fig.savefig('618_Fe_6FeH.pdf')
# #fig.savefig('test_corner.pdf')

# #fig.savefig('618_OffsetPm60.pdf')
# fig.savefig('528_OffsetPm180.pdf')


######################################

# ### Corner plot with only Miranda's dimensions (with metalicity replacing her VMR) 
# ### These plots use imshow extent to show the axes range 

# ncols = 6
# nrows = 6
# # grid = GridSpec(nrows, ncols,
# #                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)


# grid = GridSpec(nrows, ncols,
#                 left=0.15, bottom=0.15, right=0.94, top=0.94, wspace=0.15, hspace=0.15)


# fig = plt.figure(0)
# fig.clf()

# # Add axes which can span multiple grid boxes
# ax1 = fig.add_subplot(grid[0, 0])

# ax2 = fig.add_subplot(grid[1, 0])
# ax3 = fig.add_subplot(grid[1, 1])

# ax4 = fig.add_subplot(grid[2, 0])
# ax5 = fig.add_subplot(grid[2, 1])
# ax6 = fig.add_subplot(grid[2, 2])

# ax7 = fig.add_subplot(grid[3, 0])
# ax8 = fig.add_subplot(grid[3, 1])
# ax9 = fig.add_subplot(grid[3, 2])
# ax10 = fig.add_subplot(grid[3, 3])

# ax11 = fig.add_subplot(grid[4, 0])
# ax12 = fig.add_subplot(grid[4, 1])
# ax13 = fig.add_subplot(grid[4, 2])
# ax14 = fig.add_subplot(grid[4, 3])
# ax15 = fig.add_subplot(grid[4, 4])

# ax16 = fig.add_subplot(grid[5, 0])
# ax17 = fig.add_subplot(grid[5, 1])
# ax18 = fig.add_subplot(grid[5, 2])
# ax19 = fig.add_subplot(grid[5, 3])
# ax20 = fig.add_subplot(grid[5, 4])
# ax21 = fig.add_subplot(grid[5, 5])


# ax1.step(np.arange(len(margk)),margk,where='mid')



# ax2.imshow(kvmap.T,aspect='auto',origin='lower', extent=[Kps[0],Kps[-1],Vsys[0],Vsys[-1]])
# ax3.step(np.arange(len(margv)),margv,where='mid')



# ax4.imshow(akmap,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],alpha[0],alpha[-1]])
# ax5.imshow(avmap,aspect='auto',origin='lower')
# ax6.step(np.arange(len(marga)),marga,where='mid')


# ax7.imshow(kcmap,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],contrast[0],contrast[-1]])
# ax8.imshow(vcmap,aspect='auto',origin='lower')
# ax9.imshow(acmap,aspect='auto',origin='lower')
# ax10.step(np.arange(len(margc)),margc,where='mid')



# ax11.imshow(komap,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],offset[0],offset[-1]])
# ax12.imshow(vomap,aspect='auto',origin='lower')
# ax13.imshow(aomap,aspect='auto',origin='lower')
# ax14.imshow(comap.T,aspect='auto',origin='lower')
# ax15.step(np.arange(len(margo)),margo,where='mid')

# ax16.imshow(fkmap,aspect='auto',origin='lower',extent=[Kps[0],Kps[-1],vmrs[0],vmrs[-1]])
# ax17.imshow(fvmap,aspect='auto',origin='lower',extent=[Vsys[0],Vsys[-1],vmrs[0],vmrs[-1]])
# ax18.imshow(famap,aspect='auto',origin='lower',extent=[alpha[0],alpha[-1],vmrs[0],vmrs[-1]])
# ax19.imshow(fcmap,aspect='auto',origin='lower',extent=[contrast[0],contrast[-1],vmrs[0],vmrs[-1]])
# ax20.imshow(fomap,aspect='auto',origin='lower',extent=[offset[0],offset[-1],Vsys[0],vmrs[-1]])
# ax21.step(np.arange(len(margf)),margf,where='mid')

# ax2.set_ylabel(r'v$_{sys}$ (km s$^{-1}$)',labelpad=3)
# ax4.set_ylabel(r'$\alpha$',labelpad=20)
# ax7.set_ylabel(r'C',labelpad=12)
# ax11.set_ylabel(r'$\theta$ ($^\circ$)',labelpad=6)
# ax16.set_ylabel(r'$\log_{10}$(Fe/H)',labelpad=15)



# ax16.set_xlabel(r'K$_p$ (km s$^{-1}$)')
# ax17.set_xlabel(r'v$_{sys}$ (km s$^{-1}$)')
# ax18.set_xlabel(r'$\alpha$')
# ax19.set_xlabel(r'C')
# ax20.set_xlabel(r'$\theta$ ($^\circ$)')
# ax21.set_xlabel(r'$\log_{10}$(Fe/H)')



# ax1.axes.xaxis.set_visible(False)
# ax1.axes.yaxis.set_visible(False)

# ax2.axes.xaxis.set_visible(False)


# ax3.axes.xaxis.set_visible(False)
# ax3.axes.yaxis.set_visible(False)

# ax4.axes.xaxis.set_visible(False)

# ax5.axes.xaxis.set_visible(False)
# ax5.axes.yaxis.set_visible(False)

# ax6.axes.xaxis.set_visible(False)
# ax6.axes.yaxis.set_visible(False)

# ax7.axes.xaxis.set_visible(False)

# ax8.axes.xaxis.set_visible(False)
# ax8.axes.yaxis.set_visible(False)

# ax9.axes.xaxis.set_visible(False)
# ax9.axes.yaxis.set_visible(False)

# ax10.axes.xaxis.set_visible(False)
# ax10.axes.yaxis.set_visible(False)

# ax11.axes.xaxis.set_visible(False)

# ax12.axes.xaxis.set_visible(False)
# ax12.axes.yaxis.set_visible(False)

# ax13.axes.xaxis.set_visible(False)
# ax13.axes.yaxis.set_visible(False)

# ax14.axes.xaxis.set_visible(False)
# ax14.axes.yaxis.set_visible(False)

# ax15.axes.xaxis.set_visible(False)
# ax15.axes.yaxis.set_visible(False)


# ax17.axes.yaxis.set_visible(False)
# ax18.axes.yaxis.set_visible(False)
# ax19.axes.yaxis.set_visible(False)
# ax20.axes.yaxis.set_visible(False)
# ax19.axes.yaxis.set_visible(False)

# ax21.axes.xaxis.set_visible(False)
# ax21.axes.yaxis.set_visible(False)

# #fig.suptitle('Night 20190528')
# #fig.suptitle('Night 20180618')
# fig.suptitle('Nights 20180618 and 20190528')

# #ax.axes.yaxis.set_visible(False)

# #fig.savefig('618_Fe_6FeH.pdf')
# #fig.savefig('test_corner.pdf')

# #fig.savefig('618_OffsetPm60.pdf')
# #fig.savefig('528_OffsetPm60_WithScale.pdf')
# #fig.savefig('618_OffsetPm60_WithScale.pdf')

# #fig.savefig('528_OffsetPm180_WithScale.pdf')
# #fig.savefig('618_OffsetPm180_WithScale.pdf')
# fig.savefig('618_and_528_OffsetPm180_WithScale.pdf')











#################################################

##### Print constraints
print('Format: median, lower uncert, upper uncert, significance')

try:
    print('alpha: %.2f, -%.2f, +%.2f, %.2f' % (significance(alpha, marga)))
except Exception:
    print('Best alpha is at min so cannot interpolate')

try:
    print('Kp: %.2f, -%.2f, +%.2f, %.2f' % (significance(Kps, margk)))
except Exception:
    print('Best Kp is at min so cannot interpolate')
    
try:
    print('Vsys: %.2f, -%.2f, +%.2f, %.2f' % (significance(Vsys, margv)))
except Exception:
    print('Best Vsys is at min so cannot interpolate')

try:
    print('off:  %.2f, -%.2f, +%.2f, %.2f' % (significance(offset, margo)))
except Exception:
    print('Best off is at min so cannot interpolate')


try:
    print('C: > %.2f' % (lowerlim(contrast,margc)))
except Exception:
    print('Best C is at min so cannot interpolate')
#print('C:  %.2f, -%.2f, +%.2f, %.2f' % (significance(contrast, margc)))

try:
    print('Fc:  %.2f, -%.2f, +%.2f, %.2f' % (significance(fcs, margR)))
except Exception:
    print('Best Fc is at min so cannot interpolate')


try:
    print('C/O:  %.2f, -%.2f, +%.2f, %.2f' % (significance(CtoOs, margT)))
except Exception:
    print('Best C/O is at min or max so cannot interpolate')


try:    
    print('metallicity:  %.2f, -%.2f, +%.2f, %.2f' % (significance(vmrs, margf)))
except Exception:
    print('Best met is at min so cannot interpolate')



