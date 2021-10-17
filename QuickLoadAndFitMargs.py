#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:26:44 2021

@author: ariddenharper
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 
from astropy.modeling import models, fitting
from matplotlib.gridspec import  GridSpec
import os 
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
	interp_cdf = interp1d(cdf_norm, param, bounds_error=False, fill_value=np.nan)
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

def GiveSigmaLims(param,margL):
	"""
	Calculates the 1-sigma lower limit for a bounded parameter
	based on its marginalized likelihood distribution.
	"""
	cdf = np.cumsum(margL)
	cdf_norm = cdf/cdf[-1]
	interp_cdf = interp1d(cdf_norm, param)
	lowlim = interp_cdf(np.array([0.68, 0.95, 0.997]))
	return lowlim


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




# LoadPath = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/20190528P2/Fe/GridV6/margs/'
# CondLoadPath = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/20190528P2/Fe/GridV6/cond/'

night = '618_and_528'
#night = '20180618All'
#night = '20190528P2'


grid_version = 6


LoadPath = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/%s/Fe/GridV%d/margs'%(night,grid_version)
CondLoadPath = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/%s/Fe/GridV%d/cond'%(night,grid_version)
SavePath = '../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/LLOutput/%s/Fe/GridV%d'%(night,grid_version)



marga = np.load('%s/marga.npy'%(LoadPath))
margc = np.load('%s/margc.npy'%(LoadPath))
margf = np.load('%s/margf.npy'%(LoadPath))
margk = np.load('%s/margk.npy'%(LoadPath))
margo = np.load('%s/margo.npy'%(LoadPath))
margR = np.load('%s/margR.npy'%(LoadPath))
margT = np.load('%s/margT.npy'%(LoadPath))
margv = np.load('%s/margv.npy'%(LoadPath))


acmapL = np.load('%s/acmapL.npy'%(CondLoadPath))
akmapL = np.load('%s/akmapL.npy'%(CondLoadPath))
aomapL = np.load('%s/aomapL.npy'%(CondLoadPath))
avmapL = np.load('%s/avmapL.npy'%(CondLoadPath))
comapL = np.load('%s/comapL.npy'%(CondLoadPath))
famapL = np.load('%s/famapL.npy'%(CondLoadPath))
fcmapL = np.load('%s/fcmapL.npy'%(CondLoadPath))
fkmapL = np.load('%s/fkmapL.npy'%(CondLoadPath))
fomapL = np.load('%s/fomapL.npy'%(CondLoadPath))
fvmapL = np.load('%s/fvmapL.npy'%(CondLoadPath))
kcmapL = np.load('%s/kcmapL.npy'%(CondLoadPath))
komapL = np.load('%s/komapL.npy'%(CondLoadPath))
kvmapL = np.load('%s/kvmapL.npy'%(CondLoadPath))
RamapL = np.load('%s/RamapL.npy'%(CondLoadPath))
RcmapL = np.load('%s/RcmapL.npy'%(CondLoadPath))
RfmapL = np.load('%s/RfmapL.npy'%(CondLoadPath))
RkmapL = np.load('%s/RkmapL.npy'%(CondLoadPath))
RomapL = np.load('%s/RomapL.npy'%(CondLoadPath))
RvmapL = np.load('%s/RvmapL.npy'%(CondLoadPath))
TamapL = np.load('%s/TamapL.npy'%(CondLoadPath))
TcmapL = np.load('%s/TcmapL.npy'%(CondLoadPath))
TfmapL = np.load('%s/TfmapL.npy'%(CondLoadPath))
TkmapL = np.load('%s/TkmapL.npy'%(CondLoadPath))
TomapL = np.load('%s/TomapL.npy'%(CondLoadPath))
TRmapL = np.load('%s/TRmapL.npy'%(CondLoadPath))
TvmapL = np.load('%s/TvmapL.npy'%(CondLoadPath))
vcmapL = np.load('%s/vcmapL.npy'%(CondLoadPath))
vomapL = np.load('%s/vomapL.npy'%(CondLoadPath))



### For repeating a predio of 
OffsetRepeated = np.zeros((len(offset)*2-1))
margoRepeated = np.zeros_like(OffsetRepeated)

OffsetRepeated[0:len(offset)] = offset
OffsetRepeated[len(offset):] = offset[1:]+360

margoRepeated[0:len(margo)] = margo
margoRepeated[len(margo):] = margo[1:] ### skipping -180 after using 180
 

NeededIndices = [2,15] #Up to but not including 15

NeededOffsetRepeated = OffsetRepeated[NeededIndices[0]:NeededIndices[-1]]
NeededMargoRepeated = margoRepeated[NeededIndices[0]:NeededIndices[-1]]










plt.plot(alpha,marga)
plt.title('alpha')

plt.figure()
plt.plot(contrast,margc)
plt.title('c')



plt.figure()
plt.plot(vmrs,margf)
plt.title('[Fe/H]')


plt.figure()
plt.plot(Kps,margk)
plt.title('Kp')

plt.figure()
plt.plot(offset,margo)
plt.title('offset')

plt.plot(offset+(np.max(offset)-np.min(offset)),margo,'r')
plt.title('offset')


plt.figure()
plt.plot(fcs,margR)
plt.title('recirculation')

plt.figure()
plt.plot(CtoOs,margT)
plt.title('C/O')

plt.figure()
plt.plot(Vsys,margv)
plt.title('v_sys')



    
param = vmrs
margL = margf

cdf = np.cumsum(margL)
cdf_norm = cdf/cdf[-1]
interp_cdf = interp1d(cdf_norm, param)
lowlim = interp_cdf(np.array([0.68, 0.95, 0.997]))


# raise Exception 


# ##### Print constraints
print('Format: median, lower uncert, upper uncert, significance')


print('alpha: %.2f, -%.2f, +%.2f, %.2f' % (significance(alpha, marga)))

print('Best alpha is at min so cannot interpolate')


print('Kp: %.2f, -%.2f, +%.2f, %.2f' % (significance(Kps, margk)))

    

print('Vsys: %.2f, -%.2f, +%.2f, %.2f' % (significance(Vsys, margv)))



print('off:  %.2f, -%.2f, +%.2f, %.2f' % (significance(NeededOffsetRepeated, NeededMargoRepeated)))



try:
    print('Fc:  %.2f, -%.2f, +%.2f, %.2f' % (significance(fcs, margR)))
except:
    print('Cannot find mean and uncertainties for Fc')
try:
    print('C/O:  %.2f, -%.2f, +%.2f, %.2f' % (significance(CtoOs, margT)))
except:
    print('Cannot find mean and uncertainties for C/O')
    
try:
    print('C:  %.2f, -%.2f, +%.2f, %.2f' % (significance(contrast, margc)))
except:
    print('Cannot find mean and uncertainties for C/O')


    
    

print('Limits:')
print('C (sigmas):  1: %.2f, 2: %.2f, 3: %.2f' % (GiveSigmaLims(contrast,margc)[0],GiveSigmaLims(contrast,margc)[1],GiveSigmaLims(contrast,margc)[2]))

print('Fc (sigmas):  1: %.2f, 2: %.2f, 3: %.2f' % (GiveSigmaLims(fcs, margR)[0],GiveSigmaLims(fcs, margR)[1],GiveSigmaLims(fcs, margR)[2]))
#print('C: > %.2f' % (lowerlim(contrast,margc)))

print('C/O (sigmas):  1: %.2f, 2: %.2f, 3: %.2f' % (GiveSigmaLims(CtoOs, margT)[0],GiveSigmaLims(CtoOs, margT)[1],GiveSigmaLims(CtoOs, margT)[2]))

print('[Fe/H] (sigmas):  1: %.2f, 2: %.2f, 3: %.2f' % (GiveSigmaLims(vmrs, margf)[0],GiveSigmaLims(vmrs, margf)[1],GiveSigmaLims(vmrs, margf)[2]))


    
### Original attempt at the phase folded offset 
# margo_DP = np.zeros((len(margo)*2))
# margo_DP[0:len(margo)] = margo
# margo_DP[len(margo):] = margo


# offset_DP = np.zeros((len(offset)*2))
# offset_DP[0:len(margo)] = offset
# offset_DP[len(margo):] = offset+offset[-1]

# offset_excl_last = offset[0:-1]
# margo_excl_last = margo[0:-1]

# SecondOffset = offset + (offset[-1]-offset[0])

# offset_DP = np.hstack((offset_excl_last,SecondOffset))
# margo_DP = np.hstack((margo_excl_last,margo))
    
# offsetNeededIndices = np.where((offset_DP>-100)&(offset_DP<300))

# Needed_offset_DP = offset_DP[offsetNeededIndices]
# Needed_margo_DP = margo_DP[offsetNeededIndices]



# g_init = models.Gaussian1D(amplitude=np.max(Needed_margo_DP), mean=np.mean(Needed_offset_DP), stddev=100)
# fit_g = fitting.LevMarLSQFitter()
# g = fit_g(g_init, Needed_offset_DP, Needed_margo_DP)

# plt.figure()
# plt.plot(Needed_offset_DP,Needed_margo_DP,label='margo')
# plt.plot(Needed_offset_DP, g(Needed_offset_DP), label='Gaussian fit')
# plt.xlabel('offset (theta) (degrees)')
# plt.ylabel('margo (marginalized likelihood of offset)')
# plt.legend()


# print('Gaussian fit to margo:')
# print('Mean: %.2f'%(g.mean.value))
# print('std dev: %.2f'%(g.stddev.value))



#### For trying to figure out the value, uncertainties, or limits from margf where only the first two values have high likelihood
margf2 = np.flip(margf)
vmrs2 = np.flip(np.array(vmrs))*(-1)
    
# margL = margf2
# param = vmrs2

margL = margf
param = vmrs

    
### To skip trying to interpolate the lower value at 
cdf = np.cumsum(margL)
cdf_norm = cdf/cdf[-1]
interp_cdf = interp1d(cdf_norm, param,bounds_error=False,fill_value='extrapolate')
low1sig, x_med, up1sig = interp_cdf(np.array([0.5-0.68*0.5, 0.5, 0.5+0.68*0.5]))


low1sig = interp_cdf(np.array([0.5-0.68*0.5]))
x_med = interp_cdf(np.array([0.5]))
up1sig = interp_cdf(np.array([0.5+0.68*0.5]))

uncert_low = x_med-low1sig
uncert_up = up1sig-x_med

sigmalim = interp_cdf(np.array([0.68]))



snr_low = x_med/(x_med-low1sig)
snr_up = x_med/(up1sig-x_med)


if snr_low > snr_up:
	x_med, uncert_low, uncert_up, snr_up
else:
	x_med, uncert_low, uncert_up, snr_low
    

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
                left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)


fig = plt.figure(figsize=(10,10))
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



ax2.imshow(kvmapL.T,aspect='auto',origin='lower')
ax3.step(np.arange(len(margv)),margv,where='mid')



ax4.imshow(akmapL,aspect='auto',origin='lower')
ax5.imshow(avmapL,aspect='auto',origin='lower')
ax6.step(np.arange(len(marga)),marga,where='mid')


ax7.imshow(kcmapL,aspect='auto',origin='lower')
ax8.imshow(vcmapL,aspect='auto',origin='lower')
ax9.imshow(acmapL,aspect='auto',origin='lower')
ax10.step(np.arange(len(margc)),margc,where='mid')



ax11.imshow(komapL,aspect='auto',origin='lower')
ax12.imshow(vomapL,aspect='auto',origin='lower')
ax13.imshow(aomapL,aspect='auto',origin='lower')
ax14.imshow(comapL.T,aspect='auto',origin='lower')
ax15.step(np.arange(len(margo)),margo,where='mid')

ax16.imshow(fkmapL,aspect='auto',origin='lower')
ax17.imshow(fvmapL,aspect='auto',origin='lower')
ax18.imshow(famapL,aspect='auto',origin='lower')
ax19.imshow(fcmapL,aspect='auto',origin='lower')
ax20.imshow(fomapL,aspect='auto',origin='lower')
ax21.step(np.arange(len(margf)),margf,where='mid')

####

ax22.imshow(RkmapL,aspect='auto',origin='lower')
ax23.imshow(RvmapL,aspect='auto',origin='lower')
ax24.imshow(RamapL,aspect='auto',origin='lower')
ax25.imshow(RcmapL,aspect='auto',origin='lower')
ax26.imshow(RomapL,aspect='auto',origin='lower')
ax27.imshow(RfmapL.T,aspect='auto',origin='lower')
ax28.step(np.arange(len(margR)),margR,where='mid')

ax29.imshow(TkmapL,aspect='auto',origin='lower')
ax30.imshow(TvmapL,aspect='auto',origin='lower')
ax31.imshow(TamapL,aspect='auto',origin='lower')
ax32.imshow(TcmapL,aspect='auto',origin='lower')
ax33.imshow(TomapL,aspect='auto',origin='lower')
ax34.imshow(TfmapL.T,aspect='auto',origin='lower')
ax35.imshow(TRmapL.T,aspect='auto',origin='lower')
#ax36.step(CtoOs,margT,where='mid')
ax36.step(np.arange(len(CtoOs)),margT,where='mid')

### Set axis tick locations and labels 
ax29.set_xticks(np.arange(len(Kps))[::12])
ax29.set_xticklabels([int(x) for x in list(Kps[::12])])

ax30.set_xticks(np.arange(len(Vsys))[::10])
ax30.set_xticklabels([x for x in list(Vsys[::10])])
ax2.set_yticks(np.arange(len(Vsys))[::10])
ax2.set_yticklabels([x for x in list(Vsys[::10])])

ax31.set_xticks(np.arange(len(alpha))[::20])
ax31.set_xticklabels([x for x in list(alpha[::20])])
ax4.set_yticks(np.arange(len(alpha))[::20])
ax4.set_yticklabels([x for x in list(alpha[::20])])

ax32.set_xticks(np.arange(len(contrast)))
ax32.set_xticklabels([x for x in list(contrast)])
ax7.set_yticks(np.arange(len(contrast)))
ax7.set_yticklabels([x for x in list(contrast)])


ax33.set_xticks(np.arange(len(offset))[::3])
ax33.set_xticklabels([int(x) for x in list(offset[::3])])

ax11.set_yticks(np.arange(len(offset))[::3])
ax11.set_yticklabels([x for x in list(offset[::3])])


ax34.set_xticks(np.arange(len(vmrs)))
ax34.set_xticklabels([x for x in list(vmrs)])
ax16.set_yticks(np.arange(len(vmrs)))
ax16.set_yticklabels([x for x in list(vmrs)])

ax35.set_xticks(np.arange(len(fcs)))
ax35.set_xticklabels([x for x in list(fcs)])
ax22.set_yticks(np.arange(len(fcs)))
ax22.set_yticklabels([x for x in list(fcs)])


ax36.set_xticks(np.arange(len(CtoOs)))
ax36.set_xticklabels([x for x in list(CtoOs)])
ax29.set_yticks(np.arange(len(CtoOs)))
ax29.set_yticklabels([x for x in list(CtoOs)])


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

#fig.savefig('%s/Corner_plot_Goyal_and_Herman_dims_With_Scale.pdf'%(path))
fig.savefig('%s/%s_GridV%d_L_CornerPlot.pdf'%(SavePath,night,grid_version))
plt.close()

# ###########################################

# #######################################
# ### 
# # #########################################
# # #### Corner plot with no imshow extent (showing dimensions) of L and colorbars 


# ncols = 8
# nrows = 8
# # grid = GridSpec(nrows, ncols,
# #                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)


# grid = GridSpec(nrows, ncols,
#                 left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.6, hspace=0.4)


# fig = plt.figure(figsize=(20,20))
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

# AddSubplotWithColorbar(fig,ax2,kvmapL.T)
# ax3.step(np.arange(len(margv)),margv,where='mid')


# AddSubplotWithColorbar(fig,ax4,akmapL)
# AddSubplotWithColorbar(fig,ax5,avmapL)
# ax6.step(np.arange(len(marga)),marga,where='mid')


# AddSubplotWithColorbar(fig,ax7,kcmapL)
# AddSubplotWithColorbar(fig,ax8,vcmapL)
# AddSubplotWithColorbar(fig,ax9,acmapL)
# ax10.step(np.arange(len(margc)),margc,where='mid')


# AddSubplotWithColorbar(fig,ax11,komapL)
# AddSubplotWithColorbar(fig,ax12,vomapL)
# AddSubplotWithColorbar(fig,ax13,aomapL)
# AddSubplotWithColorbar(fig,ax14,comapL.T)
# ax15.step(np.arange(len(margo)),margo,where='mid')


# AddSubplotWithColorbar(fig,ax16,fkmapL)
# AddSubplotWithColorbar(fig,ax17,fvmapL)
# AddSubplotWithColorbar(fig,ax18,famapL)
# AddSubplotWithColorbar(fig,ax19,fcmapL)
# AddSubplotWithColorbar(fig,ax20,fomapL)
# ax21.step(np.arange(len(margf)),margf,where='mid')

# ####



# AddSubplotWithColorbar(fig,ax22,RkmapL)
# AddSubplotWithColorbar(fig,ax23,RvmapL)
# AddSubplotWithColorbar(fig,ax24,RamapL)    
# AddSubplotWithColorbar(fig,ax25,RcmapL)
# AddSubplotWithColorbar(fig,ax26,RomapL)
# AddSubplotWithColorbar(fig,ax27,RfmapL.T)
# ax28.step(np.arange(len(margR)),margR,where='mid')


# AddSubplotWithColorbar(fig,ax29,TkmapL)
# AddSubplotWithColorbar(fig,ax30,TvmapL)
# AddSubplotWithColorbar(fig,ax31,TamapL)
# AddSubplotWithColorbar(fig,ax32,TcmapL)
# AddSubplotWithColorbar(fig,ax33,TomapL)
# AddSubplotWithColorbar(fig,ax34,TfmapL.T)
# AddSubplotWithColorbar(fig,ax35,TRmapL.T)
# ax36.step(np.arange(len(margT)),margT,where='mid')


# ax2.set_ylabel(r'v$_{sys}$ (km s$^{-1}$)')
# ax4.set_ylabel(r'$\alpha$')
# ax7.set_ylabel(r'C')
# ax11.set_ylabel(r'$\theta$ ($^\circ$)')
# #ax16.set_ylabel(r'$\log_{10}$ (metallicity)')
# ax16.set_ylabel(r'[Fe/H]')
# ax22.set_ylabel(r'F$_c$')
# ax29.set_ylabel(r'C/O')


# ax29.set_xlabel(r'K$_p$ (km s$^{-1}$)')
# ax30.set_xlabel(r'v$_{sys}$ (km s$^{-1}$)')
# ax31.set_xlabel(r'$\alpha$')
# ax32.set_xlabel(r'C')
# ax33.set_xlabel(r'$\theta$ ($^\circ$)')
# #ax34.set_xlabel(r'$\log_{10}$ (metallicity)')
# ax34.set_xlabel(r'[Fe/H]')
# ax35.set_xlabel(r'F$_c$')
# ax36.set_xlabel(r'C/O')


# #fig.savefig('%s/Corner_plot_dims_L_colorbars.pdf'%(path))
# fig.savefig('test2.pdf')
# plt.close()

plt.figure()
plt.plot(margo)