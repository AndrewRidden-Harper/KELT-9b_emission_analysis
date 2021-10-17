#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:17:40 2019

@author: ariddenharper
"""

#To properly Doppler shift the CARMENES data, need to use Correctedwmatrix[i,:] = w[IndexOfOrder]*(1-(BarCor/c))	

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits 
import os 
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d
import pandas as pd
from FindBlaze import FindEmpiricalEnvelope,FindOneSidedEmpiricalEnvelope
import time
#from pcasub import pcasub 
##from HD209458bJDToPhaseFunc import JDtophase_radial_vel
from KELT9bJDToPhaseFunc import JDtophase_radial_vel
from astropy import units as u
#from specutils.utils.wcs_utils import air_to_vac, vac_to_air
from sysrem import sysrem_sub
#from spectres import spectres 
from scipy.optimize import least_squares
from MakePlanetSpecOverStarSpec import MakePlanetOverStellarSpecFast
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from Emily_sysrem import SingleOrder_MagnitudeAndrewError, MultiSYSREM_AndrewMod
from astropy.stats import median_absolute_deviation

# import matplotlib
# matplotlib.use('Agg')

def FlattenSpectrum(spec):
    
    e1 = FindEmpiricalEnvelope(spec,'median',window1=50)
    e2 = FindEmpiricalEnvelope(e1,'upper',window1=100)
    e3 = FindEmpiricalEnvelope(e2,'upper',window1=300)
    
    flattened = spec/e3
    renormedflattened = np.copy(flattened)
    
    renormedflattened = flattened/np.nanmean(flattened)
    
    return renormedflattened

def FlattenSpectrumV2(spec,SkyEmissionLineMode,window1,window2,window3):
    
    if SkyEmissionLineMode == 0: 
        e1 = FindEmpiricalEnvelope(spec,'upper',window1=window1)
    if SkyEmissionLineMode == 1: 
         e1 = FindEmpiricalEnvelope(spec,'median',window1=window1)
    e2 = FindEmpiricalEnvelope(e1,'upper',window1=window2)
    e3 = FindEmpiricalEnvelope(e2,'upper',window1=window3)    
    
    flattened = spec/e3
    
    if SkyEmissionLineMode == 1:    

        sc = sigma_clip(flattened,sigma=3,masked=False)
    
        rollmax = FindEmpiricalEnvelope(sc,'upper',window1=200)
        
        mean_max_of_flat = np.nanmean(rollmax)
    
        flattened = (1/mean_max_of_flat)*flattened   
    
    flattened[flattened<0] = 0
    
    flattened = flattened/np.nanmedian(flattened)
    
    return flattened

def FlattenSpectrumV3(spec,SkyEmissionLineMode,window1,window2,window3):
    
    SkyEmissionLineMode = 1
    
    if SkyEmissionLineMode == 0: 
        e1 = FindEmpiricalEnvelope(spec,'upper',window1=window1)
    if SkyEmissionLineMode == 1: 
         e1 = FindEmpiricalEnvelope(spec,'median',window1=window1)
    e2 = FindEmpiricalEnvelope(e1,'upper',window1=window2)
    e3 = FindEmpiricalEnvelope(e2,'upper',window1=window3)    
    
    flattened = spec/e3
    
    if SkyEmissionLineMode == 1:    

        sc = sigma_clip(flattened,sigma=3,masked=False)
    
        rollmax = FindEmpiricalEnvelope(sc,'upper',window1=200)
        
        mean_max_of_flat = np.nanmean(rollmax)
    
        flattened = (1/mean_max_of_flat)*flattened   
    
    flattened[flattened<0] = 0
    
    ###flattened = flattened/np.nanmedian(flattened)
    
    return 

def FlattenSpectrumV4(spec,window1,window2,window3):
    

    e1 = FindEmpiricalEnvelope(spec,'median',window1=window1)
    e2 = FindEmpiricalEnvelope(e1,'upper',window1=window2)
    e3 = FindEmpiricalEnvelope(e2,'upper',window1=window3)    
    
    flattened = spec/e3
    
    flattened[flattened<0] = 0
    
    renormedflattened = np.copy(flattened)
    
    renormedflattened = flattened/np.nanmean(flattened)
    
    return renormedflattened


def Write3DArrayToFits(filename,data):
    
    nspec,ncols,nslices = np.shape(data)
        
    hdu=pyfits.PrimaryHDU(np.array([]))
    hdulist_data = pyfits.HDUList([hdu])  
    
    for i in range(nslices):
        
        newhdu_data = pyfits.ImageHDU(data[:,:,i])
        hdulist_data.append(newhdu_data)
        
    hdulist_data.writeto(filename,overwrite=True)
        
    return None 

def quadratic(x, t):    
    return x[0]*t**2 + x[1]*t +x[2]

def fun(x, t, y):    
    return quadratic(x, t) -  y 
    

def BasicSigmaClip(d,nstd):
    
    nrow,ncol = np.shape(d)    
    
    x0 = np.ones(3)
    t_train = np.arange(nrow)    
    
    for i in range(ncol):
        if (i % 100) == 0:
            print('removing cosmics in column %d of %d'%(i,ncol))
            
        y_train = d[:,i]            
        
        res_robust = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(t_train, y_train))
        y_robust_fit = quadratic(res_robust.x, t_train)
        
        std = np.std(d[:,i])
        
        absdiff = np.abs(y_robust_fit-d[:,i])
        
        idx = np.where(absdiff>nstd*std)
        
        d[idx,i] = y_robust_fit[idx]
    
    return d

def MaskSpectra(FluxMatrix,Maskthreshold=0.8,SpectrumToUseForMask=1):
    
    '''
    Follwing Sanchez-Lopez (2019) who masked all wavelengths with flux in the
    spectrum with the hightest PVW (second one used in analysis, their Fig 1)
    with flux less than 80% of the contiuum. 
    '''
    
    nrows,ncols = np.shape(FluxMatrix)
    
    masked = np.copy(FluxMatrix)
    
    ReferenceSpec = FluxMatrix[SpectrumToUseForMask,:]
    
    IndicesToMask = np.where(ReferenceSpec<Maskthreshold)
    
    masked[:,IndicesToMask] = 1
    
    FractionMasked = len(IndicesToMask[0])/ncols
    
    return masked,FractionMasked

def SimpleShiftToPlanetFrame(WaveMatrix,SpecMatrix,RadialVelocities):
    
    numrows,numcols = SpecMatrix.shape
    ShiftedSpec = np.copy(SpecMatrix)
    
    for i in numrows:    
        
        w = WaveMatrix[i,:]  
        f = SpecMatrix[i,:]

        Shiftedw = w*(1-(RadialVelocities[i]/c_kms))  ### Using - sign to reverse the model injection below which uses +
        ShiftedSpec[i,:] = spectres(w,Shiftedw,f)
        
    return ShiftedSpec



def WeightColsByVarOrStdIn3DArray(array,var_or_std='var'):
    
    numrows,numcols,numdepth = np.shape(array)
    
    ColWeighted = np.copy(array)
    
    for i in range(numdepth):
        ColWeighted[:,:,i] = WeightColsByVarOrStd(array[:,:,i],var_or_std = var_or_std)
        
    return ColWeighted

def WeightColsByVarOrStd(DataToWeight,var_or_std='var'):    
   
    #DataToWeight -= 1.
    
    DataMinus1 = DataToWeight - 1.0
    
    #MeanSubtracted = np.copy(DataToWeight)

    if var_or_std == 'var':
        residual_col_var_or_std = np.var(DataMinus1,axis=0)     
    if var_or_std == 'std':
        residual_col_var_or_std = np.std(DataMinus1,axis=0)    
        
    residual_col_var_or_std[np.where(residual_col_var_or_std==0)] = 1e20     ## To replace variance values of 0 coming from columns of 0 flux 

    ResidualColWeighted = DataMinus1/residual_col_var_or_std      
       
    return ResidualColWeighted



def envelope(wav, depth, numbins):
    """
    """
    bins = np.linspace(min(wav), max(wav), numbins)
    digitized = np.digitize(wav, bins)
    bin_mins = [depth[digitized==i].min() for i in range(1, len(bins))]
    idxs = [np.where(depth==mi)[0][0] for mi in bin_mins]
    wavs = [wav[i] for i in idxs]
    F = interp1d(wavs, bin_mins, fill_value='extrapolate')
    return depth - F(wav)


def BinAndInterpolateOutliers(wave,error,depth,numbins=20,NumMADs=5,CutOnlyAbove=True,Interpolate=True,FractionRemovedAlertThreshold=0.1,FigSavePath=None,SpecNum=None):

    #outliers = np.copy(depth)
    RemovedOutliers = np.copy(depth)
    #outliers *= np.nan   
        
    bins = np.linspace(min(wave), max(wave), numbins+1)
    digitized = np.digitize(wave, bins)
    bin_mad = [stats.median_absolute_deviation(depth[digitized==i],nan_policy='omit') for i in range(1, len(bins))]
    bin_median = [np.nanmedian(depth[digitized==i]) for i in range(1, len(bins))]    
    
    bin_middles = (bins[0:-1] + bins[1:])/2.0 
    bin_width = (wave[-1] - wave[0])/numbins
    
    if not os.path.exists(FigSavePath):
        os.makedirs(FigSavePath)
    
    if FigSavePath != None:
        
        plt.figure()
        plt.plot(bin_middles,bin_median,label='median')
        plt.errorbar(bin_middles,np.array(bin_median)+NumMADs*np.array(bin_mad),xerr=bin_width,label='threshold')
        plt.plot(wave,depth,label='spec')
        plt.legend()
        plt.xlabel('wavelength (A)')
        plt.title('spec %d'%(SpecNum))
        plt.savefig('%s/spec%d.png'%(FigSavePath,SpecNum),dpi=400)
        plt.close()

    
    NumberOfInterpolatedPoints = 0
    
    for i in range(len(depth)):
        
        if digitized[i] <= len(bin_mad):    
            BinIndex = digitized[i]-1
            
        if digitized[i] > len(bin_mad):
            BinIndex = digitized[i]-2
            
        if depth[i] > bin_median[BinIndex] + NumMADs*bin_mad[BinIndex]:
            #outliers[i] = depth[i]
            RemovedOutliers[i] = np.nan
            error[i] *= UncertaintyIncreaseFactor

            NumberOfInterpolatedPoints += 1
                
        if not CutOnlyAbove:
            if depth[i] < bin_median[BinIndex] - NumMADs*bin_mad[BinIndex]:
                #outliers[i] = depth[i]
                RemovedOutliers[i] = np.nan
                error[i] *= UncertaintyIncreaseFactor
                NumberOfInterpolatedPoints += 1      
                
                
    
                
                
    RemovedOutliersSeries = pd.Series(RemovedOutliers)
    
    if Interpolate:
        InterpolatedRemovedOutliersSeries = RemovedOutliersSeries.interpolate(limit_area='inside')
    else:
        InterpolatedRemovedOutliersSeries = RemovedOutliersSeries
    
    FractionOfRemovedPoints = NumberOfInterpolatedPoints/len(depth)
    
    if FractionOfRemovedPoints > FractionRemovedAlertThreshold:        
        print()
        print('!!!!!!!!!!!!!!!!!!!!')
        print('%s %s order %d: Removed %.2f percent of the data points as outliers (max = %.2f)'%(night,arm,OrderIndex,FractionOfRemovedPoints*100,FractionRemovedAlertThreshold))
        raise Exception              
    
    return InterpolatedRemovedOutliersSeries.to_numpy(), error


        
def RemoveBlazeFunction(wav,d,PolyOrder,ClippingNumBinsNumMADs,PointsInBinForFlattening,PlotOutputDirectory=None):

    NumSpec, NumCols = d.shape

    RefSpec = np.nanmedian(d,axis=0)
        
    ### Divide by the reference spectrum
    RefDivided = np.copy(d)      
    for i in range(NumSpec):                
        RefDivided[i,:] /= RefSpec   
    
    
    ### Agressively remove outlying points from the ratio to get rid of the spikes 
    ClippedRefDivided = np.copy(RefDivided)    
    for i in range(NumSpec):        
        ClippedRefDivided[i,:] = BinAndInterpolateOutliers(wav,RefDivided[i,:],numbins=ClippingNumBinsNumMADs[0],NumMADs=ClippingNumBinsNumMADs[1],CutOnlyAbove=False,Interpolate=False,FractionRemovedAlertThreshold=ClippingNumBinsNumMADs[2])
     
        
    ### Bin the clipped ratio to smooth it even more     
    BinnedRefDivided = np.copy(ClippedRefDivided)
    for i in range(NumSpec):       
        BinnedRefDivided[i,:] = binspec(ClippedRefDivided[i,:],PointsInBinForFlattening)
        
    
    ### Fit a low order polynomial to the clipped and binned ratio (avoiding nans)
    LowOrderPolyFits = np.copy(BinnedRefDivided)    
    IndexArray = np.arange(NumCols)    
    for i in range(NumSpec):        
        SpecToFit = BinnedRefDivided[i,:]   
        SpecToFitNanIndices = np.isnan(SpecToFit)
        
        coeff = np.polyfit(IndexArray[~SpecToFitNanIndices],SpecToFit[~SpecToFitNanIndices],PolyOrder)
        LowOrderPolyFits[i,:] = np.polyval(coeff, IndexArray)


    ### Actually flatten the spectrum by dividing by the fit     
    FlattenedSpec = np.copy(BinnedRefDivided)    
    for i in range(NumSpec):    
        FlattenedSpec[i,:] = d[i,:]/LowOrderPolyFits[i,:]
        
    if PlotOutputDirectory != None:    
        
        if not os.path.exists(PlotOutputDirectory):
            os.makedirs(PlotOutputDirectory)
            
        #### Plotting 
        with PdfPages('%s/BlazeCorrectionDiagnosticPlots.pdf'%(PlotOutputDirectory)) as pdf:
             
            for i in range(NumSpec):                
                plt.plot(d[i,:])         
                
            plt.plot(RefSpec,color='black',label='reference (median) spectrum')
            plt.title('spectra to be flattened (contains %d values <= 0)'%(np.size(d[d<=0])))
            plt.legend()
            
            pdf.savefig()
            plt.close()
            
            for i in range(NumSpec): 
                plt.title('Flattened spectra (contains %d values <= 0)'%(np.size(FlattenedSpec[FlattenedSpec<=0])))
                plt.plot(FlattenedSpec[i,:])
            pdf.savefig()            
            plt.close()
            
            for i in range(NumSpec): 
                plt.figure()                
                
                plt.title('spectrum %d'%(i))
                plt.plot(RefDivided[i,:],label='ref divided')
                
                plt.plot(ClippedRefDivided[i,:],label='clipped ref divided')
                
                plt.plot(BinnedRefDivided[i,:],label='Binned ref divided')
                
                plt.plot(LowOrderPolyFits[i,:],label='quadratic fit') 
                
                plt.ylim(-0.2,0.2)
                
                plt.legend()
                
                pdf.savefig()            
                plt.close()
            

        
    return FlattenedSpec

def FindNanEdgesAndZero(data):
    
    '''
    Find the indices of the first and last columns without nans. 
    '''
    
    NotNanIndices = []
    
    numrows,numcols = np.shape(data)    
    
    for i in range(numcols):        
    
        if True not in np.isnan(data[:,i]):
            if np.std(data[:,i]) != 0:
                NotNanIndices.append(i)
            
    return [np.min(NotNanIndices),np.max(NotNanIndices)] 


def MakePlotOfStdDevAsFuncOfSysremIt(SysremOutput,ErrorInMags,name):
    with PdfPages('%s/DiagnosticPlots/StdDevFuncSysremIt_%s.pdf'%(OutputDirectory,name)) as StdDevSysremPDF:
        
        numrows,numcols,numsysrem = np.shape(SysremOutput)
        
        StdFullArray = np.zeros(numsysrem)
        StdPartialArray = np.zeros_like(StdFullArray)
        StdOnlyFromLowErrPixels = np.zeros_like(StdFullArray)
        
        
        SysremIterationVect = np.arange(numsysrem) + 1
        
        PartialHalfWidth = 50
        
        for i in range(numsysrem):
            
            StdFullArray[i] = np.nanstd(SysremOutput[:,:,i])
            StdPartialArray[i] = np.std(SysremOutput[:,int(numcols/2)-PartialHalfWidth:int(numcols/2)+PartialHalfWidth,i])
            
            SysremOutputFromThisIt = SysremOutput[:,:,i]            
            StdOnlyFromLowErrPixels[i] = np.std(SysremOutputFromThisIt[np.where(ErrorInMags<1)])
            
        plt.figure()
        plt.plot(SysremIterationVect,StdOnlyFromLowErrPixels)
        plt.plot(SysremIterationVect,StdOnlyFromLowErrPixels,'.',markersize=10)
        plt.xlabel('Number of Sysrem iterations')
        plt.ylabel('Standard deviation')
        plt.title('Standard deviation of pixels with MagErr < 1')
        plt.xticks(SysremIterationVect)
        plt.grid()
        StdDevSysremPDF.savefig()
        plt.close()  
        
        plt.figure()
        plt.plot(SysremIterationVect,StdFullArray)
        plt.plot(SysremIterationVect,StdFullArray,'.',markersize=10)
        plt.xlabel('Number of Sysrem iterations')
        plt.ylabel('Standard deviation')
        plt.title('Standard deviation over whole order')
        plt.xticks(SysremIterationVect)
        plt.grid()
        StdDevSysremPDF.savefig()
        plt.close()
        
        plt.figure()
        plt.plot(SysremIterationVect,StdPartialArray)
        plt.plot(SysremIterationVect,StdPartialArray,'.',markersize=10)
        plt.xlabel('Number of Sysrem iterations')
        plt.ylabel('Standard deviation')
        plt.title('Standard deviation central %d pixels'%(PartialHalfWidth*2))
        plt.xticks(SysremIterationVect)
        plt.grid()
        StdDevSysremPDF.savefig()
        plt.close()  
        

def BlazeCorrectionV2(d,order,PolyfitOrder,FigSavePath):
    
    if not os.path.exists(FigSavePath):
        os.makedirs(FigSavePath)

    if arm == 'nir':
        NumOrders = 28
    if arm == 'vis':
        NumOrders = 61    
    
    
    NumMADSsThreshold = 7.5
    #PolyOrder = 10
    PolyOrder = PolyfitOrder
    
    binsize=80
    
    RefCutoffThresholdVect = np.ones(NumOrders)*0.7    
   
    if (arm == 'nir'):
        

        RefCutoffThresholdVect[0] = 0.5
    
        
        RefCutoffThresholdVect[9] = 0.9
        RefCutoffThresholdVect[10] = 0.8
        
        RefCutoffThresholdVect[18] = 0.3
        RefCutoffThresholdVect[19] = 0.3
        RefCutoffThresholdVect[20] = 0.3
        
        RefCutoffThresholdVect[27] = 0.3
        
        if night == '20190528P2':
            RefCutoffThresholdVect[8] = 0.6
            RefCutoffThresholdVect[9] = 0.3
            RefCutoffThresholdVect[10] = 0.2
            RefCutoffThresholdVect[26] = 0.5


       
    if (arm == 'vis'):
        RefCutoffThresholdVect[24] = 0.4
        RefCutoffThresholdVect[38] = 0.85
        RefCutoffThresholdVect[54] = 0.5
        RefCutoffThresholdVect[55] = 0.2  # was 0.4
        RefCutoffThresholdVect[56] = 0.1
        RefCutoffThresholdVect[57] = 0.1
        RefCutoffThresholdVect[58] = 0.1
        RefCutoffThresholdVect[59] = 0.1
        RefCutoffThresholdVect[60] = 0.1
        
        if night == '20180618All':
            ## $$$
            ##Bl
            #RefCutoffThresholdVect[7:15] = 0.3
            RefCutoffThresholdVect[11] = 0.3 ## 0.5, 0.3 (best),  0.1 
            RefCutoffThresholdVect[24] = 0.4
            RefCutoffThresholdVect[25] = 0.5
            RefCutoffThresholdVect[49] = 0.4
            RefCutoffThresholdVect[50] = 0.5
            RefCutoffThresholdVect[52] = 0.3 ## 0.5, 0.3
            RefCutoffThresholdVect[53] = 0.1 ## 0.5, 0.3, 0.1
            RefCutoffThresholdVect[55] = 0.2
            
        if night == '20190528P2':
            RefCutoffThresholdVect[53] = 0.5  #0.5, 0.4, 0.2
            RefCutoffThresholdVect[54] = 0.4 ## 0.5, 0.4, 0.2, 0.7

    
    ### Diagnostic plots mode makes several plots of each spectrum. Was useful in development but retained in case of future use 
    DiagnosticPlots = 1
    PlotAllSpec = 1
    
    RefCutoffThreshold = RefCutoffThresholdVect[order]
    
    BlazeCorrected = np.zeros_like(d)
    BlazeCorrection = np.zeros_like(d)
    
    NumSpec,NumCols = d.shape   
    
    ##########
    ## Finding the median spectrum 
    dForMean = np.copy(d)
    for i in range(NumCols):
        
        col1 = d[:,i]
        replacementcol1 = np.copy(col1)
        
        med1 = np.median(col1)
        mad1 = stats.median_absolute_deviation(col1)
        
        idx1 = np.abs(col1-med1)>NumMADSsThreshold*mad1
        
        replacementcol1[idx1] = np.nan        
        
        dForMean[:,i] = replacementcol1   

    
    RefSpec = np.nanmean(dForMean,axis=0)   
    #RefSpec = np.nanmedian(d,axis=0) ## using the median results in bad fits for the spectra that are exactly like the median 
    
    ##########

    #####RefSpec = np.nanmedian(d,axis=0)
    #####RefSpec = np.nanmean(d,axis=0)
    
    CompleteRefSpec = np.copy(RefSpec)
    
    
    tmp = RefSpec[np.argsort(RefSpec)]
    MaxFlux = np.mean(tmp[-100:-50])
    idx_LowFlux = RefSpec<=(RefCutoffThreshold*MaxFlux)
    
    RefSpec[idx_LowFlux] = np.nan
    
    ### Divide by the reference spectrum
    RefDivided = np.copy(d)      
    # for i in range(NumSpec):                
    #     RefDivided[i,:] /= RefSpec   
    
    RefDivided=RefDivided/RefSpec[np.newaxis,: ]
    
    #ListOfSpecsToCorrect = [40]
    ListOfSpecsToCorrect = range(NumSpec)
    
    for SpecIndexToCorrect in ListOfSpecsToCorrect:
    
        data = RefDivided[SpecIndexToCorrect,:]        
        
        n_bins=int(data.shape[0]/binsize)
        #n_use=nbins+binsize
        n_use=int(n_bins*binsize)
        
        n_strt=int((data.shape[0]-n_use)/2)
        
        bin_arr=data[n_strt:n_strt+n_use].reshape(n_bins,binsize)
        
        mad=stats.median_absolute_deviation(bin_arr,axis=1)
        
        med=np.median(bin_arr,axis=1)
        
        idx=np.abs( (bin_arr-med[:,np.newaxis])/(mad[:,np.newaxis]) ) > NumMADSsThreshold    
        
        bin_arr_IncOutliers = np.copy(bin_arr)
        
        bin_arr[idx]=np.nan
        
        binned=np.nanmedian(bin_arr,axis=1)
        
        x0=np.arange(data.shape[0])
        x=x0[n_strt:n_strt+n_use].reshape(n_bins,binsize)
        #x[idx]=np.nan
        xb=np.nanmean(x,axis=1)
        
        sig_binned=np.nanstd(bin_arr,axis=1)/np.sqrt(np.sum(np.isfinite(bin_arr),axis=1))
        
        MedianSig = np.nanmedian(sig_binned)
        
        sig_binned_MAD = stats.median_absolute_deviation(sig_binned,nan_policy='omit')
        
        badidx2 = np.abs( (sig_binned-MedianSig)/(sig_binned_MAD) ) > NumMADSsThreshold
        
        binned2 = np.copy(binned)
        
        binned2[badidx2] = np.nan
        
        
        IndicesForPolyfitWithoutNans = np.isfinite(xb) & np.isfinite(binned2)
        
        coeff = np.polyfit(xb[IndicesForPolyfitWithoutNans],binned2[IndicesForPolyfitWithoutNans],PolyOrder)
        LowOrderPolyFit = np.polyval(coeff, x0)
        
        #### To replace the values of the evaluated polynomial that are beyond the limits of the binned points
        #### that the polynomial is fit to, to correct the explosion that results at the edges for high order polynomials
        PolynomialFitLimits = [np.min(xb[IndicesForPolyfitWithoutNans]),np.max(xb[IndicesForPolyfitWithoutNans])]
        PolyFitLimsInX0 = [np.argmin(np.abs(PolynomialFitLimits[0]-x0)),np.argmin(np.abs(PolynomialFitLimits[1]-x0))]
        LowOrderPolyFit[0:PolyFitLimsInX0[0]] = LowOrderPolyFit[PolyFitLimsInX0[0]]
        LowOrderPolyFit[PolyFitLimsInX0[1]:] = LowOrderPolyFit[PolyFitLimsInX0[1]]
        
        # if order in OrdersToSkip:
        #     LowOrderPolyFit[:] = 1.0
        
        CorrectedSpec = d[SpecIndexToCorrect,:]/LowOrderPolyFit
        
        BlazeCorrection[SpecIndexToCorrect,:] = LowOrderPolyFit        
        BlazeCorrected[SpecIndexToCorrect,:] = CorrectedSpec       
        
        ### Diagnostic plots mode makes several plots of each spectrum. Was useful in development but retained in case of future use 
        if DiagnosticPlots:            
            pdf = PdfPages('%s/BlazeCorrectionDiagnosticPlots_order%d.pdf'%(FigSavePath,order))    

            plt.plot(d[SpecIndexToCorrect],label='Initial spec')
            plt.plot(CompleteRefSpec,label='Complete reference spec',color='black')
            plt.plot(RefSpec,label='Used reference spec')
            plt.legend()
            plt.title('Spec %d: Initial spectrum and reference'%(SpecIndexToCorrect))
            pdf.savefig()
            plt.close()
            
            plt.figure()                
            plt.plot(bin_arr_IncOutliers.flatten(),label='Ratio with outliers')
            plt.plot(bin_arr.flatten(),label='Array to bin (after first outlier removal)')
            #plt.ylim(np.nanmin(bin_arr.flatten())*0.7,np.nanmax(bin_arr.flatten())*1.3)
            plt.legend()
            plt.title('Spec %d: Initial ratio to bin and after removing first outliers'%(SpecIndexToCorrect))
            pdf.savefig()
            plt.close()
            
            plt.figure()
            plt.plot(x0,data,label='spec/ref',color='gray',linewidth=1)
            plt.plot(xb,binned,color='red',linewidth=3,label='binned1 spec/ref')
            plt.plot(xb,binned2,color='blue',linewidth=3,label='binned2 spec/ref')
            plt.plot(x0,LowOrderPolyFit,label='fit',color='black',linewidth=3)
            #plt.ylim((np.nanmin(binned2)*0.7,np.nanmax(binned2)*1.3))
            plt.legend()
            plt.title('Spec %d: Binned ratios and fits'%(SpecIndexToCorrect))
            pdf.savefig()
            plt.close()
            
            plt.figure()
            plt.plot(sig_binned,label='bin std dev')
            PosLineForPlot = MedianSig+sig_binned_MAD*NumMADSsThreshold
            NegLineForPlot = MedianSig-sig_binned_MAD*NumMADSsThreshold
            plt.plot(PosLineForPlot*np.ones_like(sig_binned),label='upper limit')
            plt.plot(NegLineForPlot*np.ones_like(sig_binned),label='lower limit')
            plt.legend()
            plt.title('Spec %d: bins to exclude for high std dev (for binned2)'%(SpecIndexToCorrect))                 
            pdf.savefig()
            plt.close()
            
            plt.figure()                
            plt.plot(CorrectedSpec,label='blaze corrected')
            plt.plot(RefSpec,label='Reference spec')
            plt.plot(d[SpecIndexToCorrect],label='initial spec')
            plt.title('Spec %d: Corrected and initial spec'%(SpecIndexToCorrect))
            plt.legend()  
            pdf.savefig()
            plt.close()
    
            pdf.close()
            
    if PlotAllSpec: 
        
        plt.figure()
            
        plt.subplot(2,1,1)
        
        for i in range(NumSpec):       
            plt.plot(d[i,:])
        plt.title('Order %d uncorrected spec (ref cutoff %f)'%(order,RefCutoffThreshold))
        plt.plot(RefSpec,label='Reference spec',color='black')
        #plt.ylim((0.8*DataMin,1.2*DataMax))
        plt.legend()
    
        plt.subplot(2,1,2)    
        for i in range(NumSpec):        
            plt.plot(BlazeCorrected[i,:])
        plt.title('Order %d blaze corrected spec (poly fit order %d) (ref cutoff %f)'%(order,PolyfitOrder,RefCutoffThreshold))
        plt.plot(CompleteRefSpec,label='Complete reference spec',color='black',linewidth=1,linestyle=':')
        plt.plot(RefSpec,label='Used reference spec',color='black')
        plt.legend()  
        if arm == 'nir':
            if order == 18: plt.ylim((-0.05,0.05))
            if order == 19: plt.ylim((-0.005,0.005))
            if order == 20: plt.ylim((-0.025,0.025))
        if arm == 'vis':
            if order >=54:
                plt.ylim((-0.025,0.025))
        #plt.ylim((0.8*DataMin,1.2*DataMax))
        plt.tight_layout()
        plt.savefig('%s/BlazeCorrection_order%d.pdf'%(FigSavePath,order))
        plt.close()
        
    return BlazeCorrected


def FindEnvelopeWithBins(wave,depth,PointsPerBin=800,numbins=None,PolyOrder=50):
    
    if numbins == None:
        numbins = int(len(wave)/PointsPerBin)
    
    

    bins = np.linspace(min(wave), max(wave), numbins+1)
    digitized = np.digitize(wave, bins)
    binned_mins = [np.min(depth[digitized==i]) for i in range(1, len(bins))]  
    
    BinMiddles = (bins[0:-1] + bins[1:])/2
    
    coeff = np.polyfit(BinMiddles,binned_mins,PolyOrder)
    LowOrderPolyFit = np.polyval(coeff, wave)
    
    return LowOrderPolyFit, BinMiddles, binned_mins




plt.close('all')

#### Constants ####
c_kms = 299792.458
#SystemicVelocity = -14.9 # NASA Exoplanet archive
#SystemicVelocity = -15.014614563711921-3.4#0.28745283580554265  # Gaia archive with uncertainty plus a fudge factor of 3.4 to try to make stellar lines at the correct wavelength value (i.e. putting in the star's rest frame)
##SystemicVelocity = -14.7652 ### From Sanchez-Lopez 2019, taken from Mazeh et al. (2000)
SystemicVelocity = -20.567  ## kms for KELT-9b from NASA Exoplanet Archive
PlanetInclination = 87.2*np.pi/180   ## for KELT-9b, Ahlers et al. 2020 

Rplanet = 1.891*u.Rjup 
Rstar = 2.36*u.Rsun

RplanetOverRstarSquared = ((Rplanet/Rstar)**2).decompose().value


##################

#DataOrigin = 'GRACES'
DataOrigin = 'CARMENES'
#DataOrigin = 'HARPS_s1d'



###########################################


# ModelToInject = 'KELT9b_Al_0.50_+0.0_0.55_Vrot6.63'
# ## 300, 120, 75, 450
#ModelScalingFactor = 450

# ModelToInject = 'KELT9b_Al_0.50_+0.0_0.55_Vrot6.63'
# #ModelScalingFactorList = [300, 120, 75, 450]
# ModelScalingFactorList = [1, 1, 1, 1]
# #ModelScalingFactorList = [1]



# ModelToInject = 'KELT9b_Ca_0.50_+0.0_0.55_Vrot6.63'
# ### 500, 130, 100, 600
# #ModelScalingFactorList = [500, 130, 100, 600]
# ModelScalingFactorList = [1,1,1,1]

# ModelToInject = 'KELT9b_CaII_0.50_+0.0_0.55_Vrot6.63'
# #ModelScalingFactorList = [1000, 270, 200, 1000]
# ModelScalingFactorList = [1,1,1,1]

# #ModelScalingFactorList = [3,3,3,3]

# ModelToInject = 'KELT9b_Cr_0.50_+0.0_0.55_Vrot6.63'
# #ModelScalingFactorList = [200, 50, 40, 200]
# ModelScalingFactorList = [1,1,1,1]


# ModelToInject = 'KELT9b_Fe_0.50_+0.0_0.55_Vrot6.63'
# ModelScalingFactorList = [150,40,30,150]
# #ModelScalingFactorList = [30]
# ##ModelScalingFactorList = [3,3,3,3]

# ModelToInject = 'KELT9b_FeH_main_iso_0.50_+0.0_0.55_Vrot6.63'
# ### 750	200	150	750
# #ModelScalingFactorList = [750, 200, 150, 750]
# ModelScalingFactorList = [1,1,1,1]

# ModelToInject = 'KELT9b_FeII_0.50_+0.0_0.55_Vrot6.63'
# ### 1500, 400, 300, 1500
# #ModelScalingFactor = 300
# ##ModelScalingFactorList = [1500, 400, 300, 1500]
# ModelScalingFactorList = [1,1,1,1]


ModelToInject = 'KELT9b_FeII_UsingFeI_0.50_+0.0_0.55_Vrot6.63'
### 1500, 400, 300, 1500
#ModelScalingFactor = 300
ModelScalingFactorList = [1500, 400, 300, 1500]
#ModelScalingFactorList = [1,1,1,1]




# ModelToInject = 'KELT9b_K_0.50_+0.0_0.55_Vrot6.63'
# ### 500, 130, 100, 500
# #ModelScalingFactorList = [500, 130, 100, 500]
# ModelScalingFactorList = [1,1,1,1]

# ModelToInject = 'KELT9b_Mg_0.50_+0.0_0.55_Vrot6.63'
# #ModelScalingFactorList = [400, 100, 80, 400]
# ModelScalingFactorList = [1,1,1,1]



# ModelToInject = 'KELT9b_Na_0.50_+0.0_0.55_Vrot6.63'
# ### 500, 130, 100, 500
# #ModelScalingFactorList = [500, 130, 100, 500]
# ModelScalingFactorList = [1,1,1,1]

# ModelToInject = 'KELT9b_Ti_0.50_+0.0_0.55_Vrot6.63'
# #ModelScalingFactorList = [150, 45, 25, 150]
# ModelScalingFactorList = [1,1,1,1]




# ModelToInject = 'KELT9b_TiO_all_iso_Plez_0.50_+0.0_0.55_Vrot6.63'
# ## 750, 195, 150, 750
# ## 1000, 300, 230, 1000  ## this gives a slightly weaker signal 
# ### 1600, 400, 300, 1800

# ModelScalingFactorList = [750, 195, 150, 750]
##ModelScalingFactorList = [3,3,3,3]

# ModelToInject = 'KELT9b_TiO_48_Exomol_McKemmish_0.50_+0.0_0.55_Vrot6.63'
# ### 750, 195, 150, 750
# #ModelScalingFactor = 150
# #ModelScalingFactorList = [750, 195, 150, 750]
# ModelScalingFactorList = [1,1,1,1]


# # ModelToInject = 'KELT9b_TiO_48_Plez_0.50_+0.0_0.55_Vrot6.63'
# # ModelScalingFactorList = [750, 195, 150, 750]
# # ##ModelScalingFactor = 150




# ModelToInject = 'NoModel'
# ModelScalingFactorList = [0,0,0,0]





ModelShift_kms = 0
#components_to_remove = 30 
#components_to_remove = 20
components_to_remove = 10
#components_to_remove = 1

Kp = 241.0
OrbVel = Kp/np.sin(PlanetInclination)

#night = '20180616'  ## includes a transit 


#night = '20180609All'
#night = '20180618All'
#night = '20190528All'
#night = '20190604All'

#NightList = ['20180609All','20180618All','20190528All','20190604All']
NightList = ['20180609All','20180618All','20190528P2','20190604All']


#NightList = ['20180609All','20190604All']

#NightList = ['20180609All']
#NightList = ['20180618All']
#NightList = ['20190528All']
#NightList = ['20190604All']

# ^^^
#ArmSubpartList = [['vis','A'],['nir','A'],['nir','B']]
#ArmSubpartList = [['nir','A'],['nir','B']]

ArmSubpartList = [['vis','A'],['nir','A']]

#ArmSubpartList = [['vis','A']]
#ArmSubpartList = [['nir','A']]

### Not using B subpart now (NIR is all A)
#ArmSubpartList = [['nir','B']]

FirstPartOfLoadPath = '../CrossCorrelationDataAndProcessing'
#FirstPartOfLoadPath = 'F:'


#arm = 'HARPS'


#arm_subpart = 'B' ## A or B 

ResidualMode = 'MeanAllSpec'
#ResidualMode = 'MeanOOTSpec'

EmissionOrAbsorption = 'Emission'
#EmissionOrAbsorption = 'Absorption'

### Parameters for for removing cosmic rays and correcting the blaze function 
#CosmicRayRemovalParams = (20,5,0.1) 
#CosmicRayRemovalParams = (50,5,0.03) 
CosmicRayRemovalParams = (50,5.5,0.04) 




### For the very noisy orders when the above CosmicRayRemovalParams do not work 
AlternativeCosmicrayRemovalParams = (50,10,0.20)
VisAlternativeCosmicrayRemovalParams = (50,15,0.28)


### From when emission lines were explicity manually removed 
#NirOrdersForAlternativeCosmicParameters = [8,9,18,19,20,21,22,24,26]

### From when emission lines were explicity manually removed 
NirOrdersForAlternativeCosmicParameters = [19,20]
VisOrdersToSkipCosmics = [56,57,58,59,60]

BlazeCorrectionPolyOrder_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_PolynomialOrderForBlazeCorrection.csv')

EmissionPixels_df = pd.read_csv('SysremItsConfigFiles/CARMENES_telluric_emission_pixel_indices.csv')



            
##################################
### Parameters for subtracting the continuum from the the injected spectrum 
        
### Old parameters used for all models 
# InjectedSpectrumFlattening_PointsPerBin = 400
# ## InjectedSpectrumFlattening_PolyOrder = 50 ### good if fitting a minimum envelope over the entire spectral range but drastically overfits the basically straight line over a single order 
# InjectedSpectrumFlattening_PolyOrder = 2

if ModelToInject == 'KELT9b_Cr_0.50_+0.0_0.55_Vrot6.63':
    InjectedSpectrumFlattening_PointsPerBin = 400
    InjectedSpectrumFlattening_PolyOrder = 2
    
    SecondarySubList = [['vis','A'],['nir','A'],['nir','B']]


elif ModelToInject == 'KELT9b_Fe_0.50_+0.0_0.55_Vrot6.63':
    InjectedSpectrumFlattening_PointsPerBin = 400
    InjectedSpectrumFlattening_PolyOrder = 2
    
    SecondarySubList = []
    
elif ModelToInject == 'KELT9b_Ti_0.50_+0.0_0.55_Vrot6.63':
    InjectedSpectrumFlattening_PointsPerBin = 400
    InjectedSpectrumFlattening_PolyOrder = 2
    
    SecondarySubList = [['vis','A'],['nir','A'],['nir','B']]        

    
elif ModelToInject == 'KELT9b_TiO_all_iso_Plez_0.50_+0.0_0.55_Vrot6.63':
    InjectedSpectrumFlattening_PointsPerBin = 50
    InjectedSpectrumFlattening_PolyOrder = 10      
    
    SecondarySubList = [['nir','A'],['nir','B']]
    
elif ModelToInject == 'KELT9b_TiO_48_Exomol_McKemmish_0.50_+0.0_0.55_Vrot6.63':
    InjectedSpectrumFlattening_PointsPerBin = 50
    InjectedSpectrumFlattening_PolyOrder = 10      
    
    SecondarySubList = [['nir','A'],['nir','B']]

elif ModelToInject == 'KELT9b_TiO_48_Plez_0.50_+0.0_0.55_Vrot6.63':
    InjectedSpectrumFlattening_PointsPerBin = 50
    InjectedSpectrumFlattening_PolyOrder = 10      
    
    SecondarySubList = [['nir','A'],['nir','B']]    
    
    
else: 
    InjectedSpectrumFlattening_PointsPerBin = 50
    InjectedSpectrumFlattening_PolyOrder = 10   
    
    SecondarySubList = [['vis','A'],['nir','A'],['nir','B']]
    
#####################################################

# BlazeRatioClippingParams = (10,3,0.6)
# BlazeRatioNumPointsInBin = 5
# PolyOrder = 2     
###############

FullOutput = True 

###Ni

#for NightIndex in range(len(NightList)):
for NightIndex in [1,2]:
#for NightIndex in [2]:
    night = NightList[NightIndex]
    ModelScalingFactor = ModelScalingFactorList[NightIndex] 

    for ArmSubpartIndex in range(len(ArmSubpartList)):
        
        arm = ArmSubpartList[ArmSubpartIndex][0]
        arm_subpart = ArmSubpartList[ArmSubpartIndex][1]
        
        
        if ((night == '20190528P2') & (arm == 'vis')):
            OrdersForRefinedUncertaintyFromPreSYSREM = [18, 26, 27, 28]
                
        
        else:
            OrdersForRefinedUncertaintyFromPreSYSREM = []
        
        
        BlazeCorrectionPolyOrder_Vect = BlazeCorrectionPolyOrder_df['%s_%s_%s'%(night,arm,arm_subpart)].values        
            
        if DataOrigin == 'CARMENES':   
            
            if arm == 'nir':
                FileNameString = 'nir_A'
            if arm == 'vis':
                FileNameString = 'vis_A'      
            
            DataDirectory = '%s/KELT-9b_CARMENES_emission_data/raw_data/%s'%(FirstPartOfLoadPath, night)
            SkyFiberDirectory = DataDirectory 
            
            HARPSrange = (5200,17100)
            

            
        ###########   
        
        
        if EmissionOrAbsorption == 'Absorption':
        
            if ModelToInject == 'NoModel':
                if arm == 'nir':
                    m = np.genfromtxt('RyansModels/ConvertedForXCorSubbed_R80400_HD209458b_T_1200K_H2O_e-4.dat')
                if arm == 'vis' or 'HARPS':
                    m = np.genfromtxt('RyansModels/NH3/500nmTo3um/ConvertedForXCorSubbed_R94600_HD209458b_template_NH3_no_continuum.dat')
            
            
            if ModelToInject == 'H2O':
                if arm == 'nir':
                    m = np.genfromtxt('RyansModels/ConvertedForXCorSubbed_R80400_HD209458b_T_1200K_H2O_e-4.dat')
            
            if ModelToInject == 'NH3':
                if arm == 'nir':
                    m = np.genfromtxt('RyansModels/NH3/500nmTo3um/ConvertedForXCorSubbed_R80400_HD209458b_template_NH3_no_continuum.dat')
                if arm == 'vis':
                    m = np.genfromtxt('RyansModels/NH3/500nmTo3um/ConvertedForXCorSubbed_R94600_HD209458b_template_NH3_no_continuum.dat')
            
            
            mw = m[:,0]*1e4  ### Need to scale the water wavelengths but not NH3
                
            mf = 1 - (m[:,1]*ModelScalingFactor)
                
        if EmissionOrAbsorption == 'Emission':
            
            #StellarWaveFlux = np.load('%s/KELT-9b_CARMENES_emission_data/ModelSpectra/FluxDensityPerHzStellarBlackBodySpec.npy'%(FirstPartOfLoadPath))
            #StellarWaveFlux = np.load('%s/KELT-9b_CARMENES_emission_data/ModelSpectra/KELT-9b_R94600_lte10200-4.00-0.0.PHOENIX_erg_per_s_per_cm2_per_Hz.npy'%(FirstPartOfLoadPath))
            #StellarWaveFlux = np.load('%s/KELT-9b_CARMENES_emission_data/ModelSpectra/KELT-9b_R94600_vsini111.80_epsilon0.60_lte10200-4.00-0.0.PHOENIX_erg_per_s_per_cm2_per_Hz.npy'%(FirstPartOfLoadPath))
            #StellarWaveFlux = np.load('%s/KELT-9b_CARMENES_emission_data/ModelSpectra/BB_10170K_R1e6_erg_Per_s_Per_cm2_Per_Hz.npy'%(FirstPartOfLoadPath))
            StellarWaveFlux = np.load('%s/KELT-9b_CARMENES_emission_data/ModelSpectra/BB_10170K_CarmenesRes_erg_Per_s_Per_cm2_Per_Hz.npy'%(FirstPartOfLoadPath))
    
    
    
        
            ### If not injecting a model just load the original  model including all lines 
            if ((ModelToInject == 'KELT9bEmission') or (ModelToInject == 'NoModel')):
                #PlanetWaveFlux = np.load('%s/KELT-9b_CARMENES_emission_data/ModelSpectra/pRT_flux_per_Hz.npy'%(FirstPartOfLoadPath))
                #PlanetWaveFlux = np.load('%s/KELT-9b_CARMENES_emission_data/ModelSpectra/KELT9b_FeI_pRT_flux_per_Hz.npy'%(FirstPartOfLoadPath))
                PlanetWaveFlux = np.load('%s/KELT-9b_CARMENES_emission_data/ModelSpectra/KELT9b_Fe_0.50_+0.0_0.55_Vrot6.63_CarmenesRes_pRT_flux_per_Hz.npy'%(FirstPartOfLoadPath))
               
            else:
                PlanetWaveFlux = np.load('%s/KELT-9b_CARMENES_emission_data/ModelSpectra/%s_CarmenesRes_pRT_flux_per_Hz.npy'%(FirstPartOfLoadPath,ModelToInject))
                
                
            ### Convert the model's wavelengths from um to A 
            PlanetWaveFlux[:,0] *= 1e4
            StellarWaveFlux[:,0] *= 1e4
        
            
            
            #m = np.genfromtxt('RyansModels/NH3/ConvertedForXCor_HD209458b_template_NH3_no_continuum_500to700nm.dat')
            ##mw = m[:,0]#*1e4
            
            
            # NeededIndices = np.where((mw>(HARPSrange[0]))&((mw<(HARPSrange[1]))))
            
            # Unscaledmf = m[:,1]
            # flippedmf = np.max(Unscaledmf)-Unscaledmf
            # ScalingFact = DesiredMaxModelAbs/np.max(flippedmf[NeededIndices[0][0]:NeededIndices[0][-1]])
            # ScaledFlippedmf = flippedmf*ScalingFact   
            
        ###### End load and scale model for injections 
        
        Unsortedfilelist = []
        UnsortedfilelistAllFiles = os.listdir(DataDirectory)
        if 'SpectraLostToGuidanceIssue' in UnsortedfilelistAllFiles:
            UnsortedfilelistAllFiles.remove('SpectraLostToGuidanceIssue')
        
        for filename in UnsortedfilelistAllFiles:
            if FileNameString in filename:
                Unsortedfilelist.append(filename)
                
        UnsortedFilenameMJDTupList = []
        
        if DataOrigin == 'CARMENES':
            NumberOfObsCoutner = 0
            for filename in Unsortedfilelist:
                if FileNameString in filename:
                    NumberOfObsCoutner += 1
                    
            NumObs = NumberOfObsCoutner
            
        InitialShape = np.shape(pyfits.getdata('%s/%s'%(DataDirectory,Unsortedfilelist[0])))        

        if len(InitialShape) > 1: ## Check if it's 1D with len 1 or 2d with len 2
            NumOrders, InitialNumCols = InitialShape
            

        CRPIX1list = []
        CRVAL1_list = []
        CDELT1_list = []
        
        numcolsList = []
        ListOfMinWaves = []
        ListOfMaxWaves = []
        
        ListOfWaves = []
        
        StartTime = time.time()
        ### Ensure that the file names are correctly ordered with time 
        for i in range(len(Unsortedfilelist)):     
            
            filename = Unsortedfilelist[i]
            
            fitsfile = pyfits.open('%s/%s'%(DataDirectory,filename))    
            header = fitsfile[0].header   
            
            if (DataOrigin == 'CARMENES') & (FileNameString in filename):    
                
                BJDFROMHEADER = float(header['HIERARCH CARACAL BJD'])-0.5  ## subtracting 0.5 to make it actual mjd instead of their definition of -2400000.0        
                EXPTIMEFROMHEADER = (header['EXPTIME']/2)/(3600*24)
                AdjustedBJDFromHeader = BJDFROMHEADER + EXPTIMEFROMHEADER        
                
                UnsortedFilenameMJDTupList.append((filename,AdjustedBJDFromHeader)) 
           
            if DataOrigin == 'GRACES':
                UnsortedFilenameMJDTupList.append((filename,float(header['MJDATE'])))
                
            if DataOrigin == 'HARPS-N_e2ds':
                UnsortedFilenameMJDTupList.append((filename,float(header['MJD-OBS'])))
                
            if DataOrigin == 'HARPS_s1d':
                UnsortedFilenameMJDTupList.append((filename,float(header['MJD-OBS'])))
                
                
            #header.tofile('CARMENES_KELT-9bSampleHeader.txt', sep='\n ', endcard=False, padding=False,overwrite=True)
        
         
        SortedFilenameTimeTupsList = sorted(UnsortedFilenameMJDTupList, key=lambda x: x[1])
        
        SortedFileList = [x[0] for x in SortedFilenameTimeTupsList]
        SortedMJDList = [x[1] for x in SortedFilenameTimeTupsList]    
        
        #np.savetxt('testCARMENES_bjd.txt',SortedMJDList)
        
        ### End sorting the order 
        ################
        
        phase,radv,transitlims,num_orbits = JDtophase_radial_vel(np.array(SortedMJDList)+2400000.5,vorbital=OrbVel)
        
        NettModelRV = np.copy(radv)
        
        MeanAirMass = []
        BarycentricRVcorrection = np.zeros((len(SortedMJDList)))
        HeliocentricJD_UTCMidExposure = []
        exptime = []
        
        TargetObjectList = []
        PointingRAList = []
        PointingDECList = []
        
        CAHAPointingRAList = []
        CAHAPointingDECList = []
        
        ############################
        ### Find the number of columns     
        filename = SortedFileList[0]
        
        
        fitsfile = pyfits.open('%s/%s'%(DataDirectory,filename))
        
        if DataOrigin == 'CARMENES':
            
            d = fitsfile[1].data 
        
            NumOrders,NumCols = np.shape(d)
            
            ### If the subpart is needed for the nir arm 
            # if arm == 'nir':
            #     if arm_subpart == 'A':
            #         NumCols = MiddleGapPixels[0] 
            #     if arm_subpart == 'B':
            #         NumCols = NumCols - MiddleGapPixels[1]            
        
        else:
            d = fitsfile[0].data 
            
            if DataOrigin == 'GRACES':
                nrows,NumCols = np.shape(d)
            if DataOrigin == 'HARPS-N':
                NumCols = len(d)
            if DataOrigin == 'HARPS-N_e2ds':
                NumCols = np.shape(d)[1]        
            if DataOrigin == 'HARPS_s1d':
                NumCols = InitialNumCols
                
        ### End the number of columns
        ############################
            
        # ### Start Order loop here 
        
        
        ###########################################################################
        ##########################################################################
        ##### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ##### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ###########################################################################
        # ***
         
        
        #ListOfOrdersToDo = range(30)                    
        #ListOfOrdersToDo = list(range(30,NumOrders)) 
        ListOfOrdersToDo = range(NumOrders)
        
        # ListOfOrdersToDo = range(15)                    
        # #ListOfOrdersToDo = range(15,30)
        # #ListOfOrdersToDo = range(30,45)
        # #ListOfOrdersToDo = range(45,NumOrders)
        
        #ListOfOrdersToDo = range(8)                    
        #ListOfOrdersToDo = range(8,16)
        #ListOfOrdersToDo = range(16,24)
        #ListOfOrdersToDo = range(24,32)
        #ListOfOrdersToDo = range(32,40)
        #ListOfOrdersToDo = range(40,48)
        #ListOfOrdersToDo = range(48,56)
        #ListOfOrdersToDo = range(56,NumOrders)
                
        #ListOfOrdersToDo = range(25,30)
        #ListOfOrdersToDo = range(30,35)
        #ListOfOrdersToDo = range(35,40)
#        ListOfOrdersToDo = range(40,45)
                
        #ListOfOrdersToDo = range(56,NumOrders)
                
        ###Li
        #ListOfOrdersToDo = [27]
                




        
                
        #ListOfOrdersToDo = [60]
            
        
        #ListOfOrdersToDo = list(range(30,NumOrders)) 
        #ListOfOrdersToDo = range(NumOrders)
        
        
        #### Need to do order 0 to get the airmasses
        #ListOfOrdersToDo = range(14) 
        #ListOfOrdersToDo = [0]+list(range(14,NumOrders)) 
        #ListOfOrdersToDo = [23]
        #ListOfOrdersToDo = [0,1]
        
        #ListOfOrdersToDo = range(4) 
        #ListOfOrdersToDo = range(4,8)
        #ListOfOrdersToDo = range(8,12) 
        #ListOfOrdersToDo = range(12,16) 
        #ListOfOrdersToDo = range(16,20) 
        #ListOfOrdersToDo = range(20,24)
        #ListOfOrdersToDo = range(24,NumOrders)

        #ListOfOrdersToDo = range(18,20)
    
        for OrderIndex in ListOfOrdersToDo:
        #for OrderIndex in range(53,NumOrders):
        
            ### Arrays to store the processed data     
            InitialFluxArray = np.empty((NumObs,NumCols))
            InitialWaveArray = np.empty((NumObs,NumCols))
            InitialUncertaintyArray = np.empty((NumObs,NumCols))
            
            
            
            FluxWithInjectedArray = np.empty_like(InitialFluxArray)   
          
            ShiftedFluxArray = np.empty_like(InitialFluxArray)
            
            ShiftedUcertaintyArray = np.empty_like(InitialFluxArray)
            
            CosmicsRemovedArray = np.empty_like(InitialFluxArray)   
            
            WaveArray = np.empty_like(InitialFluxArray)
            InjectionArray = np.ones_like(InitialFluxArray)
            
            if DataOrigin == 'CARMENES':
                SkyWaveArray = np.empty_like(InitialFluxArray)
                SkyFluxArray = np.empty_like(InitialFluxArray)
                SkyFluxUncArray = np.empty_like(InitialFluxArray)
            ###   
            
            ##############################################################
            ### Define the input and output paths for each "DataOrigin" 
                
            if DataOrigin == 'CARMENES':
                if FullOutput == True:
                    #OutputDirectory = '%s/ProcessedData/%s/%s/%s/ModelScalingFactor%0.5e/ModelShift_%.3e_kms/Order%d/'%(DataOrigin,ModelToInject,night,arm,ModelScalingFactor,ModelShift_kms,OrderIndex)
                    #OutputDirectory = 'E:/%s/ProcessedData/%s/%s/%s/ModelScalingFactor%0.5e/ModelShift_%.3e_kms/Order%d/'%(DataOrigin,ModelToInject,night,arm,ModelScalingFactor,ModelShift_kms,OrderIndex)
                    
                    OutputDirectory = '%s/KELT-9b_CARMENES_emission_data/ProcessedData/%s/%s/%s/Part%s/ModelScalingFactor%0.5e/ModelShift_%.3e_kms/Order%d/'%(FirstPartOfLoadPath,ModelToInject,night,arm,arm_subpart,ModelScalingFactor,ModelShift_kms,OrderIndex)
           
                    
                    if not os.path.exists(OutputDirectory):
                        os.makedirs(OutputDirectory)
    
                mjdOutputDir = '%s/KELT-9b_CARMENES_emission_data/ProcessedData/%s/%s/%s'%(FirstPartOfLoadPath,ModelToInject,night,arm) 
                
            ##############################################
            ### Loop over all of the spectrum files to construct matrices where each 
            ### row is a 1D spectrum. Different DataOrigins have different spectrum
            ### file formats so the wavelength, flux and flux uncertainty (if available)
            ### need to be extracted on a per DataOrigin basis.
            ###
            ### Once the wavelength, flux and flux uncertainty (if available) have been
            ### extracted and the 2D matrix of 1D spectra has been formed, the code 
            ### proceeds in the same way for all data origins     
        
            for i in range(len(SortedFileList)):                
                
                print('doing spec %d of %d for Order %d of %d'%(i+1,len(SortedFileList),OrderIndex+1,NumOrders))
                
                filename = SortedFileList[i]
            
                fitsfile = pyfits.open('%s/%s'%(DataDirectory,filename))
                header = fitsfile[0].header            
                    
                ####if DataOrigin == 'CARMENES':                    
                SkyFiberFileName = '%s%s'%(filename[:-6],'B.fits')
                SkyFiberFile = pyfits.open('%s/%s'%(SkyFiberDirectory,SkyFiberFileName)) 
                
                 
                SkyFiberFlux = SkyFiberFile[1].data[OrderIndex,:]
                SkyFiberWave = SkyFiberFile[4].data[OrderIndex,:] 
                SkyFiberFluxUnc = SkyFiberFile[3].data[OrderIndex,:]                    

                SkyWaveArray[i,:] = SkyFiberWave
                SkyFluxArray[i,:] = SkyFiberFlux        
                SkyFluxUncArray[i,:] = SkyFiberFluxUnc
                
                if OrderIndex == ListOfOrdersToDo[0]:  
                #if OrderIndex == 0:  
                    MeanAirMass.append(float(header['AIRMASS']))
                    BarycentricRVcorrection[i] = header['HIERARCH CARACAL BERV']
                    HeliocentricJD_UTCMidExposure.append(header['HIERARCH CARACAL BJD'])
                    exptime.append(header['EXPTIME'])
                    TargetObjectList.append(header['OBJECT'])
                    PointingRAList.append(header['RA'])
                    PointingDECList.append(header['DEC'])
                    
                    CAHAPointingRAList.append(header['HIERARCH CAHA TEL POS SET RA'])
                    CAHAPointingDECList.append(header['HIERARCH CAHA TEL POS SET DEC'])                    
                
                #NetVelocity = -SystemicVelocity + header['HIERARCH CARACAL BERV'] ## The combingation of -/+, +/-, -/-, +/+ that works best.  -/- gives a closer match but the lines are not aligned in the residuals 
                BaryRV = header['HIERARCH CARACAL BERV']
               
                d = fitsfile[1].data[OrderIndex,:]
                uncertainty = fitsfile[3].data[OrderIndex,:]
                
                UncorrectedWave = fitsfile[4].data[OrderIndex,:]                             
                
                ### !!!!!!!!!!!!! 
                ### A quick hack to not do the Barycentric correction until after Sysrem 
                ShiftedWave = UncorrectedWave
                ShiftedFlux = d
                ShiftedUncertainty = uncertainty
                ###
                ### !!!!!!!!!!!!!
                    

        
                ######################################################
                ### At this point the wavelength, flux and flux uncertainty (if available)
                ### have been extracted, so they can be entered into the 2D array 
                ### and all DataOrigins can be treated in the same way        
                
                InitialFluxArray[i,:] = d
                InitialWaveArray[i,:] = UncorrectedWave
                InitialUncertaintyArray[i,:] = uncertainty
        
                ### Shifted flux is not actually shifted at this stage 
                WaveArray[i,:] = ShiftedWave        
                ShiftedFluxArray[i,:] = ShiftedFlux  
                ShiftedUcertaintyArray[i,:] = ShiftedUncertainty              
                 
                ### Inject the model spectrum 
                NettPlanetRV = radv[i] + SystemicVelocity - BarycentricRVcorrection[i] + ModelShift_kms ##  (observed RV) = (heliocentric RV) - (barycentric RV) ## checked against the RVs in the Sanchez-Lopez paper and it needs to be - barycentric to make the last 7 in-transit spectra have planet RVs within +-2.5 km/s as required 
                
                NettModelRV[i] = NettPlanetRV
                        
                if EmissionOrAbsorption == 'Emission':            
                    #Shiftedmf = 1 + MakePlanetOverStellarSpecFast(PlanetWaveFlux,StellarWaveFlux,RplanetOverRstarSquared,radv[i],SystemicVelocity,0,phase[i],PlanetInclination,ShiftedWave,ModelScalingFactor,ModelShift_kms)  ## set input barycentric correction to zero since it is now corrected for          
                    ## Below the ShiftedWave is actually not shifted yet. Therefore, BarycentricRVcorrection is passed in again 
                    ####Shiftedmf = 1 + MakePlanetOverStellarSpecFast(PlanetWaveFlux,StellarWaveFlux,RplanetOverRstarSquared,radv[i],SystemicVelocity,BarycentricRVcorrection[i],phase[i],PlanetInclination,ShiftedWave,ModelScalingFactor,ModelShift_kms)    
                    
                    ### For the update to flatten the injected spectrum
                    shifted_ratio = MakePlanetOverStellarSpecFast(PlanetWaveFlux,StellarWaveFlux,RplanetOverRstarSquared,radv[i],SystemicVelocity,BarycentricRVcorrection[i],phase[i],PlanetInclination,ShiftedWave,ModelScalingFactor,ModelShift_kms)    
                    
                    xIndexArray = np.arange(len(shifted_ratio))
                    
                    env,binmiddles,binnedmins = FindEnvelopeWithBins(xIndexArray,shifted_ratio,PointsPerBin=InjectedSpectrumFlattening_PointsPerBin,numbins=None,PolyOrder=InjectedSpectrumFlattening_PolyOrder)
                    
                    
                    ### Plots useful in development and debugging to see the minimum envelope for each spectrum
                    # plt.figure()
                    # plt.plot(PlanetWaveFlux[:,0],PlanetWaveFlux[:,1])
                    
                    # plt.figure()
                    # plt.plot(StellarWaveFlux[:,0],StellarWaveFlux[:,1])
                    
                    # plt.figure()
                    # plt.plot(xIndexArray,shifted_ratio)
                    # plt.plot(xIndexArray,env)
                    # plt.plot(binmiddles,binnedmins,'ro')         
    
                    
                    Shiftedmf = shifted_ratio - env 
                    
                    if [arm,arm_subpart] in SecondarySubList:
                        Shiftedmf -= np.sort(Shiftedmf)[100]
                    
                    Shiftedmf += 1.0
                             
                FluxWithInjectedArray[i,:] = ShiftedFlux*Shiftedmf
                InjectionArray[i,:] *= Shiftedmf
                
                ### This is the end of the loop over the file names. 
                ### Now the 2D matrices have a spectrum on every row.
                ################################################
                
            ##############################
            ### Now the whole complete 2D matrix can be used
            
            ### A quick hack to replace the screwed up spectrum 19 in night 618, arm = Vis, order = 1
            if ((night == '20180618All') & (arm == 'vis') & (OrderIndex == 1)):            
                FluxWithInjectedArray[19,:] = (FluxWithInjectedArray[18,:]+FluxWithInjectedArray[20,:])/2
                
            ### A quick hack to replace the screwed up spectra 36 in night 20190604All, arm = Vis, order = 24 - 28 (inclusive)
            if ((night == '20190604All') & (arm == 'vis') & ((OrderIndex>=24)&(OrderIndex<=28))):            
                FluxWithInjectedArray[36,:] = (FluxWithInjectedArray[35,:]+FluxWithInjectedArray[37,:])/2
    
            ### A quick hack to replace the screwed up spectra 36 in night 20190604All, arm = Nir A and B, order = 14 - 16 (inclusive)
            if ((night == '20190604All') & (arm == 'nir') & ((OrderIndex>=14)&(OrderIndex<=16))):            
                FluxWithInjectedArray[36,:] = (FluxWithInjectedArray[35,:]+FluxWithInjectedArray[37,:])/2

            
            ##### Just a test 
            ### As a test to add a "cosmic" to see if it is removed 
            ### FluxWithInjectedArray[0,1000:1005] *= 100
            #####   
            
            ### Find the indices to exclude nans at the left and right edges 
            LeftExclusionEnd,RightExclusionStart = FindNanEdgesAndZero(ShiftedFluxArray)
            ##BlazeCorrectedArray = RemoveBlazeFunction(WaveArray[0,:],CosmicsRemovedArray,PolyOrder,BlazeRatioClippingParams,BlazeRatioNumPointsInBin,PlotOutputDirectory='%s/DiagnosticPlots'%(OutputDirectory))    
            
            BlazeCorrectedArray_WithCosmics = BlazeCorrectionV2(FluxWithInjectedArray,OrderIndex,BlazeCorrectionPolyOrder_Vect[OrderIndex],'%s/DiagnosticPlots/'%(OutputDirectory))    
            


                
                
                
            
            BlazeCorrectedArray_WithCosmicsAndEmissionToSave = np.copy(BlazeCorrectedArray_WithCosmics)
            
            ### Not actually shifted but a hold over from when the spectra were initially shifted into the barycentric rest frame 
            ShiftedFractionalUncertainty = ShiftedUcertaintyArray/FluxWithInjectedArray  

            
            
            
            
            ### Replace any nans in uncertainty with 0.999 fractional flux uncertainty 
            ShiftedFractionalUncertainty[np.where(np.isnan(ShiftedFractionalUncertainty))] = 0.999        
                   
            BlazeCorrectedUncArray = BlazeCorrectedArray_WithCosmics*ShiftedFractionalUncertainty   
            UncertaintyIncreaseFactor = 1e4  
            
            ###########
            ### Some spectra in VIS have a nan pixel in the same place in all spectra. e.g., 3606 in VIS order 25.
            ### Set that column to have uncertainty UncertaintyIncreaseFactor* the median uncertainty value 
            if True in np.isnan(BlazeCorrectedUncArray):
                
                nr2, nc2 = BlazeCorrectedUncArray.shape
                
                print('')
                print('Replacing unc nans')
                
                for i in range(nr2):
                    
                    UncToFix = BlazeCorrectedUncArray[i,:]
                    NanIndices = np.where(np.isnan(UncToFix))
                    
                    UncToFix[NanIndices] = UncertaintyIncreaseFactor*np.nanmedian(UncToFix)
                    BlazeCorrectedUncArray[i,:] = UncToFix
            ### End replacing the uncertainties on pixels with containing nan instead of data
            ####################


                    
            
            
            
            ####
            ### NIR Orders 19 and 20 and VIS orders 56 - 59 have no flux and its just noise so do clipping             
            if arm == 'nir':
                if ((OrderIndex == 19) or (OrderIndex == 20)):
                    BlazeCorrectedArray_WithCosmics[BlazeCorrectedArray_WithCosmics<0] = 0.0
                    BlazeCorrectedUncArray[BlazeCorrectedArray_WithCosmics<0] *= UncertaintyIncreaseFactor
                    
                    Bmad = stats.median_absolute_deviation(BlazeCorrectedArray_WithCosmics,nan_policy='omit')
                    Bmedian = np.nanmedian(BlazeCorrectedArray_WithCosmics)
                    BNumMads = 5
                    
                    OutlyingPoints = np.where((np.abs(BlazeCorrectedArray_WithCosmics-Bmedian)/Bmad)>BNumMads)
                    BlazeCorrectedArray_WithCosmics[OutlyingPoints] = Bmedian
                    BlazeCorrectedUncArray[OutlyingPoints] *= UncertaintyIncreaseFactor
                    
            if arm == 'vis':
                if OrderIndex in [56,57,58,59,60]:
                    print('')
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('Should be cleaning it up')
                    print()

                    BlazeCorrectedArray_WithCosmics[BlazeCorrectedArray_WithCosmics<0] = 0.0
                    BlazeCorrectedUncArray[BlazeCorrectedArray_WithCosmics<0] *= UncertaintyIncreaseFactor
                    
                    Bmad = stats.median_absolute_deviation(BlazeCorrectedArray_WithCosmics,nan_policy='omit')
                    Bmedian = np.nanmedian(BlazeCorrectedArray_WithCosmics)
                    BNumMads = 5
                    
                    OutlyingPoints = np.where((np.abs(BlazeCorrectedArray_WithCosmics-Bmedian)/Bmad)>BNumMads)
                    BlazeCorrectedArray_WithCosmics[OutlyingPoints] = Bmedian
                    BlazeCorrectedUncArray[OutlyingPoints] *= UncertaintyIncreaseFactor

                    
                    
                    
                    
            ################
            #################################################################
            ### To remove the emission lines in the NIR arm
                

                
            
            if arm == 'nir':               
                
                EmissionPixelsList = EmissionPixels_df['order%d'%(OrderIndex)]
                
                #EmissionPixelsList = EmissionPixels[np.where(np.isfinite(EmissionPixels))]
                
                for i in range(len(EmissionPixelsList)):
                
                    if type(EmissionPixelsList[i]) == type(''):
                    
                        StartIndex = int(EmissionPixelsList[i].split(',')[0])
                        EndIndex = int(EmissionPixelsList[i].split(',')[-1])
                        
                        ReplaceValue = (np.nanmedian(BlazeCorrectedArray_WithCosmics[:,StartIndex-1]) + \
                            np.nanmedian(BlazeCorrectedArray_WithCosmics[:,EndIndex]))/2.0

                        BlazeCorrectedArray_WithCosmics[:,StartIndex:EndIndex] = ReplaceValue
                        BlazeCorrectedUncArray[:,StartIndex:EndIndex] *= UncertaintyIncreaseFactor
                        

                        
            BlazeCorrectedArray_NoEmission_WithCosmics_ToSave = np.copy(BlazeCorrectedArray_WithCosmics)
            
            
            ### End replacing emission lines in NIR arm
            ####################################################################
            ### Manually remove small cosmic rays that are difficult to automacially remove 
            
            if night == '20180618All':
                if arm == 'nir':
                    if OrderIndex == 22:
                        BlazeCorrectedArray_WithCosmics[6,1318:1333] = np.mean([BlazeCorrectedArray_WithCosmics[6,1317],BlazeCorrectedArray_WithCosmics[6,1334]]) 
                        BlazeCorrectedUncArray[6,1318:1333] *= UncertaintyIncreaseFactor
                        
                    if OrderIndex == 23:
                        BlazeCorrectedArray_WithCosmics[6,1140:1154] = np.mean([BlazeCorrectedArray_WithCosmics[6,1139],BlazeCorrectedArray_WithCosmics[6,1155]]) 
                        BlazeCorrectedUncArray[6,1140:1154] *= UncertaintyIncreaseFactor
                        
            
            if night == '20190528P2':
                if arm == 'nir':
                    if OrderIndex == 12:
                        BlazeCorrectedArray_WithCosmics[9,386:401] = np.mean([BlazeCorrectedArray_WithCosmics[9,385],BlazeCorrectedArray_WithCosmics[9,402]]) 
                        BlazeCorrectedUncArray[9,386:401] *= UncertaintyIncreaseFactor
                        
                        BlazeCorrectedArray_WithCosmics[7,3404:3409] = np.mean([BlazeCorrectedArray_WithCosmics[7,3403],BlazeCorrectedArray_WithCosmics[7,3410]]) 
                        BlazeCorrectedUncArray[7,3404:3409] *= UncertaintyIncreaseFactor
                        
                        BlazeCorrectedArray_WithCosmics[17,2179:2191] = np.mean([BlazeCorrectedArray_WithCosmics[17,2178],BlazeCorrectedArray_WithCosmics[17,2192]]) 
                        BlazeCorrectedUncArray[17,2179:2191] *= UncertaintyIncreaseFactor

                    if OrderIndex == 14:
                        BlazeCorrectedArray_WithCosmics[6,1682:1702] = np.mean([BlazeCorrectedArray_WithCosmics[6,1681],BlazeCorrectedArray_WithCosmics[6,1703]]) 
                        BlazeCorrectedUncArray[6,1682:1702] *= UncertaintyIncreaseFactor
                        
                    if OrderIndex == 23:
                        BlazeCorrectedArray_WithCosmics[7,325:336] = np.mean([BlazeCorrectedArray_WithCosmics[7,324],BlazeCorrectedArray_WithCosmics[7,337]]) 
                        BlazeCorrectedUncArray[7,325:336] *= UncertaintyIncreaseFactor
                        
                        BlazeCorrectedArray_WithCosmics[0,653:663] = np.mean([BlazeCorrectedArray_WithCosmics[0,652],BlazeCorrectedArray_WithCosmics[0,664]]) 
                        BlazeCorrectedUncArray[0,653:663] *= UncertaintyIncreaseFactor


                    if OrderIndex == 26:
                        BlazeCorrectedArray_WithCosmics[22,2198:2270] = np.mean([BlazeCorrectedArray_WithCosmics[22,2197],BlazeCorrectedArray_WithCosmics[22,2271]]) 
                        BlazeCorrectedUncArray[22,2198:2270] *= UncertaintyIncreaseFactor



           #################################
            
            # ### Remove cosmic rays on a per spectrum basis (not considering specs before and after)
            nr, nc = np.shape(BlazeCorrectedArray_WithCosmics)
            
            if OrderIndex not in VisOrdersToSkipCosmics:  ##### The NIR orders only get up to 28 so there's no need for an arm check
            
                for i in range(nr):            
                    if ((arm == 'nir') & (OrderIndex in NirOrdersForAlternativeCosmicParameters)):
                        CosmicsRemovedArray[i,:], ShiftedUcertaintyArray[i,:] = BinAndInterpolateOutliers(WaveArray[i,:],ShiftedUcertaintyArray[i,:],BlazeCorrectedArray_WithCosmics[i,:],numbins=AlternativeCosmicrayRemovalParams[0],NumMADs=AlternativeCosmicrayRemovalParams[1],CutOnlyAbove=True,Interpolate=True,FractionRemovedAlertThreshold=AlternativeCosmicrayRemovalParams[2],FigSavePath='%s/DiagnosticPlots/CosmicRemovalPlots'%(OutputDirectory),SpecNum=i)
                        
                    # elif ((arm == 'vis') & (OrderIndex in VisOrdersForAlternativeCosmicParameters)):
                    #     CosmicsRemovedArray[i,:], ShiftedUcertaintyArray[i,:] = BinAndInterpolateOutliers(WaveArray[i,:],ShiftedUcertaintyArray[i,:],BlazeCorrectedArray_WithCosmics[i,:],numbins=VisAlternativeCosmicrayRemovalParams[0],NumMADs=VisAlternativeCosmicrayRemovalParams[1],CutOnlyAbove=True,Interpolate=True,FractionRemovedAlertThreshold=VisAlternativeCosmicrayRemovalParams[2],FigSavePath='%s/DiagnosticPlots/CosmicRemovalPlots'%(OutputDirectory),SpecNum=i)
            
                        
                    else:            
                        CosmicsRemovedArray[i,:], ShiftedUcertaintyArray[i,:] = BinAndInterpolateOutliers(WaveArray[i,:],ShiftedUcertaintyArray[i,:],BlazeCorrectedArray_WithCosmics[i,:],numbins=CosmicRayRemovalParams[0],NumMADs=CosmicRayRemovalParams[1],CutOnlyAbove=True,Interpolate=True,FractionRemovedAlertThreshold=CosmicRayRemovalParams[2],FigSavePath='%s/DiagnosticPlots/CosmicRemovalPlots'%(OutputDirectory),SpecNum=i)            
                BlazeCorrectedArray = np.copy(CosmicsRemovedArray)
                
            elif OrderIndex in VisOrdersToSkipCosmics:  ##### The NIR orders only get up to 28 so there's no need for an arm check
                BlazeCorrectedArray = np.copy(BlazeCorrectedArray_WithCosmics)    

            #BlazeCorrectedArray = np.copy(BlazeCorrectedArray_WithCosmics)    


            
            ###BlazeCorrectedArray = np.copy(BlazeCorrectedArray_WithCosmics)

            ##BlazeCorrectedArrayAfterCosmicAndEmission_ToSave = np.copy(BlazeCorrectedArray)

            #############################################################
            ### Identify cosmic rays by comparing all spectra and replace with the median value of the column 

            # MAD_per_col = median_absolute_deviation(BlazeCorrectedArray_WithCosmics,axis=0,ignore_nan=True)  ### This doesn't do rescaling to a normal distribution like the scipy.stats function does so 7.5 MADS is actually 5 sigma in this case 
            # Median_per_col = np.nanmedian(BlazeCorrectedArray_WithCosmics,axis=0)
            # NumMadsCosmics = 7.5            
            
            # nr, nc = BlazeCorrectedArray_WithCosmics.shape
            # NumCosmicsRemove = 0
            # for i in range(nc):
                
            #     if MAD_per_col[i] > 0:
            #         col = np.copy(BlazeCorrectedArray_WithCosmics[:,i])
            #         unccol = np.copy(BlazeCorrectedUncArray[:,i])
                    
            #         CosmicIndicesInCol = np.where(np.abs(col-Median_per_col[i])>=NumMadsCosmics*MAD_per_col[i])
                    
            #         NumCosmicsRemove += len(CosmicIndicesInCol[0])
                    
            #         if NumCosmicsRemove > 0:
                    
            #             col[CosmicIndicesInCol] = Median_per_col[i]
            #             CosmicsRemovedArray[:,i] = col            

            #             unccol[CosmicIndicesInCol] *= UncertaintyIncreaseFactor
            #             BlazeCorrectedUncArray[:,i] = unccol
                        
            # BlazeCorrectedArray = np.copy(CosmicsRemovedArray)
            
            # PercentageOfPointsContainingCosmics = 100*NumCosmicsRemove/(nc*nr) 
            # np.savetxt('%s/Order%d_Percentage_of_points_replaced_as_cosmics.txt'%(OutputDirectory,OrderIndex),np.array([PercentageOfPointsContainingCosmics]))
            
            ##BlazeCorrectedArray = np.copy(BlazeCorrectedArray_WithCosmics)
            

            # ### End per column cosmic removal
            ########################################################################
           
            NormalizedNotBlazeCorrected = np.copy(ShiftedFluxArray)  ## keeping this just for easy comparisons 
            for i in range(NumObs):
                NormalizedNotBlazeCorrected[i,:]/=np.nanmean(ShiftedFluxArray[i,:])
              
            
            #MedNormOnlyOutOfTransit = np.delete(MedNorm,transitlims[0],axis=0)
            Residual = np.copy(BlazeCorrectedArray)
            for i in range(NumCols):
                # if ResidualMode == 'MeanOOTSpec':
                #     Residual[:,i]/=np.nanmean(MedNormOnlyOutOfTransit[:,i])  ## To only divide by the average out-of-transit spectrum         
                if ResidualMode == 'MeanAllSpec':
                    Residual[:,i]/=np.nanmean(BlazeCorrectedArray[:,i])    
            ###Residual = np.nan_to_num(Residual)     
           
            
            #pcasubbed = pcasub(BlazeCorrectedArray[:,LeftExclusionEnd:RightExclusionStart],components_to_remove)
            
            MedNormToSysrem = np.copy(BlazeCorrectedArray[:,LeftExclusionEnd:RightExclusionStart+1])  ## up to but not including so need the +1
            UncArrayToSysrem = np.copy(BlazeCorrectedUncArray[:,LeftExclusionEnd:RightExclusionStart+1])    
            WaveArrayOfFluxGivenToSysrem = np.copy(WaveArray[:,LeftExclusionEnd:RightExclusionStart+1])
            
            ResidualMedNormToSysrem = MedNormToSysrem/np.median(MedNormToSysrem,axis=0)       
            
            ### !!!!!!!!!!!!!!!!!
            ### Replacing values <= 0 
            
            MedNormToSysremOnlyPositive = np.copy(MedNormToSysrem)
            UncArrayToSysremOnlyPositive = np.copy(UncArrayToSysrem)
            
            MedNormToSysremOnlyPositiveInitial = np.copy(MedNormToSysremOnlyPositive)
            UncArrayToSysremOnlyPositiveInitial = np.copy(UncArrayToSysremOnlyPositive)
            
            LowestPositiveValue = np.min(MedNormToSysrem[np.where(MedNormToSysrem>0)])
            
            HighestUncertainty = np.nanmax(UncArrayToSysremOnlyPositive)
            
            
            ### Replace the nans in flux and increase their corresponding uncertainty
            for WavelengthIndex in range(np.shape(MedNormToSysrem)[1]):
                
                col = np.copy(MedNormToSysrem[:,WavelengthIndex])
                unccol = np.copy(UncArrayToSysrem[:,WavelengthIndex])            
                IndicesOfNans = np.where(np.isnan(col))
                
                if len(IndicesOfNans[0]) == len(col):
                    # print('!!!!!!!!!!!!!!!!!!!!!!!')
                    # print('WavelengthIndex %d has column of nans'%(WavelengthIndex))
                    col[IndicesOfNans] = np.nanmedian(MedNormToSysrem)
                    
                    MedNormToSysremOnlyPositive[:,WavelengthIndex] = col
                    
                    unccol[IndicesOfNans] = HighestUncertainty*UncertaintyIncreaseFactor
                    UncArrayToSysremOnlyPositive[:,WavelengthIndex] = unccol            
                
                
                if len(IndicesOfNans[0]) > 0:
                    # print('!!!!!!!!!!!!!!!!!!!!!!!')
                    # print('WavelengthIndex %d has some nans'%(WavelengthIndex))
                    
    
                    col[IndicesOfNans] = np.nanmedian(col)
                    MedNormToSysremOnlyPositive[:,WavelengthIndex] = col       
                    
                    unccol[IndicesOfNans] = HighestUncertainty*UncertaintyIncreaseFactor
                    UncArrayToSysremOnlyPositive[:,WavelengthIndex] = unccol 
                    
    
                    
            ### Replace the negative values in flux and increase their corresponding uncertainty       
            ######################       
            for WavelengthIndex in range(np.shape(MedNormToSysremOnlyPositive)[1]):
                
                col = np.copy(MedNormToSysremOnlyPositive[:,WavelengthIndex])
                unccol = np.copy(UncArrayToSysremOnlyPositive[:,WavelengthIndex])
                
                IndicesOfPosValuesInCol = np.where(col>0)
                IndicesOfNegValuesInCol = np.where(col<=0)
                
                if len(IndicesOfPosValuesInCol[0]) == 0:
                    MedNormToSysremOnlyPositive[:,WavelengthIndex] = LowestPositiveValue
                    UncArrayToSysremOnlyPositive[:,WavelengthIndex] = HighestUncertainty*UncertaintyIncreaseFactor
                    
                else:
                    col[IndicesOfNegValuesInCol] = np.median(col[IndicesOfPosValuesInCol])
                    MedNormToSysremOnlyPositive[:,WavelengthIndex] = col
                    
                    unccol[IndicesOfNegValuesInCol] = HighestUncertainty*UncertaintyIncreaseFactor
                    UncArrayToSysremOnlyPositive[:,WavelengthIndex] = unccol                              
    
            if True in np.isnan(MedNormToSysremOnlyPositive):
                print()
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('arm %s %s order %d'%(arm,arm_subpart,OrderIndex))          
                
                print('MedNormToSysremOnlyPositive contains nans')
            
            if True in np.isnan(UncArrayToSysremOnlyPositive):
                print()
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('arm %s %s order %d'%(arm,arm_subpart,OrderIndex))
    
                print('UncArrayToSysremOnlyPositive contains nans')
                raise Exception 
                
            
            ### !!!!!!!!!!!!!!!!!
            ### End replacing values <= 0         
            
            FluxInMagsNotResidual = -2.5*np.log10(MedNormToSysremOnlyPositive)
            # ########
            # #### To test with all points having equal unertainties   
            # UncArrayToSysremOnlyPositive[:] = 1.0
            # #######
            
            FluxInMagsToSysrem, FluxErrInMagsToSysrem = SingleOrder_MagnitudeAndrewError(MedNormToSysremOnlyPositive, WaveArrayOfFluxGivenToSysrem, UncArrayToSysremOnlyPositive)
            
            # #### Ernst's suggestion of setting the uncertainties for pixels 0-400 and pixel 3600 onwards to a very large number (e.g., 1e5 or higher) when running SYSREM
            ### ## and running Sysrem twice, once to get uncertainties and then rerunning with those uncertainties 
    
            AverageFluxErrInMags = np.mean(FluxErrInMagsToSysrem,axis=0)  
            
            ##ValueToReplaceAlreadyHighNoiseWith = 1e6
            SysremIterationsForInitialRunToRefineNoise = 1
            
            #### To replace columns with noise greater than a threshold of MADs. Note that the refined errors below are also changed like this
            #NumMadsPS = 7.5
            # NumMadsPS = 15
            # MedianValuePS = np.nanmedian(AverageFluxErrInMags)  
            # MADPS = stats.median_abs_deviation(AverageFluxErrInMags,nan_policy='omit')
            # OutlierIndices = np.abs(AverageFluxErrInMags - MedianValuePS) >= NumMadsPS*MADPS 
            #FluxErrInMagsToSysrem[:,OutlierIndices] = ValueToReplaceAlreadyHighNoiseWith
            ########################################3
            
            ### For effectively excluding the highest percent of noisy wavelengths
            
            #UpperPercentileToExclude = 80
            #NumPointsCorrespondingToUpperPercentile = int((UpperPercentileToExclude/100)*len(AverageFluxErrInMags))
            #UpperLimitThreshold = np.sort(AverageFluxErrInMags)[NumPointsCorrespondingToUpperPercentile]
            
            #ColsToSetHighUncertaintyValue = AverageFluxErrInMags>=UpperLimitThreshold
            
            #FluxErrInMagsToSysrem[:,ColsToSetHighUncertaintyValue] = ValueToReplaceAlreadyHighNoiseWith       
            
            ### For excluding the first and last 100
            # FluxErrInMagsToSysrem[:,0:100] = ValueToReplaceAlreadyHighNoiseWith
            # FluxErrInMagsToSysrem[:,-100:] = ValueToReplaceAlreadyHighNoiseWith
            
            ###sysremed1 = sysrem_sub(FluxInMagsToSysrem,FluxErrInMagsToSysrem,SysremIterationsForInitialRunToRefineNoise,a_j=np.array(MeanAirMass))
            
            
            if OrderIndex in OrdersForRefinedUncertaintyFromPreSYSREM:
                ### To get refined uncertainties just by dividing by the median spectrum ("first" component)
                std1V2 = np.std(FluxErrInMagsToSysrem,axis=0)
                std2V2 = np.std(FluxErrInMagsToSysrem,axis=1)
                RefinedUncertaintyForSysrem = np.sqrt(np.outer(std2V2,std1V2))
                
            else:
                ### Get the refined uncertainties from the result of running SYSREM once 
                sysremed1 = MultiSYSREM_AndrewMod(FluxInMagsToSysrem,FluxErrInMagsToSysrem,np.array(MeanAirMass),SysremIterationsForInitialRunToRefineNoise)
        
                ### To get refined uncertainties after running SYSREM once 
                std1 = np.std(sysremed1[:,:,SysremIterationsForInitialRunToRefineNoise-1],axis=0)
                std2 = np.std(sysremed1[:,:,SysremIterationsForInitialRunToRefineNoise-1],axis=1)
                
                RefinedUncertaintyForSysrem = np.sqrt(np.outer(std2,std1)) ## in the order that gives a matrix of the correct shape
                
                
            ## Ernst suggeston: To replace columns with noise greater than a threshold of MADs. Note that the refined errors below are also changed like this 
            ###RefinedUncertaintyForSysrem[:,OutlierIndices] = ValueToReplaceAlreadyHighNoiseWith   
            
            ### For excluding a percentage of the noisiest wavelengths 
            #RefinedUncertaintyForSysrem[:,ColsToSetHighUncertaintyValue] = ValueToReplaceAlreadyHighNoiseWith   
            # RefinedUncertaintyForSysrem[:,0:100] = ValueToReplaceAlreadyHighNoiseWith  
            # RefinedUncertaintyForSysrem[:,-100:] = ValueToReplaceAlreadyHighNoiseWith  
            
            ###sysremed = sysrem_sub(FluxInMagsToSysrem,RefinedUncertaintyForSysrem,components_to_remove,a_j=np.array(MeanAirMass))
            

            

            
            ### To use refined uncertainties 
            sysremed = MultiSYSREM_AndrewMod(FluxInMagsToSysrem,RefinedUncertaintyForSysrem,np.array(MeanAirMass),components_to_remove)
    
            ### To use initial uncertainties 
            #sysremed = MultiSYSREM_AndrewMod(FluxInMagsToSysrem,FluxErrInMagsToSysrem,np.array(MeanAirMass),components_to_remove)
            
            
            ### Original way before Ernst's suggested improvements 
            #sysremed = sysrem_sub(FluxInMagsToSysrem,FluxErrInMagsToSysrem,components_to_remove,a_j=np.array(MeanAirMass))
    
            
            ### Compare to Emily's Sysrem algorithm 
            # sysremed_EmilyAlg = MultiSYSREM_AndrewMod(FluxInMagsToSysrem, FluxErrInMagsToSysrem, np.array(MeanAirMass), components_to_remove)
            # sysremed_EmilyAlg_InFlux = 10**(-sysremed_EmilyAlg/2.5)
            # MakePlotOfStdDevAsFuncOfSysremIt(sysremed_EmilyAlg,'Mag_TelluricFrame_EmilyAlg')
            # MakePlotOfStdDevAsFuncOfSysremIt(sysremed_EmilyAlg_InFlux,'Flux_TelluricFrame_EmilyAlg')
    
            # Write3DArrayToFits('%s/FluxInMag_TelluricFrame_EmilyAlg_sysremedTo%d.fits'%(OutputDirectory,components_to_remove),sysremed_EmilyAlg)
            # Write3DArrayToFits('%s/Flux_TelluricFrame_EmilyAlg_sysremedTo%d.fits'%(OutputDirectory,components_to_remove),sysremed_EmilyAlg_InFlux)
            ### End compare to Emily's Sysrem algorithm 
            ########################
            
            #sysremed = sysrem_sub(MedNormToSysrem,UncArrayToSysrem,components_to_remove,a_j=np.array(MeanAirMass))
            #sysremed = MedNormToSysrem#sysrem_sub(MedNormToSysrem,UncArrayToSysrem,components_to_remove,a_j=np.array(MeanAirMass))
            
            ### Convert result of Sysrem in magnitudes back to fluxes 
            sysremed_InFlux = 10**(-sysremed/2.5)
            
            # DataToWeight = sysremed_InFlux
            # var_or_std='var'    
            # MeanSubtracted = DataToWeight - 1
        
            # if var_or_std == 'var':
            #     residual_col_var_or_std = np.var(MeanSubtracted,axis=0)     
            # if var_or_std == 'std':
            #     residual_col_var_or_std = np.std(MeanSubtracted,axis=0)    
                
            # residual_col_var_or_std[np.where(residual_col_var_or_std==0)] = 1e20     ## To replace variance values of 0 coming from columns of 0 flux 
        
            # ResidualColWeighted = DataToWeight/residual_col_var_or_std      
       
            
            
            
            # Write3DArrayToFits('%s/test.fits'%(OutputDirectory),sysremed_InFlux)     

            #raise Exception
            
            ### Some infinite values are coming through from very small values of residuals from Sysrem when converting to flux
            #sysremed_InFlux[np.where(np.isfinite(sysremed_InFlux)==False)] = np.median(sysremed_InFlux[np.where(np.isfinite(sysremed_InFlux)==True)])
            
            # DataToWeight = sysremed_InFlux
            # var_or_std = 'std'
            
            # DataToWeight -= 1.
        
            # if var_or_std == 'var':
            #     residual_col_var_or_std = np.var(DataToWeight,axis=0)     
            # if var_or_std == 'std':
            #     residual_col_var_or_std = np.std(DataToWeight,axis=0)    
                
            # residual_col_var_or_std[np.where(residual_col_var_or_std==0)] = 1e20     ## To replace variance values of 0 coming from columns of 0 flux 
        
            # ResidualColWeighted = DataToWeight/residual_col_var_or_std       
       
            # raise Exception 
    
            
            # ShiftedFluxPS = np.copy(MedNormToSysrem)
            # ShiftedUncertaintyPS = np.copy(UncArrayToSysrem)
            # ShiftedWavePS = np.copy(WaveArrayOfFluxGivenToSysrem)
            
            # Shifted_sysremed_InFlux = np.copy(sysremed_InFlux) 
            
            
            #### Shift from telluric Earth frame into the barycentric frame       
            #### Note that the output of Sysrem is 3D with the depth axis being the Sysrem components removed 
            # for i in range(len(SortedFileList)):
            
            #     ShiftedWavePostSysrem = WaveArrayOfFluxGivenToSysrem[i,:]*(1+(BarycentricRVcorrection[i]/c_kms))   
                
            #     if i == 0:
            #         FirstSpecShiftedWavePS = ShiftedWavePostSysrem      
                    
            #     ShiftedWavePS[i,:] = FirstSpecShiftedWavePS
                
            #     ShiftedFluxFunctionPS = interp1d(ShiftedWavePostSysrem,MedNormToSysremOnlyPositive[i,:],kind='linear',bounds_error=False,fill_value='extrapolate')        
            #     ShiftedFluxPS[i,:] = ShiftedFluxFunctionPS(FirstSpecShiftedWavePS)  
                    
            #     ShiftedUncertaintyFunctionPS = interp1d(ShiftedWavePostSysrem,UncArrayToSysremOnlyPositive[i,:],kind='linear',bounds_error=False,fill_value='extrapolate')             
            #     ShiftedUncertaintyPS[i,:] = ShiftedUncertaintyFunctionPS(FirstSpecShiftedWavePS) 
            # ShiftedMagErr = np.abs(-2.5*np.log(10)*(ShiftedUncertaintyPS/ShiftedFluxPS))
                
            # for SysremItIndex in range(components_to_remove):
                
            #     for i in range(len(SortedFileList)):
                    
            #         ShiftedWavePostSysrem = WaveArrayOfFluxGivenToSysrem[i,:]*(1+(BarycentricRVcorrection[i]/c_kms))  
                     
            #         ShiftedSysremedFluxFunctionPS = interp1d(ShiftedWavePostSysrem,sysremed_InFlux[i,:,SysremItIndex],kind='linear',bounds_error=False,fill_value='extrapolate')        
                    
            #         Shifted_sysremed_InFlux_SingleSpec = ShiftedSysremedFluxFunctionPS(FirstSpecShiftedWavePS)  
                    
            #         ### Even though sysremed_InFlux doesn't have infinite values, some are still appearing after interpolating onto the wavelength grid, I guess just because the numbers are so large that they are effectively infinite 
            #         if (False in np.isfinite(Shifted_sysremed_InFlux_SingleSpec)):
            #             InfIndices = np.where(False == np.isfinite(Shifted_sysremed_InFlux_SingleSpec))
            #             NotInfIndices = np.where(np.isfinite(Shifted_sysremed_InFlux_SingleSpec))
            #             Shifted_sysremed_InFlux_SingleSpec[InfIndices] = np.median(Shifted_sysremed_InFlux_SingleSpec[NotInfIndices])                 
                                   
            #         if True in np.isnan(Shifted_sysremed_InFlux_SingleSpec):
            #             print('ShiftedSysremedFluxFunctionPS has nan on sysrem %d, spec %d'%(SysremItIndex,i))
            #             raise Exception
                    
            #         Shifted_sysremed_InFlux[i,:,SysremItIndex] = Shifted_sysremed_InFlux_SingleSpec
            
            # MakePlotOfStdDevAsFuncOfSysremIt(sysremed,FluxErrInMagsToSysrem,'Mag_TelluricFrame')
            # MakePlotOfStdDevAsFuncOfSysremIt(sysremed_InFlux,FluxErrInMagsToSysrem,'Flux_TelluricFrame')
            # MakePlotOfStdDevAsFuncOfSysremIt(Shifted_sysremed_InFlux,ShiftedMagErr,'Flux_BaryFrame')
                
            # ShiftedFluxPSResidual =  ShiftedFluxPS/np.median(ShiftedFluxPS,axis=0)           
    

            sysremedColWeightedByVar = WeightColsByVarOrStdIn3DArray(sysremed_InFlux,var_or_std='var')

            sysremedColWeightedByStd = WeightColsByVarOrStdIn3DArray(sysremed_InFlux,var_or_std='std')

    
            
            # if True in np.isnan(sysremedColWeightedByVar):
            #     print()
            #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #     print('night %s arm %s %s order %d'%(night,arm,arm_subpart,OrderIndex))
    
                
            #     # print('sysremedColWeighted has nans')
                
            #     # print('Nans also in Shifted_sysremed_InFlux (T/F):')
            #     # print(True in np.isnan(Shifted_sysremed_InFlux))           
                
                
            #     raise Exception
            
            # MakePlotOfStdDevAsFuncOfSysremIt(sysremedColWeighted,ShiftedMagErr,'Flux_BaryFrame_ColWeighted')
            
            ##pyfits.writeto('BlazeCorrectedArray_dump.fits',BlazeCorrectedArray,overwrite=True)

            if FullOutput == True:            
                pyfits.writeto('%s/InjectionArray.fits'%(OutputDirectory),InjectionArray,overwrite=True)
                
                if DataOrigin == 'CARMENES':   
                    
                    # print('!!!!!!!!!!!!!')
                    # print(OutputDirectory)
                    # print()
                    # raise Exception
                    
                    pyfits.writeto('%s/Flux1_InitialFractionalUncertainty.fits'%(OutputDirectory),ShiftedFractionalUncertainty,overwrite=True)
                    pyfits.writeto('%s/BlazeCorrectedUncArray.fits'%(OutputDirectory),BlazeCorrectedUncArray,overwrite=True)
        
                    pyfits.writeto('%s/CroppedBlazeCorrectedUncArray.fits'%(OutputDirectory),UncArrayToSysrem,overwrite=True)
        
                    
                    #pyfits.writeto('%s/FluxUncertainty.fits'%(OutputDirectory),ShiftedUcertaintyArray,overwrite=True)
                    pyfits.writeto('%s/SkyWaveArray.fits'%(OutputDirectory),SkyWaveArray,overwrite=True)  
                    pyfits.writeto('%s/SkyFluxArray.fits'%(OutputDirectory),SkyFluxArray,overwrite=True)  
                    pyfits.writeto('%s/SkyFluxUncArray.fits'%(OutputDirectory),SkyFluxUncArray,overwrite=True)  
                
                # if DataOrigin != 'CARMENES':    
                #     pyfits.writeto('%s/FlattenedData.fits'%(OutputDirectory),FluxArray,overwrite=True)  
                
                #pyfits.writeto('%s/Flux0_InitialNotBaryCorrected.fits'%(OutputDirectory),InitialFluxArray,overwrite=True)  
                pyfits.writeto('%s/Wave1_Initial.fits'%(OutputDirectory),InitialWaveArray,overwrite=True)              
                pyfits.writeto('%s/Flux1_Initial_Order%d.fits'%(OutputDirectory,OrderIndex),InitialFluxArray,overwrite=True)  
                pyfits.writeto('%s/Flux1_Err_Initial.fits'%(OutputDirectory),InitialUncertaintyArray,overwrite=True)
                
                #pyfits.writeto('%s/Flux2_NoCosmics.fits'%(OutputDirectory),CosmicsRemovedArray,overwrite=True)  
                pyfits.writeto('%s/Flux3a_BlazeCorrWithEmissionAndCosmics_Order%d.fits'%(OutputDirectory,OrderIndex),BlazeCorrectedArray_WithCosmicsAndEmissionToSave, overwrite=True)  
                pyfits.writeto('%s/Flux3b_BlazeCorrNoEmissionAndCosmics_Order%d.fits'%(OutputDirectory,OrderIndex),BlazeCorrectedArray_NoEmission_WithCosmics_ToSave, overwrite=True)  
                pyfits.writeto('%s/Flux3c_BlazeCorrNoEmissionNoCosmics_Order%d.fits'%(OutputDirectory,OrderIndex),CosmicsRemovedArray, overwrite=True)  
                
                pyfits.writeto('%s/Flux3_BlazeCorrected_Order%d.fits'%(OutputDirectory,OrderIndex),BlazeCorrectedArray, overwrite=True)  
                
                
                ##pyfits.writeto('%s/Flux3_BlazeCorrected_AfterCosandEm_Order%d.fits'%(OutputDirectory,OrderIndex),BlazeCorrectedArrayAfterCosmicAndEmission_ToSave, overwrite=True)  

                
                
                
                #pyfits.writeto('%s/Flux3a_SimpleResidual.fits'%(OutputDirectory),Residual,overwrite=True)  
                
                pyfits.writeto('%s/Flux4_ToSysrem.fits'%(OutputDirectory),MedNormToSysremOnlyPositive,overwrite=True)  
                pyfits.writeto('%s/Flux4Err_ToSysrem.fits'%(OutputDirectory),UncArrayToSysremOnlyPositive,overwrite=True)  
             
                
                
                
                pyfits.writeto('%s/Wave4_ToSysrem.fits'%(OutputDirectory),WaveArrayOfFluxGivenToSysrem,overwrite=True)  
                #pyfits.writeto('%s/ResidualFlux_ToSysrem.fits'%(OutputDirectory),ResidualMedNormToSysrem,overwrite=True)  
                
                
                
                            
                #pyfits.writeto('%s/FluxInMagsNotResidual.fits'%(OutputDirectory),FluxInMagsNotResidual,overwrite=True)  
                
                pyfits.writeto('%s/Flux5_InMags_ToSysrem.fits'%(OutputDirectory),FluxInMagsToSysrem,overwrite=True)  
                pyfits.writeto('%s/Flux5_ErrInMags_ToSysrem.fits'%(OutputDirectory),FluxErrInMagsToSysrem,overwrite=True)  
                pyfits.writeto('%s/Flux5_RefinedFluxErrInMags_ToSysrem.fits'%(OutputDirectory),RefinedUncertaintyForSysrem,overwrite=True)              
    
                
                #Write3DArrayToFits('%s/Flux4a_cropped_pcasubbedTo%d.fits'%(OutputDirectory,components_to_remove),pcasubbed)
                Write3DArrayToFits('%s/Flux6_InMag_TelluricFrame_sysremedTo%d.fits'%(OutputDirectory,components_to_remove),sysremed)
                Write3DArrayToFits('%s/Flux7_TelluricFrame_sysremedTo%d.fits'%(OutputDirectory,components_to_remove),sysremed_InFlux)
                
                
                Write3DArrayToFits('%s/Flux8a_TelluricFrame_sysremedTo%d_ColWeighted_var.fits'%(OutputDirectory,components_to_remove),sysremedColWeightedByVar)
                Write3DArrayToFits('%s/Flux8b_TelluricFrame_sysremedTo%d_ColWeighted_std.fits'%(OutputDirectory,components_to_remove),sysremedColWeightedByStd)
                
                # Write3DArrayToFits('%s/Flux4_BarycentricFrame_sysremedTo%d.fits'%(OutputDirectory,components_to_remove),Shifted_sysremed_InFlux)
                
               
                # pyfits.writeto('%s/Wave_PostSysrem_ShiftedForBaryCorrection.fits'%(OutputDirectory),ShiftedWavePS,overwrite=True)  
                # pyfits.writeto('%s/Flux_PostSysrem_ShiftedForBaryCorrection.fits'%(OutputDirectory),ShiftedFluxPS,overwrite=True)  
                # pyfits.writeto('%s/FluxErr_PostSysrem_ShiftedForBaryCorrection.fits'%(OutputDirectory),ShiftedUncertaintyPS,overwrite=True)  
                # pyfits.writeto('%s/ResidualFlux_PostSysrem_ShiftedForBaryCorrection.fits'%(OutputDirectory),ShiftedFluxPSResidual,overwrite=True)  
                
                #Write3DArrayToFits('%s/Flux5_BaryFrame_sysremedTo%d_ColWeighted.fits'%(OutputDirectory,components_to_remove),sysremedColWeighted)
                

                # plt.figure()
                
                # nr, nc = BlazeCorrectedArray.shape
                
                # for i in range(nr):
                #     plt.plot(BlazeCorrectedArray[i,:])
                # plt.title('order %d'%(OrderIndex))
                    
                # plt.savefig('test%d.png'%(OrderIndex))
                
                
                numspec,numcols = np.shape(FluxWithInjectedArray)
                with PdfPages('%s/DiagnosticPlots/CosmicRemovalDiagnostic.pdf'%(OutputDirectory)) as CosmicDiagnosticOutputPDF:
                    
                    for i in range(numspec):
                        
                        plt.figure()
                        plt.plot(BlazeCorrectedArray_WithCosmics[i,:],label='With cosmics')
                        plt.plot(BlazeCorrectedArray[i,:],label='No cosmics')
                        
                        plt.legend()
                        
                        DiffBetweenWithAndWithoutCosmics = BlazeCorrectedArray_WithCosmics - BlazeCorrectedArray
                        
                        
                        
                        TotalPercentageOfPointsContainingCosmics = 100*np.count_nonzero(DiffBetweenWithAndWithoutCosmics)/np.size(DiffBetweenWithAndWithoutCosmics)
                      
                        plt.title('Spec %d. Cosmics removed %d (%.2e pc)'%\
                                  (i,len(np.where(np.nan_to_num(BlazeCorrectedArray_WithCosmics[i,:])!=np.nan_to_num(BlazeCorrectedArray[i,:]))[0]),\
                                   100*len(np.where(np.nan_to_num(BlazeCorrectedArray_WithCosmics[i,:])!=np.nan_to_num(BlazeCorrectedArray[i,:]))[0])/len(BlazeCorrectedArray[i,:])))

                        
                        CosmicDiagnosticOutputPDF.savefig()
                        
                        np.savetxt('%s/Order%d_Percentage_of_points_replaced_as_cosmics.txt'%(OutputDirectory,OrderIndex),np.array([TotalPercentageOfPointsContainingCosmics]))

                        
                        plt.close()    
         
                 
                ### might need these ? 
                #pyfits.writeto('%s/MedNorm.fits'%(OutputDirectory),MedNorm,overwrite=True)  
         
                #pyfits.writeto('%s/MedNormToSysrem.fits'%(OutputDirectory),MedNormToSysrem,overwrite=True)    
                #pyfits.writeto('%s/UncArrayToSysrem.fits'%(OutputDirectory),UncArrayToSysrem,overwrite=True)    
        
                
            
        # np.save('%s/AllOrdersResiduals.npy'%(AllOrderCubeOutput),ResidualsAllOrders)
        # np.save('%s/AllOrdersWave.npy'%(AllOrderCubeOutput),WaveAllOrders)
        # np.save('%s/AllOrdersInjectionArray.npy'%(AllOrderCubeOutput),InjectionArrayAllOrders)
        np.savetxt('%s/exptime.txt'%(mjdOutputDir),np.array(exptime))
        np.savetxt('%s/airmass.txt'%(mjdOutputDir),np.array(MeanAirMass))
        np.savetxt('%s/mjd.txt'%(mjdOutputDir),np.array(SortedMJDList))
        np.savetxt('%s/phase.txt'%(mjdOutputDir),phase)
        np.savetxt('%s/radv.txt'%(mjdOutputDir),radv)
        np.savetxt('%s/BarycentricRVcorrection_kms.txt'%(mjdOutputDir),np.array(BarycentricRVcorrection))
        np.savetxt('%s/transitlims.txt'%(mjdOutputDir),transitlims[0])
        np.savetxt('%s/NettPlanetRV.txt'%(mjdOutputDir),NettModelRV)
        np.savetxt('%s/dec.txt'%(mjdOutputDir),np.array(PointingDECList))
        np.savetxt('%s/RA.txt'%(mjdOutputDir),np.array(PointingRAList))
        

        
    
        
    EndTime = time.time()
    
    TimeTaken = EndTime-StartTime
    
    print('Time taken to do %d spectra was %f minutes'%(NumObs,TimeTaken/60))


# plt.figure()
# for i in range(nr):
#     plt.plot(ShiftedFractionalUncertainty[i,:])
# plt.ylim((0,0.2))

# i = 7
# plt.plot(ShiftedFractionalUncertainty[i,:],label=i,color='black')


# plt.figure()
# i = 0
# plt.plot(BlazeCorrectedUncArray[i,:],label=i)


# i = 17
# plt.plot(BlazeCorrectedUncArray[i,:],label=i)

# i = 27
# plt.plot(BlazeCorrectedUncArray[i,:],label=i)

# plt.ylim((0,0.1))

# plt.legend()


# plt.figure()
# i = 6
# plt.plot(FluxWithInjectedArray[i,:],label=i)


# i = 8
# plt.plot(FluxWithInjectedArray[i,:],label=i)

# i = 7
# plt.plot(FluxWithInjectedArray[i,:],label=i)

# #plt.ylim((0,0.1))

# plt.legend()

# plt.figure()
# i = 6
# plt.plot(BlazeCorrectedArray[i,:],label=i)


# i = 8
# plt.plot(BlazeCorrectedArray[i,:],label=i)

# i = 7
# plt.plot(BlazeCorrectedArray[i,:],label=i)

# #plt.ylim((0,0.1))

# plt.legend()

# ### 
# plt.figure()
# i = 6
# plt.plot(ShiftedUcertaintyArray[i,:],label=i)


# i = 8
# plt.plot(ShiftedUcertaintyArray[i,:],label=i)

# i = 7
# plt.plot(ShiftedUcertaintyArray[i,:],label=i)

# plt.ylim((0.015,0.02))

# plt.legend()

# ###########
# plt.figure()
# i = 6
# plt.plot(ShiftedFractionalUncertainty[i,:],label=i)


# i = 8
# plt.plot(ShiftedFractionalUncertainty[i,:],label=i)

# i = 7
# plt.plot(ShiftedFractionalUncertainty[i,:],label=i)

# plt.ylim((0.015,0.02))

# plt.legend()

# ################


# mbstd = np.nanstd(BlazeCorrectedArray,axis = 1)

# plt.plot(mbstd,'.-')
# plt.ylabel('std dev of blaze corrected spec')
# plt.xlabel('spectrum number')

# mbfunc = np.nanmean(ShiftedFractionalUncertainty,axis = 1)

# plt.plot(mbfunc,'.-')
# plt.ylabel('Mean fractional uncertainty')
# plt.xlabel('spectrum number')

# mInitialUnc = np.nanmedian(ShiftedUcertaintyArray,axis = 1)

# plt.plot(mInitialUnc,'.-')
# plt.ylabel('Mean initial uncertainty')
# plt.xlabel('spectrum number')

# t = InitialUncertaintyArray/InitialFluxArray
# tt = np.nanmean(t,axis=1)
# plt.plot(tt,'.-')
# plt.ylabel('Mean fractional uncertainty from initial flux and unc')
# plt.xlabel('spectrum number')



