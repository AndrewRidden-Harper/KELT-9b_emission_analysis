# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:44:12 2020

@author: ar-h1
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
import os
import astropy.io.fits as pyfits 
from astropy.stats import sigma_clip
from photutils import centroid_com, centroid_1dg, centroid_2dg
from scipy import interpolate
import pandas as pd 
from astropy.modeling import models, fitting
from astropy.stats import median_absolute_deviation
from scipy.optimize import curve_fit


import matplotlib as mpl
mpl.use('Agg')


def LoadKpPerOrderAndWeights(TargetForCrossCor,night,arm,arm_subpart,ModelScalingFactor,SysremIterations,ModelForCrossCor,
                             ScalingFactorUsedForFindingWeights,SysremIterationsUsedForFindingWeights,OverideWeights=None):
    
    
    # if KpMode == 'SingleKp':
    #     KpVsysOrSingle_KpVsys = 'Single_KpVsys'
    # if KpMode == 'MultiKp':
    #     KpVsysOrSingle_KpVsys = 'KpVsys'
    
    
    ## I think XCorr will need to be replaced with KpVsys here but not for the RV vect 
    # KpLoadDirOrder0 = '%s/KELT-9b_CARMENES_emission_data/XCorr/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d/KpVsys'%(FirstPartOfLoadPath,
    #                                                                                                                     night,TargetForCrossCor,
    #                                                                                                                     ModelForCrossCor,
    #                                                                                                                     ModelScalingFactor,
    #                                                                                                                     arm,arm_subpart,
    #                                                                                                                     0,
    #                                                                                                                     SysremIterations[0])
    
    # KpLoadDirOrder0 = '%s/KELT-9b_CARMENES_emission_data/XCorrUsingRVPM1000/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d/KpVsys'%(FirstPartOfLoadPath,
    #                                                                                                                     night,TargetForCrossCor,
    #                                                                                                                     ModelForCrossCor,
    #                                                                                                                     ModelScalingFactor,
    #                                                                                                                     arm,arm_subpart,
    #                                                                                                                     0,
    #                                                                                                                     SysremIterations[0])
    
    
    
    #### New Kp location 

    KpLoadDirOrder0 = '%s/KELT-9b_CARMENES_emission_data/%s/SpecWeightConfig%d/CCFMedSub%d/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d/'%(FirstPartOfLoadPath,
                                                                                                                      KpMode,
                                                                                                                      SpecWeightConfig,
                                                                                                                      CCFMedSub,
                                                                                                                        night,TargetForCrossCor,
                                                                                                                        ModelForCrossCor,
                                                                                                                        ModelScalingFactor,
                                                                                                                        arm,arm_subpart,
                                                                                                                        0,
                                                                                                                        SysremIterations[0])
    
    
##############################################################
    
    
    KpVsysMatrixOrder0 = np.load('%s/KpXCorrOrder%d.npy'%(KpLoadDirOrder0,0))
    NumKps,NumRVs = KpVsysMatrixOrder0.shape    
    NumOrders = len(SysremIterations)    

    KpVsysMatrix = np.zeros((NumKps,NumRVs,NumOrders))
    OrderWeights = np.zeros((NumOrders))
    
    
    
    
    KpVect = np.loadtxt('%s/KpVect.txt'%(KpLoadDirOrder0))
    
    ################################################################################
    
    

    
    
    RVLoadDirOrder0 = '%s/KELT-9b_CARMENES_emission_data/XCorr/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d'%(FirstPartOfLoadPath,
                                                                                                                        night,TargetForCrossCor,
                                                                                                                        ModelForCrossCor,
                                                                                                                        ModelScalingFactor,
                                                                                                                        arm,arm_subpart,
                                                                                                                        0,
                                                                                                                        SysremIterations[0])   
    
    
    
    #################################################################################
    
    RVvect = np.loadtxt('%s/sysremed_%d_XCorrRadVelVect.txt'%(RVLoadDirOrder0,SysremIterations[0]))


    
    

    
    for OrderIndex in range(len(SysremIterations)):
        
    
        # KpLoadDirOrder = '%s/KELT-9b_CARMENES_emission_data/XCorr/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d/KpVsys'%(FirstPartOfLoadPath,
        #                                                                                                                     night,TargetForCrossCor,
        #                                                                                                                     ModelForCrossCor,
        #                                                                                                                     ModelScalingFactor,
        #                                                                                                                     arm,arm_subpart,
        #                                                                                                                     OrderIndex,
        #                                                                                                                     SysremIterations[OrderIndex])
        
        KpLoadDirOrder = '%s/KELT-9b_CARMENES_emission_data/%s/SpecWeightConfig%d/CCFMedSub%d/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d'%(FirstPartOfLoadPath,
                                                                                                                        KpMode,
                                                                                                                        SpecWeightConfig,
                                                                                                                        CCFMedSub,
                                                                                                                            night,TargetForCrossCor,
                                                                                                                            ModelForCrossCor,
                                                                                                                            ModelScalingFactor,
                                                                                                                            arm,arm_subpart,
                                                                                                                            OrderIndex,
                                                                                                                            SysremIterations[OrderIndex])       
        
        if AllorOddorEven == 'All':
            KpVsysMatrix[:,:,OrderIndex] = np.load('%s/KpXCorrOrder%d.npy'%(KpLoadDirOrder,OrderIndex))
        if AllorOddorEven == 'Odd':
            KpVsysMatrix[:,:,OrderIndex] = np.load('%s/KpXCorrOrder%d_OnlyOddSpectra.npy'%(KpLoadDirOrder,OrderIndex))        
        if AllorOddorEven == 'Even':
            KpVsysMatrix[:,:,OrderIndex] = np.load('%s/KpXCorrOrder%d_OnlyEvenSpectra.npy'%(KpLoadDirOrder,OrderIndex))
        
        
##########################################################################


        # OrderWeightsLoadDir = '%s/KELT-9b_CARMENES_emission_data/XCorr/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d/KpVsys'%(FirstPartOfLoadPath,
        #                                                                                                                         night,ModelForCrossCor,
        #                                                                                                                         ModelForCrossCor,
        #                                                                                                                         ScalingFactorUsedForFindingWeights,
        #                                                                                                                         arm,arm_subpart,
        #                                                                                                                         OrderIndex,
        #                                                                                                                         SysremIterationsUsedForFindingWeights[OrderIndex])
        
        # OrderWeightsLoadDir = '%s/KELT-9b_CARMENES_emission_data/XCorrUsingRVPM1000/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d/KpVsys'%(FirstPartOfLoadPath,
        #                                                                                                                         night,ModelForCrossCor,
        #                                                                                                                         ModelForCrossCor,
        #                                                                                                                         ScalingFactorUsedForFindingWeights,
        #                                                                                                                         arm,arm_subpart,
        #                                                                                                                         OrderIndex,
        #                                                                                                                         SysremIterationsUsedForFindingWeights[OrderIndex])

        
        
        if LoadOrderWeights:
            #### New Kp location 
            OrderWeightsLoadDir = '%s/KELT-9b_CARMENES_emission_data/%s/SpecWeightConfig%d/CCFMedSub%d/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d/'%(FirstPartOfLoadPath,
                                                                                                                                    'SingleKp',
                                                                                                                                    SpecWeightConfig,
                                                                                                                                    CCFMedSub,
                                                                                                                                    night,ModelForCrossCor,
                                                                                                                                    ModelForCrossCor,
                                                                                                                                    ScalingFactorUsedForFindingWeights,
                                                                                                                                    arm,arm_subpart,
                                                                                                                                    OrderIndex,
                                                                                                                                    SysremIterationsUsedForFindingWeights[OrderIndex])
            
            
            
            
    ###########################################################################################################################
    
        
            OrderWeights[OrderIndex] = np.loadtxt('%s/OrderWeightsFromInjectedSignal.txt'%(OrderWeightsLoadDir))
            
            # if OrderWeights[OrderIndex] < 3:
            #     OrderWeights[OrderIndex] = 0        
            
            # if arm == 'nir':
            #     OrderWeights[OrderIndex] = 0
            
            # if (arm == 'nir') and (arm_subpart == 'A'):
            #     OrderWeights[OrderIndex] = 0
                
            # if arm == 'vis':
            #     OrderWeights[43:] = 0
                
            #OrderWeights[OrderWeights<3] = 0.0
            
        if not LoadOrderWeights:
            OrderWeights = np.ones((NumOrders))
            
        
    
    if OverideWeights is not None:
        NumWeights = len(OrderWeights)
        for i in range(NumWeights):
            if np.isfinite(OverideWeights[i]):
                OrderWeights[i] = OverideWeights[i]
        
    return [KpVsysMatrix,KpVect,RVvect,OrderWeights]


#######################################################################
    
def LoadOrdersForConfidenceInterval(NumRandom,RunNum,TargetForCrossCor,night,arm,arm_subpart,ModelScalingFactor,SysremIterations,ModelForCrossCor,
                             ScalingFactorUsedForFindingWeights,SysremIterationsUsedForFindingWeights,OverideWeights=None):
    

    
    KpLoadDirOrder0 = '%s/KELT-9b_CARMENES_emission_data/ConfidenceIntervals/NumRandom%d/RunNum%d/SpecWeightConfig%d/CCFMedSub%d/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d'%(FirstPartOfLoadPath,  
                                                                                                                                        NumRandom,
                                                                                                                                        RunNum,
                                                                                                                                        SpecWeightConfig,
                                                                                                                                        CCFMedSub,
                                                                                                                                        night,
                                                                                                                                        TargetForCrossCor,
                                                                                                                                        ModelForCrossCor,
                                                                                                                                        ModelScalingFactor,
                                                                                                                                        arm,
                                                                                                                                        arm_subpart,
                                                                                                                                        0,
                                                                                                                                        SysremIterations[0])       

    
    
##############################################################
    
    
    KpVsysMatrixOrder0 = np.load('%s/KpXCorrOrder%d.npy'%(KpLoadDirOrder0,0))
    NumKps,NumRVs = KpVsysMatrixOrder0.shape    
    NumOrders = len(SysremIterations)    

    KpVsysMatrix = np.zeros((NumKps,NumRVs,NumOrders))
   
    for OrderIndex in range(len(SysremIterations)):

        
        KpLoadDirOrder = '%s/KELT-9b_CARMENES_emission_data/ConfidenceIntervals/NumRandom%d/RunNum%d/SpecWeightConfig%d/CCFMedSub%d/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d'%(FirstPartOfLoadPath,  
                                                                                                                                    NumRandom,
                                                                                                                                    RunNum,
                                                                                                                                    SpecWeightConfig,
                                                                                                                                    CCFMedSub,
                                                                                                                                    night,
                                                                                                                                    TargetForCrossCor,
                                                                                                                                    ModelForCrossCor,
                                                                                                                                    ModelScalingFactor,
                                                                                                                                    arm,
                                                                                                                                    arm_subpart,
                                                                                                                                    OrderIndex,
                                                                                                                                    SysremIterations[OrderIndex])       

        
        ##################################
        
        KpVsysMatrix[:,:,OrderIndex] = np.load('%s/KpXCorrOrder%d.npy'%(KpLoadDirOrder,OrderIndex))
        
        
    return KpVsysMatrix


########################################################################



def CalculateWeightedAverage(TupleOfKpVysMatrices,TupleOfOrderWeights):
    
    AllKpVsysArray = np.dstack(TupleOfKpVysMatrices)
    AllOrderWeightsArray = np.hstack(TupleOfOrderWeights)    
    
    print('Mean weight %.3e'%(np.mean(AllOrderWeightsArray)))    
    
    numrows,numcols,numorders = np.shape(AllKpVsysArray)        
    WeigtnedAverage = np.zeros((numrows,numcols))
    
    for i in range(numorders):
        WeigtnedAverage += AllKpVsysArray[:,:,i]*AllOrderWeightsArray[i]
    WeigtnedAverage /= np.sum(AllOrderWeightsArray)
    
    #WeigtnedAverage[WeigtnedAverage>1e6] = 0
    
    return WeigtnedAverage

def binspec(spec,pointsinbin=5):
    
    '''
    A way to bin every N points in a spectrum based on 
    doing a clever reshape then average along a row
    
    It returns a spectrum with representative bin sizes in a simple way.    
    
    '''   
        
    numcols = len(spec)
    
    remainderPoints = numcols%pointsinbin 
    
    extraRequiredPoints = pointsinbin-remainderPoints
    
    NANfiller_array = np.empty((numcols+extraRequiredPoints))
    NANfiller_array[:] = np.NAN
    
    NANfiller_array[:numcols] = spec
    
    binnedspec = np.nanmean(NANfiller_array.reshape(-1, pointsinbin), axis=1)
    
    full_length_binned_spec = np.empty_like(spec)
    
    for i in range(len(binnedspec)):
        
        full_length_binned_spec[i*pointsinbin:(i+1)*pointsinbin] = binnedspec[i]
        
    return full_length_binned_spec


def BinnnerCentrePoint(VectToBin,PointsInBin,CentrePoint):
    
    '''
    centre point should be an index (starting at 0)
    
    This is a function that depends on binspec which adds the extra 
    functionality to allow the point that the bins should be centred around.
    It places a central bin around this central point, then uses binspec 
    on the left and right parts of the vector around the centred point    
    '''
    
    binned_vect = np.empty_like(VectToBin)
    binned_vect[:] = np.NAN
    
    halfbin = int(np.floor(PointsInBin/2.0))
    
    if PointsInBin % 2.0 != 0: #is an odd number 
    
        print('odd number of points in bin')
    
        binned_vect[CentrePoint-halfbin:CentrePoint+halfbin+1] =  np.mean(VectToBin[CentrePoint-halfbin:CentrePoint+halfbin+1])    
        
        leftpart = VectToBin[:CentrePoint-halfbin]
        rightpart = VectToBin[CentrePoint+halfbin+1:]
        
        flippedleftpart = leftpart[::-1]    
        flippedbinned_leftpart = binspec(flippedleftpart,PointsInBin)    
        binned_leftpart = flippedbinned_leftpart[::-1]
        
        binned_vect[:CentrePoint-halfbin] = binned_leftpart    
           
        ## put the binned right part in the binned vector    
        binned_vect[CentrePoint+halfbin+1:] = binspec(rightpart,PointsInBin)
    
    if PointsInBin % 2.0 == 0: #is an even number 
    
        print('even number of points in bin')    
    
        binned_vect[CentrePoint-halfbin:CentrePoint+halfbin] =  np.mean(VectToBin[CentrePoint-halfbin:CentrePoint+halfbin])
            
        leftpart = VectToBin[:CentrePoint-halfbin]
        rightpart = VectToBin[CentrePoint+halfbin:]
    
        flippedleftpart = leftpart[::-1]
        
        flippedbinned_leftpart = binspec(flippedleftpart,PointsInBin)
        
        binned_leftpart = flippedbinned_leftpart[::-1]
        
        binned_vect[:CentrePoint-halfbin] = binned_leftpart    
        
        binned_vect[CentrePoint+halfbin:] = binspec(rightpart,PointsInBin)
        
    return binned_vect

def WeightedNanMeanOverRows(ArrayToAverage,weights):
    
    ma = np.ma.MaskedArray(ArrayToAverage, mask=np.isnan(ArrayToAverage))
    WeightedAvg = np.ma.average(ma, weights=weights,axis=0)
    
    return WeightedAvg

def MeasureDetectionSNR(ccf,SignalRegionIndex=(975,985),Adjust=False):
    
    maxindex = SignalRegionIndex[0] + np.argmax(ccf[SignalRegionIndex[0]:SignalRegionIndex[1]])
    SignalStrength = ccf[maxindex]
    
    ExcludeSignalMask = np.zeros_like(ccf)
    ExcludeSignalMask[SignalRegionIndex[0]:SignalRegionIndex[1]] = 1
    
    ### Exclude leading and trailing points  
    NumberOfLeadingAndTrailingPointsToExclude = 500 ## Good for -1000 to 1000 in steps of 1 

    ExcludeSignalMask[0:NumberOfLeadingAndTrailingPointsToExclude] = 1
    ExcludeSignalMask[-NumberOfLeadingAndTrailingPointsToExclude:] = 1
    
    ma = np.ma.MaskedArray(ccf, mask=ExcludeSignalMask)
    
    noise = ma.std()    
    
    if Adjust:
        SNR = (SignalStrength-noise)/noise
    
    if not Adjust:
        SNR = SignalStrength/noise
    
    return SNR, SignalStrength, noise

def MakePlots(KpVsys,ModelForXCor,name,SaveLoadable=False):
    
    name = name + '_%s'%(InjStrForWeights)
    
    #SigConfInt = np.loadtxt('CalculatedConfidenceIntervals/ProcessedDataNoRotBroadening/NumShuffles10000/TargetKELT9b_Al_0.25_+2.0_0.55_InjScalingFactor1.000e+00_ModelKELT9b_Al_0.25_+2.0_0.55_Sysrem9_StdDevConfidenceInterval_Shuffles10000.txt')

    
    # KpVsys = sigma_clip(KpVsys, sigma=3, masked='False')
    
    DetSig = MeasureDetectionSNR(KpVsys[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)
    
    #pyfits.writeto('KpVsysDump.fits',KpVsys,overwrite=True)


    
    with PdfPages('%s/%s.pdf'%(SavePath,name)) as pdf:          
         
        if KpMode == 'MultiKp':
            plt.figure()
            plt.imshow(KpVsys[:,ImShowXLimits[0]:ImShowXLimits[1]],interpolation='None',origin='lower',aspect='auto',extent=(KpVsysXaxis[0],KpVsysXaxis[-1],KpVect[0],KpVect[-1]))
            plt.xlabel('Systemic velocity (km s$^{-1}$)')
            plt.ylabel(r'K$_p$ (km s$^{-1}$)')      
            plt.plot([KpVsysXaxis[0],KpVsysXaxis[-1]],[RealKp,RealKp],'w:')
            plt.plot([BestSys,BestSys],[KpVect[0],KpVect[-1]],'w:')
            plt.colorbar()
            #plt.xlim(RadVPlotLimits[0],RadVPlotLimits[1])
            #plt.title('%s'%(name))
            plt.tight_layout()
            pdf.savefig()  
            plt.close()
        
        
        plt.figure()
        plt.plot(KpVsysXaxis,KpVsys[ClosestKpRow,ImShowXLimits[0]:ImShowXLimits[1]])
        plt.xlabel('Systemic velocity (km s$^{-1}$)')
        plt.ylabel('CCF')
        plt.xlim(BestKpXlims)
        #plt.plot([BestSys,BestSys],[-0.001,0.0016],'k:')
        #plt.title('%s Det. Sig. %f'%(name,DetSig[0]))
        
        # plt.plot(KpVsysXaxis,StdConfIntToPlot,'k--')
        # plt.plot(KpVsysXaxis,SigConfInt[ImShowXLimits[0]:ImShowXLimits[1]]*3,'k--')
        # plt.plot(SigConfInt[ImShowXLimits[0]:ImShowXLimits[1]]*5,'k--')
        
        if PlotConfidenceIntervalString == 'WithConfInt':
            plt.fill_between(KpVsysXaxis, -StdConfIntToPlot, StdConfIntToPlot, alpha=0.2)
            plt.fill_between(KpVsysXaxis, -StdConfIntToPlot*3, StdConfIntToPlot*3, alpha=0.2)
            
        QuickStd = np.std(KpVsys[ClosestKpRow,:])
        
        plt.plot(KpVsysXaxis,np.ones_like(KpVsysXaxis)*QuickStd,'--')
        plt.plot(KpVsysXaxis,-np.ones_like(KpVsysXaxis)*QuickStd,'--')
        
        plt.plot(KpVsysXaxis,np.ones_like(KpVsysXaxis)*QuickStd*3,':')
        plt.plot(KpVsysXaxis,-np.ones_like(KpVsysXaxis)*QuickStd*3,':')
        
            
            
            
        plt.tight_layout()        
        pdf.savefig()       
        plt.close()
        
        if SaveLoadable:
            LoadableKpCCF = np.zeros((len(RadVelVect),2))
            LoadableKpCCF[:,0] = RadVelVect
            LoadableKpCCF[:,1] = KpVsys[ClosestKpRow,:]
            np.savetxt('%s/%s.txt'%(SavePath,name),LoadableKpCCF)
            
            np.save('%s/%s_KpVsys.npy'%(SavePath,name),KpVsys)    
        
        print('measured SNR of %s %.10f'%(name,DetSig[0]))




def LoadAllCCFs(night,TargetForCrossCor,ModelScalingFactor,arm,arm_subpart,SysremIterations,ModelForCrossCor):
    
    CCFLoadDir0 = '%s/KELT-9b_CARMENES_emission_data/XCorr/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d'%(FirstPartOfLoadPath,
                                                                       night,
                                                                       TargetForCrossCor,
                                                                       ModelForCrossCor,
                                                                       ModelScalingFactor,
                                                                       arm,
                                                                       arm_subpart,
                                                                       0,
                                                                       SysremIterations[0])
    
    CCF0 = np.load('%s/Xcorr_order%d_Sysrem%d.npy'%(CCFLoadDir0,0,SysremIterations[0]))
    
    RadVelsForCCFs = np.loadtxt('%s/sysremed_%d_XCorrRadVelVect.txt'%(CCFLoadDir0,SysremIterations[0]))
    
    NumRowsCCF,NumColsCCF = np.shape(CCF0)
    
    CCFPerOrderArray = np.zeros((NumRowsCCF,NumColsCCF,len(SysremIterations)))
    
    for OrderIndex in range(len(SysremIterations)):
        
        CCFLoadDir = '%s/KELT-9b_CARMENES_emission_data/XCorr/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d'%(FirstPartOfLoadPath,
                                                                           night,
                                                                           TargetForCrossCor,
                                                                           ModelForCrossCor,
                                                                           ModelScalingFactor,
                                                                           arm,
                                                                           arm_subpart,
                                                                           OrderIndex,
                                                                           SysremIterations[OrderIndex])
        ####################################################
        
        CCF = np.load('%s/Xcorr_order%d_Sysrem%d.npy'%(CCFLoadDir,OrderIndex,SysremIterations[OrderIndex]))

        CCFPerOrderArray[:,:,OrderIndex] = CCF
        
    return RadVelsForCCFs, CCFPerOrderArray


#DataProcessingVersion = 'ProcessedDataNoRotBroadening'    
DataProcessingVersion = 'ProcessedData'        

DataOrigin = 'KELT-9b_CARMENES_emission_data'

FirstPartOfLoadPath = '../CrossCorrelationDataAndProcessing'
#FirstPartOfLoadPath = 'F:'

ModelShift_kms = 0  ## Refers to the model scaling factor used when injecting the model spectrum 
# SysremComponentsRemoved = 6
#SysremComponentsRemovedForOrderWeights = 6

NumVisOrders = 61
NumNirOrders = 28

#ConstSysremIterations = 6
# VisASysremIterationsVect = np.ones(NumVisOrders)*ConstSysremIterations
# NirASysremIterationsVect = np.ones(NumNirOrders)*ConstSysremIterations
# NirBSysremIterationsVect = np.ones(NumNirOrders)*ConstSysremIterations


# VisASysremIterationsUsedForFindingWeightsVect = VisASysremIterationsVect
# NirASysremIterationsUsedForFindingWeightsVect = NirASysremIterationsVect
# NirBSysremIterationsUsedForFindingWeightsVect = NirBSysremIterationsVect


# SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig1.csv')
# SysremItDescriptor = 'SysremItConfig1'

# SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig2.csv')
# SysremItDescriptor = 'SysremItConfig2'


# SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig4.csv')
# SysremItDescriptor = 'SysremConfig4'

SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig6.csv')
SysremItDescriptor = 'SysremConfig6'

#CCFMedSub = 0 ## Unmodified CCFs without subtracting the median 
CCFMedSub = 1 ## Subtract from each CCF its median 
#CCFMedSub = 2 ## Subtract from each CCF its meidan then subtract the median from every column 

#KpMode = 'SingleKp'
KpMode = 'MultiKp'

#SpecWeightConfig = 1 ## Weighting by product of Lambert and squared average error per spectrum 
#SpecWeightConfig = 2  ## Weighting by product of Lambert and average error per spectrum 
SpecWeightConfig = 3 ## Not weighting spec 
#SpecWeightConfig = 4 ## Weighting by Lambert sphere 

########################################
### $$$
#OrderWeightConfig = 1 ## Weighting by injected signal recovery
OrderWeightConfig = 2  ## Weighting by 1 or 0 if no injected signal recovered
#OrderWeightConfig = 3  ## weighting all by 1 
#OrderWeightConfig = 0 ## Override weights provided 
#OrderWeightConfig = 100 ## Override weights provided 

LoadOrderWeights = True

#SysremItDescriptor = 'SysremItConfigAll6'
AllorOddorEven = 'All'
#AllorOddorEven = 'Odd'
#AllorOddorEven = 'Even'



Night20180609AllVisSysremIts = SysremItConfig_df['20180609All_vis_A'].values
Night20180618AllVisSysremIts = SysremItConfig_df['20180618All_vis_A'].values
Night20190528AllVisSysremIts = SysremItConfig_df['20190528P2_vis_A'].values
Night20190604AllVisSysremIts = SysremItConfig_df['20190604All_vis_A'].values


Night20180609AllNirASysremIts = SysremItConfig_df['20180609All_nir_A'].values[0:NumNirOrders]
Night20180618AllNirASysremIts = SysremItConfig_df['20180618All_nir_A'].values[0:NumNirOrders]
Night20190528AllNirASysremIts = SysremItConfig_df['20190528P2_nir_A'].values[0:NumNirOrders]
Night20190604AllNirASysremIts = SysremItConfig_df['20190604All_nir_A'].values[0:NumNirOrders]

# Night20180609AllNirBSysremIts = SysremItConfig_df['20180609All_nir_B'].values[0:NumNirOrders]
# Night20180618AllNirBSysremIts = SysremItConfig_df['20180618All_nir_B'].values[0:NumNirOrders]
# Night20190528AllNirBSysremIts = SysremItConfig_df['20190528All_nir_B'].values[0:NumNirOrders]
# Night20190604AllNirBSysremIts = SysremItConfig_df['20190604All_nir_B'].values[0:NumNirOrders]



# Night20180609AllVisSysremIts[:] = 6
# Night20180618AllVisSysremIts[:] = 6
# Night20190528AllVisSysremIts[:] = 6
# Night20190604AllVisSysremIts[:] = 6

# Night20180609AllNirASysremIts[:] = 6
# Night20180618AllNirASysremIts[:] = 6
# Night20190528AllNirASysremIts[:] = 6
# Night20190604AllNirASysremIts[:] = 6

# Night20180609AllNirBSysremIts[:] = 6
# Night20180618AllNirBSysremIts[:] = 6
# Night20190528AllNirBSysremIts[:] = 6
# Night20190604AllNirBSysremIts[:] = 6

Night20180609AllVisSysremItsForWeights = np.copy(Night20180609AllVisSysremIts)
Night20180618AllVisSysremItsForWeights = np.copy(Night20180618AllVisSysremIts)
Night20190528AllVisSysremItsForWeights = np.copy(Night20190528AllVisSysremIts)
Night20190604AllVisSysremItsForWeights = np.copy(Night20190604AllVisSysremIts)

Night20180609AllNirASysremItsForWeights = np.copy(Night20180609AllNirASysremIts)
Night20180618AllNirASysremItsForWeights = np.copy(Night20180618AllNirASysremIts)
Night20190528AllNirASysremItsForWeights = np.copy(Night20190528AllNirASysremIts)
Night20190604AllNirASysremItsForWeights = np.copy(Night20190604AllNirASysremIts)


# Night20180609AllNirBSysremItsForWeights = np.copy(Night20180609AllNirBSysremIts)
# Night20180618AllNirBSysremItsForWeights = np.copy(Night20180618AllNirBSysremIts)
# Night20190528AllNirBSysremItsForWeights = np.copy(Night20190528AllNirBSysremIts)
# Night20190604AllNirBSysremItsForWeights = np.copy(Night20190604AllNirBSysremIts)





############################

# ModelForXCor = 'KELT9b_Al_0.50_+0.0_0.55_Vrot6.63'
# ModelForWeights = ModelForXCor
# # ModelForWeights = 'KELT9b_Al_0.50_+0.0_0.55_Vrot6.63'

# # InjectedSignalScalingFactor = 0
# # TargetForCrossCor = 'NoModel'

# TargetForCrossCor = ModelForXCor
# InjectedSignalScalingFactor = 1




# InjectionScalingFactorFor20190528All = 75

# InjectionScalingFactorFor20180609All = 300
# InjectionScalingFactorFor20180618All = 120
# InjectionScalingFactorFor20190604All = 450


# ModelForXCor = 'KELT9b_Al_0.50_+2.3_0.55_Vrot6.63'
# ModelForWeights = ModelForXCor
# TargetForCrossCor = 'KELT9b_Al_0.50_+2.3_0.55_Vrot6.63'


# InjectedSignalScalingFactor = 1

# InjectionScalingFactorFor20190528All = 75

# InjectionScalingFactorFor20180609All = 300
# InjectionScalingFactorFor20180618All = 120
# InjectionScalingFactorFor20190604All = 450







# ModelForXCor = 'KELT9b_Ca_0.50_+0.0_0.55_Vrot6.63'
# ModelForWeights = ModelForXCor
# TargetForCrossCor = 'NoModel'
# InjectedSignalScalingFactor = 0

# # TargetForCrossCor = ModelForXCor
# # InjectedSignalScalingFactor = 1

# InjectionScalingFactorFor20190528All = 100

# InjectionScalingFactorFor20180609All = 500
# InjectionScalingFactorFor20180618All = 130
# InjectionScalingFactorFor20190604All = 600




# ModelForXCor = 'KELT9b_CaII_0.50_+0.0_0.55_Vrot6.63'
# ModelForWeights = ModelForXCor
# TargetForCrossCor = 'NoModel'
# InjectedSignalScalingFactor = 0

# # TargetForCrossCor = ModelForXCor
# # InjectedSignalScalingFactor = 1


# InjectionScalingFactorFor20190528All = 200

# InjectionScalingFactorFor20180609All = 1000
# InjectionScalingFactorFor20180618All = 270
# InjectionScalingFactorFor20190604All = 1000



# OLD 
# #InjectionScalingFactorFor20190528All = 300

# #InjectionScalingFactorFor20180609All = 3000
# #InjectionScalingFactorFor20180618All = 540
# #InjectionScalingFactorFor20190604All = 3000





# ModelForXCor = 'KELT9b_Cr_0.50_+0.0_0.55_Vrot6.63'
# ModelForWeights = ModelForXCor
# TargetForCrossCor = 'NoModel'
# InjectedSignalScalingFactor = 0

# # TargetForCrossCor = ModelForXCor
# # InjectedSignalScalingFactor = 1

# InjectionScalingFactorFor20190528All = 40

# InjectionScalingFactorFor20180609All = 200
# InjectionScalingFactorFor20180618All = 50
# InjectionScalingFactorFor20190604All = 200


ModelForXCor = 'KELT9b_Fe_0.50_+0.0_0.55_Vrot6.63'
ModelForWeights = ModelForXCor

InjectedSignalScalingFactor = 0
TargetForCrossCor = 'NoModel'


InjectionScalingFactorFor20190528All = 30

InjectionScalingFactorFor20180609All = 150
InjectionScalingFactorFor20180618All = 40
InjectionScalingFactorFor20190604All = 150


# ModelForXCor = 'KELT9b_FeH_main_iso_0.50_+0.0_0.55_Vrot6.63'

# ModelForWeights = ModelForXCor
# # InjectedSignalScalingFactor = 0
# # TargetForCrossCor = 'NoModel'
# TargetForCrossCor = ModelForXCor
# InjectedSignalScalingFactor = 1

# InjectionScalingFactorFor20190528All = 150

# InjectionScalingFactorFor20180609All = 750
# InjectionScalingFactorFor20180618All = 200
# InjectionScalingFactorFor20190604All = 750

# ModelForXCor = 'KELT9b_FeII_0.50_+0.0_0.55_Vrot6.63'
# ModelForWeights = ModelForXCor
# # TargetForCrossCor = 'NoModel'
# # InjectedSignalScalingFactor = 0

# TargetForCrossCor = ModelForXCor
# InjectedSignalScalingFactor = 1

# InjectionScalingFactorFor20190528All = 300

# InjectionScalingFactorFor20180609All = 1500
# InjectionScalingFactorFor20180618All = 400
# InjectionScalingFactorFor20190604All = 1500

# ModelForXCor = 'KELT9b_FeII_UsingFeI_0.50_+0.0_0.55_Vrot6.63'
# ModelForWeights = ModelForXCor
# TargetForCrossCor = 'NoModel'
# InjectedSignalScalingFactor = 0

# # TargetForCrossCor = ModelForXCor
# # InjectedSignalScalingFactor = 1

# InjectionScalingFactorFor20190528All = 300

# InjectionScalingFactorFor20180609All = 1500
# InjectionScalingFactorFor20180618All = 400
# InjectionScalingFactorFor20190604All = 1500



# ModelForXCor = 'KELT9b_K_0.50_+0.0_0.55_Vrot6.63'
# ModelForWeights = ModelForXCor
# # TargetForCrossCor = 'NoModel'
# # InjectedSignalScalingFactor = 0

# TargetForCrossCor = ModelForXCor
# InjectedSignalScalingFactor = 1

# # InjectionScalingFactorFor20190528All = 300

# # InjectionScalingFactorFor20180609All = 1500
# # InjectionScalingFactorFor20180618All = 400
# # InjectionScalingFactorFor20190604All = 1500  

# InjectionScalingFactorFor20190528All = 100

# InjectionScalingFactorFor20180609All = 500
# InjectionScalingFactorFor20180618All = 130
# InjectionScalingFactorFor20190604All = 500  


# ModelForXCor = 'KELT9b_Mg_0.50_+0.0_0.55_Vrot6.63'
# ModelForWeights = ModelForXCor
# TargetForCrossCor = 'NoModel'
# InjectedSignalScalingFactor = 0

# # TargetForCrossCor = ModelForXCor
# # InjectedSignalScalingFactor = 1

# InjectionScalingFactorFor20190528All = 80

# InjectionScalingFactorFor20180609All = 400
# InjectionScalingFactorFor20180618All = 100
# InjectionScalingFactorFor20190604All = 400




# ModelForXCor = 'KELT9b_Na_0.50_+0.0_0.55_Vrot6.63'
# ModelForWeights = ModelForXCor
# # TargetForCrossCor = 'NoModel'
# # InjectedSignalScalingFactor = 0

# TargetForCrossCor = ModelForXCor
# InjectedSignalScalingFactor = 1

# # InjectionScalingFactorFor20190528All = 300

# # InjectionScalingFactorFor20180609All = 1500
# # InjectionScalingFactorFor20180618All = 390
# # InjectionScalingFactorFor20190604All = 1500


# InjectionScalingFactorFor20190528All = 100

# InjectionScalingFactorFor20180609All = 500
# InjectionScalingFactorFor20180618All = 130
# InjectionScalingFactorFor20190604All = 500


  





# ModelForXCor = 'KELT9b_Ti_0.50_+0.0_0.55_Vrot6.63'
# ModelForWeights = ModelForXCor
# TargetForCrossCor = 'NoModel'
# InjectedSignalScalingFactor = 0

# # TargetForCrossCor = ModelForXCor
# # InjectedSignalScalingFactor = 1

# InjectionScalingFactorFor20190528All = 25

# InjectionScalingFactorFor20180609All = 150
# InjectionScalingFactorFor20180618All = 45
# InjectionScalingFactorFor20190604All = 150






# ModelForXCor = 'KELT9b_TiO_all_iso_Plez_0.50_+0.0_0.55_Vrot6.63'
# ModelForWeights = ModelForXCor
# TargetForCrossCor = 'NoModel'

# InjectedSignalScalingFactor = 0

# ## Some weights are too strong 
# # InjectionScalingFactorFor20190528All = 150

# # InjectionScalingFactorFor20180609All = 1000
# # InjectionScalingFactorFor20180618All = 260
# # InjectionScalingFactorFor20190604All = 1000

# # # # ### ok values V1
# InjectionScalingFactorFor20190528All = 150

# InjectionScalingFactorFor20180609All = 750
# InjectionScalingFactorFor20180618All = 195
# InjectionScalingFactorFor20190604All = 750

# # Weight values V2, actually give a slightly worse result 
# InjectionScalingFactorFor20190528All = 230

# InjectionScalingFactorFor20180609All = 1000
# InjectionScalingFactorFor20180618All = 300
# InjectionScalingFactorFor20190604All = 1000

# ### stronger weights 
# InjectionScalingFactorFor20190528All = 300

# InjectionScalingFactorFor20180609All = 1600
# InjectionScalingFactorFor20180618All = 400
# InjectionScalingFactorFor20190604All = 1800




# ModelForXCor = 'KELT9b_TiO_48_Exomol_McKemmish_0.50_+0.0_0.55_Vrot6.63'
# ModelForWeights = ModelForXCor
# InjectedSignalScalingFactor = 0
# TargetForCrossCor = 'NoModel'

# # InjectedSignalScalingFactor = 1
# # TargetForCrossCor = ModelForXCor

# InjectionScalingFactorFor20190528All = 150

# InjectionScalingFactorFor20180609All = 750
# InjectionScalingFactorFor20180618All = 195
# InjectionScalingFactorFor20190604All = 750

# ModelForXCor = 'KELT9b_TiO_48_Plez_0.50_+0.0_0.55_Vrot6.63'
# ModelForWeights = ModelForXCor
# TargetForCrossCor = 'NoModel'

# InjectedSignalScalingFactor = 0

# InjectionScalingFactorFor20190528All = 150

# InjectionScalingFactorFor20180609All = 750
# InjectionScalingFactorFor20180618All = 195
# InjectionScalingFactorFor20190604All = 750

############################################################



#VisOverrideWeights = np.ones((NumVisOrders))*np.nan
# NirAOverrideWeights = np.ones((NumNirOrders))*0
# NirBOverrideWeights = np.copy(NirAOverrideWeights)*0

# VisOverrideWeights[10:19] = 0
# VisOverrideWeights[21:24] = 0
# VisOverrideWeights[42:45] = 0
# VisOverrideWeights[46:51] = 0

if (OrderWeightConfig == 3) or (OrderWeightConfig == 0):  ### no order weightings 
    InjStrForWeights = '%d_%d_%d_%d'%(0,0,0,0)
    
else:
    InjStrForWeights = '%d_%d_%d_%d'%(InjectionScalingFactorFor20180609All,
                                      InjectionScalingFactorFor20180618All,
                                      InjectionScalingFactorFor20190528All,
                                      InjectionScalingFactorFor20190604All)




### !!! 
PlotConfidenceIntervalString = 'WithConfInt'
#PlotConfidenceIntervalString = 'NoConfInt'

SavePath = 'plots/%s/%s/OrderWeightConfig%d/SpecWeightConfig%d/CCFMedSub%d/CombinedPlots/%s/%s_%.2e/%s/%s/%s/%s'%(DataProcessingVersion,
                                                  KpMode,
                                                  OrderWeightConfig,
                                                  SpecWeightConfig,
                                                  CCFMedSub,
                                                  SysremItDescriptor,
                                                  TargetForCrossCor,
                                                  InjectedSignalScalingFactor,                                                  
                                                  ModelForXCor,
                                                  InjStrForWeights,
                                                  AllorOddorEven,
                                                  PlotConfidenceIntervalString)

print()
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('SavePath')
print(SavePath)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print()


    
if not os.path.exists(SavePath):
    os.makedirs(SavePath)








VisOverrideWeights = None
NirAOverrideWeights = None
NirBOverrideWeights = None




#######################

if InjectedSignalScalingFactor == 0:

    # NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9 = LoadKpPerOrderAndWeights('NoModel','20180609All','vis','A',0,Night20180609AllVisSysremIts,ModelForXCor,InjectionScalingFactorFor20180609All,Night20180609AllVisSysremItsForWeights,VisOverrideWeights)
    # NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9 = LoadKpPerOrderAndWeights('NoModel','20180609All','nir','A',0,Night20180609AllNirASysremIts,ModelForXCor,InjectionScalingFactorFor20180609All,Night20180609AllNirASysremItsForWeights,NirAOverrideWeights)
    # # NoModel_20180609All_nir_B_0_0_9_KELT9b_FeI_50_9 = LoadKpPerOrderAndWeights('NoModel','20180609All','nir','B',0,Night20180609AllNirBSysremIts,ModelForXCor,InjectionScalingFactorFor20180609All,Night20180609AllNirBSysremItsForWeights,NirBOverrideWeights)
    
    
    # NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9 = LoadKpPerOrderAndWeights('NoModel','20180618All','vis','A',0,Night20180618AllVisSysremIts,ModelForXCor,InjectionScalingFactorFor20180618All,Night20180618AllVisSysremItsForWeights,VisOverrideWeights)
    # NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9 = LoadKpPerOrderAndWeights('NoModel','20180618All','nir','A',0,Night20180618AllNirASysremIts,ModelForXCor,InjectionScalingFactorFor20180618All,Night20180618AllNirASysremItsForWeights,NirAOverrideWeights)
    # # NoModel_20180618All_nir_B_0_0_9_KELT9b_FeI_10_9 = LoadKpPerOrderAndWeights('NoModel','20180618All','nir','B',0,Night20180618AllNirBSysremIts,ModelForXCor,InjectionScalingFactorFor20180618All,Night20180618AllNirBSysremItsForWeights,NirBOverrideWeights)
    
    
    
    # NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9 = LoadKpPerOrderAndWeights('NoModel','20190528All','vis','A',0,Night20190528AllVisSysremIts,ModelForXCor,InjectionScalingFactorFor20190528All,Night20190528AllVisSysremItsForWeights,VisOverrideWeights)
    # NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9 = LoadKpPerOrderAndWeights('NoModel','20190528All','nir','A',0,Night20190528AllNirASysremIts,ModelForXCor,InjectionScalingFactorFor20190528All,Night20190528AllNirASysremItsForWeights,NirAOverrideWeights)
    # # NoModel_20190528All_nir_B_0_0_9_KELT9b_FeI_10_9 = LoadKpPerOrderAndWeights('NoModel','20190528All','nir','B',0,Night20190528AllNirBSysremIts,ModelForXCor,InjectionScalingFactorFor20190528All,Night20190528AllNirBSysremItsForWeights,NirBOverrideWeights)
    
    
    # NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9 = LoadKpPerOrderAndWeights('NoModel','20190604All','vis','A',0,Night20190604AllVisSysremIts,ModelForXCor,InjectionScalingFactorFor20190604All,Night20190604AllVisSysremItsForWeights,VisOverrideWeights)
    # NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9 = LoadKpPerOrderAndWeights('NoModel','20190604All','nir','A',0,Night20190604AllNirASysremIts,ModelForXCor,InjectionScalingFactorFor20190604All,Night20190604AllNirASysremItsForWeights,NirAOverrideWeights)
    # NoModel_20190604All_nir_B_0_0_9_KELT9b_FeI_50_9 = LoadKpPerOrderAndWeights('NoModel','20190604All','nir','B',0,Night20190604AllNirBSysremIts,ModelForXCor,InjectionScalingFactorFor20190604All,Night20190604AllNirBSysremItsForWeights,NirBOverrideWeights)


    
    ###### !!! only load 618 and 528     

    
    
    NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9 = LoadKpPerOrderAndWeights('NoModel','20180618All','vis','A',0,Night20180618AllVisSysremIts,ModelForXCor,InjectionScalingFactorFor20180618All,Night20180618AllVisSysremItsForWeights,VisOverrideWeights)
    NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9 = LoadKpPerOrderAndWeights('NoModel','20180618All','nir','A',0,Night20180618AllNirASysremIts,ModelForXCor,InjectionScalingFactorFor20180618All,Night20180618AllNirASysremItsForWeights,NirAOverrideWeights)
    
    
    
    NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9 = LoadKpPerOrderAndWeights('NoModel','20190528P2','vis','A',0,Night20190528AllVisSysremIts,ModelForXCor,InjectionScalingFactorFor20190528All,Night20190528AllVisSysremItsForWeights,VisOverrideWeights)
    NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9 = LoadKpPerOrderAndWeights('NoModel','20190528P2','nir','A',0,Night20190528AllNirASysremIts,ModelForXCor,InjectionScalingFactorFor20190528All,Night20190528AllNirASysremItsForWeights,NirAOverrideWeights)
    

    NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9 = NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9
    NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9 = NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9
    

    
    NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9 = NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9
    NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9 = NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9
  

    
else:
    # NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9 = LoadKpPerOrderAndWeights(ModelForXCor,'20180609All','vis','A',InjectedSignalScalingFactor,Night20180609AllVisSysremIts,ModelForXCor,InjectionScalingFactorFor20180609All,Night20180609AllVisSysremItsForWeights,VisOverrideWeights)
    # NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9 = LoadKpPerOrderAndWeights(ModelForXCor,'20180609All','nir','A',InjectedSignalScalingFactor,Night20180609AllNirASysremIts,ModelForXCor,InjectionScalingFactorFor20180609All,Night20180609AllNirASysremItsForWeights,NirAOverrideWeights)
    
    
    NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9 = LoadKpPerOrderAndWeights(ModelForXCor,'20180618All','vis','A',InjectedSignalScalingFactor,Night20180618AllVisSysremIts,ModelForXCor,InjectionScalingFactorFor20180618All,Night20180618AllVisSysremItsForWeights,VisOverrideWeights)
    NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9 = LoadKpPerOrderAndWeights(ModelForXCor,'20180618All','nir','A',InjectedSignalScalingFactor,Night20180618AllNirASysremIts,ModelForXCor,InjectionScalingFactorFor20180618All,Night20180618AllNirASysremItsForWeights,NirAOverrideWeights)
    
    
    
    NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9 = LoadKpPerOrderAndWeights(ModelForXCor,'20190528P2','vis','A',InjectedSignalScalingFactor,Night20190528AllVisSysremIts,ModelForXCor,InjectionScalingFactorFor20190528All,Night20190528AllVisSysremItsForWeights,VisOverrideWeights)
    NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9 = LoadKpPerOrderAndWeights(ModelForXCor,'20190528P2','nir','A',InjectedSignalScalingFactor,Night20190528AllNirASysremIts,ModelForXCor,InjectionScalingFactorFor20190528All,Night20190528AllNirASysremItsForWeights,NirAOverrideWeights)
    
    
    # NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9 = LoadKpPerOrderAndWeights(ModelForXCor,'20190604All','vis','A',InjectedSignalScalingFactor,Night20190604AllVisSysremIts,ModelForXCor,InjectionScalingFactorFor20190604All,Night20190604AllVisSysremItsForWeights,VisOverrideWeights)
    # NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9 = LoadKpPerOrderAndWeights(ModelForXCor,'20190604All','nir','A',InjectedSignalScalingFactor,Night20190604AllNirASysremIts,ModelForXCor,InjectionScalingFactorFor20190604All,Night20190604AllNirASysremItsForWeights,NirAOverrideWeights)
    
    
    NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9 = NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9
    NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9 = NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9
    
    NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9 = NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9
    NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9 = NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9

########################################################################################################
######## When using all 4 nights,  this part was to set orders with a non-zero weight from only one night to zero 
    
NumVisOrders = len(NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[3])
NumNirOrders = len(NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[3])

VIS_DetSigs = np.zeros((4,NumVisOrders))
NIRA_DetSigs = np.zeros((4,NumNirOrders))
NIRB_DetSigs = np.zeros((4,NumNirOrders))

VIS_DetSigs[0,:] = NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[3]
VIS_DetSigs[1,:] = NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[3]
VIS_DetSigs[2,:] = NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[3]
VIS_DetSigs[3,:] = NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9[3]

NIRA_DetSigs[0,:] = NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[3]
NIRA_DetSigs[1,:] = NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[3]
NIRA_DetSigs[2,:] = NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[3]
NIRA_DetSigs[3,:] = NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9[3]

# np.savetxt('TiO_VIS_order_weights.txt',VIS_DetSigs)
# np.savetxt('TiO_NIR_order_weights.txt',NIRA_DetSigs)

# raise Exception


# NIRB_DetSigs[0,:] = NoModel_20180609All_nir_B_0_0_9_KELT9b_FeI_50_9[3]
# NIRB_DetSigs[1,:] = NoModel_20180618All_nir_B_0_0_9_KELT9b_FeI_10_9[3]
# NIRB_DetSigs[2,:] = NoModel_20190528All_nir_B_0_0_9_KELT9b_FeI_10_9[3]
# NIRB_DetSigs[3,:] = NoModel_20190604All_nir_B_0_0_9_KELT9b_FeI_50_9[3]

Initial_VIS_DetSigs = np.copy(VIS_DetSigs)
Initial_NIRA_DetSigs = np.copy(NIRA_DetSigs)
Initial_NIRB_DetSigs = np.copy(NIRB_DetSigs)


VIS_NumNonZeroPerCol = np.count_nonzero(VIS_DetSigs,axis=0)
NIRA_NumNonZeroPerCol = np.count_nonzero(NIRA_DetSigs,axis=0)
NIRB_NumNonZeroPerCol = np.count_nonzero(NIRB_DetSigs,axis=0)

MinNumberNonZero = 2

#### This is the part that zeros orders with only signal in one night 

for i in range(NumVisOrders):
    if VIS_NumNonZeroPerCol[i] < MinNumberNonZero:
        VIS_DetSigs[:,i] = 0.0
        

for i in range(NumNirOrders):
    if NIRA_NumNonZeroPerCol[i] < MinNumberNonZero:
        NIRA_DetSigs[:,i] = 0.0        
        
    if NIRB_NumNonZeroPerCol[i] < MinNumberNonZero:
        NIRB_DetSigs[:,i] = 0.0    
        
############
### For overriding weights         
        
if (OrderWeightConfig == 0) or (OrderWeightConfig == 100): #### Override weights 
    
    ManuallyIdentifiedWeights_df = pd.read_csv('Manual_order_weights/KELT-9b_vis_manual_weights - Sheet1.csv')
    
    vis_OrderWeightsFromSpec = ManuallyIdentifiedWeights_df['vis_WeightFromOrderNoise'].values
    vis_OrderWeightsFromSpec = ManuallyIdentifiedWeights_df['vis_%s'%(ModelForXCor)].values    
    vis_manual_weights = vis_OrderWeightsFromSpec*vis_OrderWeightsFromSpec
    
    
    nir_OrderWeightsFromSpec = ManuallyIdentifiedWeights_df['nir_WeightFromOrderNoise'].values[0:28]
    nir_OrderWeightsFromSpec = ManuallyIdentifiedWeights_df['nir_%s'%(ModelForXCor)].values[0:28]    
    nir_manual_weights = nir_OrderWeightsFromSpec*nir_OrderWeightsFromSpec

    
    
    VIS_DetSigs[0,:] = vis_manual_weights
    VIS_DetSigs[1,:] = vis_manual_weights
    VIS_DetSigs[2,:] = vis_manual_weights
    VIS_DetSigs[3,:] = vis_manual_weights
    
    NIRA_DetSigs[0,:] = nir_manual_weights
    NIRA_DetSigs[1,:] = nir_manual_weights
    NIRA_DetSigs[2,:] = nir_manual_weights
    NIRA_DetSigs[3,:] = nir_manual_weights
    
    NIRB_DetSigs[:] = 0.0
    
        
#     OverrideNirAWeights = np.zeros(28)
#     OverrideNirBWeights = np.zeros(28)
    
#     OverrideVisWeights = np.ones(61)
    
#     ## [16:18, 22, 52:56] 
#     ## [1:20, 22, 24, 50, 51, 54:56] 
    
#     ### intrinsic to the line list 
#     OverrideVisWeights[16:19] = 0.0  ## up to but not including 
#     OverrideVisWeights[22] = 0.0  ## up to but not including 
#     OverrideVisWeights[52:57] = 0.0
    
#     ## From Cont's injection tests 
#     # OverrideVisWeights[1:21] = 0.0
    
#     # OverrideVisWeights[22] = 0.0
    
#     # OverrideVisWeights[24] = 0.0
    
#     # OverrideVisWeights[50] = 0.0
    
#     # OverrideVisWeights[51] = 0.0
    
#     # OverrideVisWeights[54:57] = 0.0
    
#     VIS_DetSigs[0,:] = OverrideVisWeights
#     VIS_DetSigs[1,:] = OverrideVisWeights
#     VIS_DetSigs[2,:] = OverrideVisWeights
#     VIS_DetSigs[3,:] = OverrideVisWeights
    
#     NIRA_DetSigs[:] = 0.0
#     NIRB_DetSigs[:] = 0.0

###########
####################
        

#### For implementing Ernst's thought of no order weights 
if OrderWeightConfig == 1:
    ### leave them as they are 
    pass
    
if OrderWeightConfig == 2:
    VIS_DetSigs[VIS_DetSigs>0] = 1
    NIRA_DetSigs[NIRA_DetSigs>0] = 1 
    NIRB_DetSigs[NIRB_DetSigs>0] = 1 
    
if OrderWeightConfig == 3:
    
    VIS_DetSigs[:] = 1
    NIRA_DetSigs[:] = 1 
    NIRB_DetSigs[:] = 1    

### For TiO 
if ModelForXCor == 'KELT9b_TiO_48_Exomol_McKemmish_0.50_+0.0_0.55_Vrot6.63':
    NIRA_DetSigs[:] = 0.0
    
if ModelForXCor == 'KELT9b_CaII_0.50_+0.0_0.55_Vrot6.63':
    NIRA_DetSigs[:] = 0.0

##### To put the weights from the VIS_DetSigs and NIRA_DetSigs back into the weight list 
    
### Using unique weights from each night 
# NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[3] = VIS_DetSigs[0,:]
# NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[3] = VIS_DetSigs[1,:] 
# NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[3] = VIS_DetSigs[2,:]
# NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9[3] = VIS_DetSigs[3,:]


# NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[3] = NIRA_DetSigs[0,:]
# NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[3] = NIRA_DetSigs[1,:]
# NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[3] = NIRA_DetSigs[2,:]
# NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9[3] = NIRA_DetSigs[3,:]


############################################################################
    
### Using weights for 618 and 528. Putting the modified weights back into hte weight lists 
NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[3] = VIS_DetSigs[2,:]
NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[3] = VIS_DetSigs[1,:] 
NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[3] = VIS_DetSigs[2,:]
NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9[3] = VIS_DetSigs[2,:]


NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[3] = NIRA_DetSigs[2,:]
NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[3] = NIRA_DetSigs[1,:]
NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[3] = NIRA_DetSigs[2,:]
NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9[3] = NIRA_DetSigs[2,:]





#############################################################
### Using the weights for 20190528All for all nights (probably most sense for weights of 0 and 1)
    
# NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[3] = VIS_DetSigs[2,:]
# NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[3] = VIS_DetSigs[2,:] 
# NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[3] = VIS_DetSigs[2,:]
# NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9[3] = VIS_DetSigs[2,:]


# NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[3] = NIRA_DetSigs[2,:]
# NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[3] = NIRA_DetSigs[2,:]
# NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[3] = NIRA_DetSigs[2,:]
# NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9[3] = NIRA_DetSigs[2,:]


####### End putting weights from the VIS_DetSigs and NIRA_DetSigs back into the weight list  
##########################################################################



### Plot the detection significances from a strong injected signal to inform weights. 
#### Include from all orders and rejecting orders with a spurious signal 
with PdfPages('%s/RecoveredDetSigForWeights_From_pl.pdf'%(SavePath)) as OrderWeightsPDF1:
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(Initial_VIS_DetSigs[0,:],label='20180609All Inj %dx'%(InjectionScalingFactorFor20180609All))
    plt.plot(Initial_VIS_DetSigs[1,:],label='20180618All Inj %dx'%(InjectionScalingFactorFor20180618All))
    plt.plot(Initial_VIS_DetSigs[2,:],label='20190528All Inj %dx'%(InjectionScalingFactorFor20190528All))
    plt.plot(Initial_VIS_DetSigs[3,:],label='20190604All Inj %dx'%(InjectionScalingFactorFor20190604All))    
    plt.legend()
    plt.title('All Orders (including spurious)')
    
    plt.subplot(2,1,2)
    plt.plot(VIS_DetSigs[0,:],label='20180609All Inj %dx'%(InjectionScalingFactorFor20180609All))
    plt.plot(VIS_DetSigs[1,:],label='20180618All Inj %dx'%(InjectionScalingFactorFor20180618All))
    plt.plot(VIS_DetSigs[2,:],label='20190528All Inj %dx'%(InjectionScalingFactorFor20190528All))
    plt.plot(VIS_DetSigs[3,:],label='20190604All Inj %dx'%(InjectionScalingFactorFor20190604All))
    plt.title('Rejecting spurious orders')
    
    plt.suptitle('VIS arm')
    
    OrderWeightsPDF1.savefig()
    plt.close()
    
    plt.figure()
    plt.subplot(2,1,1)
    
    plt.plot(Initial_NIRA_DetSigs[0,:],label='20180609All Inj %dx'%(InjectionScalingFactorFor20180609All))
    plt.plot(Initial_NIRA_DetSigs[1,:],label='20180618All Inj %dx'%(InjectionScalingFactorFor20180618All))
    plt.plot(Initial_NIRA_DetSigs[2,:],label='20190528All Inj %dx'%(InjectionScalingFactorFor20190528All))
    plt.plot(Initial_NIRA_DetSigs[3,:],label='20190604All Inj %dx'%(InjectionScalingFactorFor20190604All))    
    plt.legend()
    plt.title('All Orders (including spurious)')

    plt.subplot(2,1,2)
    plt.plot(NIRA_DetSigs[0,:],label='20180609All Inj %dx'%(InjectionScalingFactorFor20180609All))
    plt.plot(NIRA_DetSigs[1,:],label='20180618All Inj %dx'%(InjectionScalingFactorFor20180618All))
    plt.plot(NIRA_DetSigs[2,:],label='20190528All Inj %dx'%(InjectionScalingFactorFor20190528All))
    plt.plot(NIRA_DetSigs[3,:],label='20190604All Inj %dx'%(InjectionScalingFactorFor20190604All))
    plt.title('Rejecting spurious orders')
    
    plt.suptitle('NIR A arm')
    
    OrderWeightsPDF1.savefig()
    plt.close()
    
    
    plt.figure()
    plt.subplot(2,1,1)
    
    plt.plot(Initial_NIRB_DetSigs[0,:],label='20180609All Inj %dx'%(InjectionScalingFactorFor20180609All))
    plt.plot(Initial_NIRB_DetSigs[1,:],label='20180618All Inj %dx'%(InjectionScalingFactorFor20180618All))
    plt.plot(Initial_NIRB_DetSigs[2,:],label='20190528All Inj %dx'%(InjectionScalingFactorFor20190528All))
    plt.plot(Initial_NIRB_DetSigs[3,:],label='20190604All Inj %dx'%(InjectionScalingFactorFor20190604All))    
    plt.legend()
    plt.title('All Orders (including spurious)')

    plt.subplot(2,1,2)
    plt.plot(NIRB_DetSigs[0,:],label='20180609All Inj %dx'%(InjectionScalingFactorFor20180609All))
    plt.plot(NIRB_DetSigs[1,:],label='20180618All Inj %dx'%(InjectionScalingFactorFor20180618All))
    plt.plot(NIRB_DetSigs[2,:],label='20190528All Inj %dx'%(InjectionScalingFactorFor20190528All))
    plt.plot(NIRB_DetSigs[3,:],label='20190604All Inj %dx'%(InjectionScalingFactorFor20190604All))
    plt.title('Rejecting spurious orders')
    
    plt.suptitle('NIR B arm')

    OrderWeightsPDF1.savefig()
    plt.close()
    

##############################################################

if OrderWeightConfig == 1:
    Night20180609AllScalingFactor = InjectionScalingFactorFor20190528All/InjectionScalingFactorFor20180609All
    Night20180618AllScalingFactor = InjectionScalingFactorFor20190528All/InjectionScalingFactorFor20180618All
    Night20190604AllScalingFactor = InjectionScalingFactorFor20190528All/InjectionScalingFactorFor20190604All

else:  
    ### The weights will all be 1 or all 0 and 1 
    Night20180609AllScalingFactor = 1
    Night20180618AllScalingFactor = 1
    Night20190604AllScalingFactor = 1
    
    


print('Night20180609AllScalingFactor')
print(Night20180609AllScalingFactor)
print()
print('Night20180618AllScalingFactor')
print(Night20180618AllScalingFactor)
print()
print('Night20190604AllScalingFactor')
print(Night20190604AllScalingFactor)


# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# print(NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[3])





# Night609OrderList = [NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[0],
#                       NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[0],
#                       NoModel_20180609All_nir_B_0_0_9_KELT9b_FeI_50_9[0]]  

Night609OrderList = [NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[0],
                      NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[0]]  
                      

# Night609OrderWeightsList = [NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[3]*Night20180609AllScalingFactor,
#                             NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[3]*Night20180609AllScalingFactor,
#                             NoModel_20180609All_nir_B_0_0_9_KELT9b_FeI_50_9[3]*Night20180609AllScalingFactor]

Night609OrderWeightsList = [NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[3]*Night20180609AllScalingFactor,
                            NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[3]*Night20180609AllScalingFactor]




Night609VISOrderList = [NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[0]]  
Night609VISOrderWeightsList = [NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[3]*Night20180609AllScalingFactor]

Night609NIROrderList = [NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[0]]  

Night609NIROrderWeightsList = [NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[3]*Night20180609AllScalingFactor]

###################################

Night618OrderList = [NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[0],
                     NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[0]]                      
                      

Night618OrderWeightsList = [NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor,
                            NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor]

Night618VISOrderList = [NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[0]]     
Night618VISOrderWeightsList = [NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor]

Night618NIROrderList = [NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[0]]     


Night618NIROrderWeightsList = [NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor]

##########################################


Night528OrderList = [NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[0],
                      NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[0]]   
                      

Night528OrderWeightsList = [NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[3],
                            NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[3]]





Night528VISOrderList = [NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[0]]                         

Night528VISOrderWeightsList = [NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[3]] 

Night528NIROrderList = [NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[0]]   
                      

Night528NIROrderWeightsList = [NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[3]]  




###################################

Night604OrderList = [NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9[0],
                      NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9[0]]
                      

Night604OrderWeightsList = [NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9[3]*Night20190604AllScalingFactor,
                            NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9[3]*Night20190604AllScalingFactor]





Night604VISOrderList = [NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9[0]]
Night604VISOrderWeightsList = [NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9[3]*Night20190604AllScalingFactor]

Night604NIROrderList = [NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9[0]]
                      

Night604NIROrderWeightsList = [NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9[3]*Night20190604AllScalingFactor]

#########################################



# NightAllOrderList = [NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[0],
#                       NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[0],                 
#                       NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[0],
#                       NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[0],
#                       NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[0],
#                       NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[0],
#                       NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9[0],
#                       NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9[0]]
                      

# NightAllOrderWeightsList = [NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[3]*Night20180609AllScalingFactor,
#                             NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[3]*Night20180609AllScalingFactor,
#                             NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor,
#                             NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor,
#                             NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[3],
#                             NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[3],
#                             NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9[3]*Night20190604AllScalingFactor,
#                             NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9[3]*Night20190604AllScalingFactor]


# NightAllVISOrderList = [NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[0],                   
#                        NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[0],                   
#                        NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[0],                    
#                        NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9[0]]
                      

NightAllVISOrderWeightsList = [NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[3]*Night20180609AllScalingFactor,                          
                              NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor,                         
                              NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[3],          
                              NoModel_20190604All_vis_A_0_0_9_KELT9b_FeI_50_9[3]*Night20190604AllScalingFactor]

# NightAllVISOrderWeightsList = [NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor,                         
#                               NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[3]]

                            


# NightAllNIROrderList = [NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[0],
#                         NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[0],
#                         NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[0],
#                         NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9[0]]
                      

NightAllNIROrderWeightsList = [NoModel_20180609All_nir_A_0_0_9_KELT9b_FeI_50_9[3]*Night20180609AllScalingFactor,
                                NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor,
                                NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[3],
                                NoModel_20190604All_nir_A_0_0_9_KELT9b_FeI_50_9[3]*Night20190604AllScalingFactor]

# NightAllNIROrderWeightsList = [NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor,
#                                 NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[3]]


###########################################

Nights618And528OrderList = [NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[0],
                            NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[0],                                             
                            NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[0],
                            NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[0]]
                            
                      

Nights618And528OrderWeightsList = [NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor,
                                   NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor,
                                   NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[3],
                                   NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[3]]


Nights618VISAnd528VISandNIROrderList = [NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[0],
                            NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[0],
                            NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[0]]
                      

Nights618VISAnd528VISandNIROrderWeightsList = [NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor,
                                   NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[3],
                                   NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[3]]


Nights618and528OnlyVISOrderList = [NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[0],
                                   NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[0]]
                      

Nights618and528OnlyVISOrderWeightsList = [NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor,
                                          NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[3]]




Nights618And528OnlyNIROrderList = [NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[0],
                                  NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[0]]
                      

Nights618And528OnlyNIROrderWeightsList = [NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[3]*Night20180618AllScalingFactor,
                                         NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[3]]


#######################################




with PdfPages('%s/OrderWeightsForAverage_From_pl.pdf'%(SavePath)) as OrderWeightsPDF2:
    
    plt.figure()
    plt.title('VIS')
    plt.plot(NightAllVISOrderWeightsList[0],label='20180609All (scaling factor %.2f)'%(Night20180609AllScalingFactor))
    plt.plot(NightAllVISOrderWeightsList[1],label='20180618All (scaling factor %.2f)'%(Night20180618AllScalingFactor))
    plt.plot(NightAllVISOrderWeightsList[2],label='20190528All (scaling factor %.2f)'%(1))
    plt.plot(NightAllVISOrderWeightsList[3],label='20190604All (scaling factor %.2f)'%(Night20190604AllScalingFactor))
    plt.legend()
    plt.ylabel('Order weight')
    plt.xlabel('order')    
    OrderWeightsPDF2.savefig()
    plt.close()
    
    plt.figure()
    plt.title('NIR A')
    plt.plot(NightAllNIROrderWeightsList[0],label='20180609All')
    plt.plot(NightAllNIROrderWeightsList[1],label='20180618All')
    plt.plot(NightAllNIROrderWeightsList[2],label='20190528All')
    plt.plot(NightAllNIROrderWeightsList[3],label='20190604All')
    plt.legend()
    plt.ylabel('Order weight')
    plt.xlabel('order')
    OrderWeightsPDF2.savefig()
    plt.close()
        




# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# print('Shape of NightAllOrderWeightsList:')
# print(np.shape(np.hstack(NightAllOrderWeightsList)))
# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# print('Shape of NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[3]:')
# print(np.shape(NoModel_20180609All_vis_A_0_0_9_KELT9b_FeI_50_9[3]))
# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')               

# np.savetxt('AllOrderWeightsFromLoadKpVsysForDebugging.txt',np.hstack(NightAllOrderWeightsList))



                            
###################################

# with PdfPages('plots/CombinedPlots/Night528FeIOrderWeights.pdf') as pdf:     
    
#     plt.figure()
#     plt.plot(NoModel_20190528All_vis_A_0_0_10_KELT9bEmission_10[3])
#     pdf.savefig()
    
#     plt.figure()
#     plt.plot(NoModel_20190528All_nir_A_0_0_10_KELT9bEmission_10[3])
#     pdf.savefig()
    
#     plt.figure()
#     plt.plot(NoModel_20190528All_nir_B_0_0_10_KELT9bEmission_10[3])
#     pdf.savefig()
    

Night609WeigtnedAverage = CalculateWeightedAverage(Night609OrderList, Night609OrderWeightsList)


Night609VISWeigtnedAverage = CalculateWeightedAverage(Night609VISOrderList, Night609VISOrderWeightsList)
Night609NIRWeigtnedAverage = CalculateWeightedAverage(Night609NIROrderList, Night609NIROrderWeightsList)

Night618WeigtnedAverage = CalculateWeightedAverage(Night618OrderList, Night618OrderWeightsList)
Night618VISWeigtnedAverage = CalculateWeightedAverage(Night618VISOrderList, Night618VISOrderWeightsList)
Night618NIRWeigtnedAverage = CalculateWeightedAverage(Night618NIROrderList, Night618NIROrderWeightsList)


Night528WeigtnedAverage = CalculateWeightedAverage(Night528OrderList, Night528OrderWeightsList)
Night528VISWeigtnedAverage = CalculateWeightedAverage(Night528VISOrderList, Night528VISOrderWeightsList)
Night528NIRWeigtnedAverage = CalculateWeightedAverage(Night528NIROrderList, Night528NIROrderWeightsList)


Night604WeigtnedAverage = CalculateWeightedAverage(Night604OrderList, Night604OrderWeightsList)
Night604VISWeigtnedAverage = CalculateWeightedAverage(Night604VISOrderList, Night604VISOrderWeightsList)
Night604NIRWeigtnedAverage = CalculateWeightedAverage(Night604NIROrderList, Night604NIROrderWeightsList)

# NightAllWeigtnedAverage = CalculateWeightedAverage(NightAllOrderList, NightAllOrderWeightsList)
# NightAllVISWeigtnedAverage = CalculateWeightedAverage(NightAllVISOrderList, NightAllVISOrderWeightsList)
# NightAllNIRWeigtnedAverage = CalculateWeightedAverage(NightAllNIROrderList, NightAllNIROrderWeightsList)

##################

Nights618And528AllWeigtnedAverage = CalculateWeightedAverage(Nights618And528OrderList, Nights618And528OrderWeightsList)

Nights618VISAnd528VISandNIRWeigtnedAverage = CalculateWeightedAverage(Nights618VISAnd528VISandNIROrderList, Nights618VISAnd528VISandNIROrderWeightsList)

Nights618and528OnlyVISWeigtnedAverage = CalculateWeightedAverage(Nights618and528OnlyVISOrderList, Nights618and528OnlyVISOrderWeightsList)

Nights618and528OnlyNIRWeigtnedAverage = CalculateWeightedAverage(Nights618And528OnlyNIROrderList, Nights618And528OnlyNIROrderWeightsList)

####################################

#np.save('TiAllNightsOutputForDebug.npy',NightAllWeigtnedAverage)

RadVelVect = NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[2]
NumRadVelPoints = len(RadVelVect)    
MiddleRadVelIndex = int(NumRadVelPoints/2)

SignalRegionIndices = ((MiddleRadVelIndex-25,MiddleRadVelIndex-15))



KpVect = NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[1]

### Set best Kp and Vsys 

### Standard values
RealKp = 241 ### 
BestSys = -20    


## For TiO, update the values 
if ModelForXCor == 'KELT9b_TiO_48_Exomol_McKemmish_0.50_+0.0_0.55_Vrot6.63':
    RealKp = 248 ### 
    BestSys = -15   
    
if ModelForXCor == 'KELT9b_Ca_0.50_+0.0_0.55_Vrot6.63':
    RealKp = 240 ### 
    BestSys = -23   
    #BestSys = -20
    
if ModelForXCor == 'KELT9b_CaII_0.50_+0.0_0.55_Vrot6.63':
    RealKp = 241 ### 
    BestSys = -15   

if ModelForXCor == 'KELT9b_Cr_0.50_+0.0_0.55_Vrot6.63':
    RealKp = 243 ### 
    BestSys = -17   

    


ClosestKpRow = np.argmin(np.abs(KpVect-RealKp)) 
ClosestVsysCol = np.argmin(np.abs(BestSys-RadVelVect))

HalfVsysRange = 250
#HalfVsysRange = 500

RadVPlotLimits = (-HalfVsysRange,HalfVsysRange)

#ImShowXLimits = MiddleRadVelIndex + np.array([-200,200])
ImShowXLimits = MiddleRadVelIndex + np.array([-HalfVsysRange,HalfVsysRange])

KpVsysXaxis = RadVelVect[ImShowXLimits[0]:ImShowXLimits[1]]


BestKpXlims = (-HalfVsysRange,HalfVsysRange)

### ***

#CCFsFromShuffles = np.load('CalculatedConfidenceIntervals/ProcessedData/KELT9b_Ti_0.50_+0.0_0.55_Vrot6.63/NumShuffles1000/Run0/TargetNoModel_InjScalingFactor0.000e+00_ModelKELT9b_Ti_0.50_+0.0_0.55_Vrot6.63_Sysrem6_AllShuffleResults_1000.npy')
#CCFsFromShuffles = np.load('CalculatedConfidenceIntervals/ProcessedData/KELT9b_Al_0.50_+0.0_0.55_Vrot6.63/NumShuffles1000/Run0/TargetNoModel_InjScalingFactor0.000e+00_ModelKELT9b_Al_0.50_+0.0_0.55_Vrot6.63_Sysrem6_AllShuffleResults_1000.npy')
#CCFsFromShuffles = np.load('CalculatedConfidenceIntervals/ProcessedData/KELT9b_FeH_main_iso_0.50_+0.0_0.55_Vrot6.63/NumShuffles1000/Run0/TargetNoModel_InjScalingFactor0.000e+00_ModelKELT9b_FeH_main_iso_0.50_+0.0_0.55_Vrot6.63_Sysrem6_AllShuffleResults_1000.npy')CCFsFromShuffles = np.load('CalculatedConfidenceIntervals/ProcessedData/KELT9b_FeH_main_iso_0.50_+0.0_0.55_Vrot6.63/NumShuffles1000/Run0/TargetNoModel_InjScalingFactor0.000e+00_ModelKELT9b_FeH_main_iso_0.50_+0.0_0.55_Vrot6.63_Sysrem6_AllShuffleResults_1000.npy')
#CCFsFromShuffles = np.load('CalculatedConfidenceIntervals/ProcessedData/KELT9b_Ca_0.50_+0.0_0.55_Vrot6.63/NumShuffles1000/Run0/TargetNoModel_InjScalingFactor0.000e+00_ModelKELT9b_Ca_0.50_+0.0_0.55_Vrot6.63_Sysrem6_AllShuffleResults_1000.npy')
#CCFsFromShuffles = np.load('CalculatedConfidenceIntervals/ProcessedData/KELT9b_Na_0.50_+0.0_0.55_Vrot6.63/NumShuffles1000/Run0/TargetNoModel_InjScalingFactor0.000e+00_ModelKELT9b_Na_0.50_+0.0_0.55_Vrot6.63_Sysrem6_AllShuffleResults_1000.npy')
#CCFsFromShuffles = np.load('CalculatedConfidenceIntervals/ProcessedData/KELT9b_TiO_all_iso_Plez_0.50_+0.0_0.55_Vrot6.63/NumShuffles1000/Run0/TargetNoModel_InjScalingFactor0.000e+00_ModelKELT9b_TiO_all_iso_Plez_0.50_+0.0_0.55_Vrot6.63_Sysrem6_AllShuffleResults_1000.npy')
#CCFsFromShuffles = np.load('CalculatedConfidenceIntervals/ProcessedData/KELT9b_Cr_0.50_+0.0_0.55_Vrot6.63/NumShuffles1000/Run0/TargetNoModel_InjScalingFactor0.000e+00_ModelKELT9b_Cr_0.50_+0.0_0.55_Vrot6.63_Sysrem6_AllShuffleResults_1000.npy')
#CCFsFromShuffles = np.load('CalculatedConfidenceIntervals/ProcessedData/KELT9b_CaII_0.50_+0.0_0.55_Vrot6.63/NumShuffles1000/Run0/TargetNoModel_InjScalingFactor0.000e+00_ModelKELT9b_CaII_0.50_+0.0_0.55_Vrot6.63_Sysrem6_AllShuffleResults_1000.npy')

#### This TiO_all_iso confidence intervals were with all nir weights set to 0
#CCFsFromShuffles = np.load('CalculatedConfidenceIntervals/ProcessedData/KELT9b_TiO_all_iso_Plez_0.50_+0.0_0.55_Vrot6.63/NumShuffles1000/Run1/TargetNoModel_InjScalingFactor0.000e+00_ModelKELT9b_TiO_all_iso_Plez_0.50_+0.0_0.55_Vrot6.63_Sysrem6_AllShuffleResults_1000.npy')
#CCFsFromShuffles = np.load('CalculatedConfidenceIntervals/ProcessedData/KELT9b_TiO_48_Exomol_McKemmish_0.50_+0.0_0.55_Vrot6.63/NumShuffles1000/Run1/TargetNoModel_InjScalingFactor0.000e+00_ModelKELT9b_TiO_48_Exomol_McKemmish_0.50_+0.0_0.55_Vrot6.63_Sysrem6_AllShuffleResults_1000.npy')

#CCFsFromShuffles = np.load('CalculatedConfidenceIntervals/ProcessedData/KELT9b_Mg_0.50_+0.0_0.55_Vrot6.63/NumShuffles1000/Run0/TargetNoModel_InjScalingFactor0.000e+00_ModelKELT9b_Mg_0.50_+0.0_0.55_Vrot6.63_Sysrem6_AllShuffleResults_1000.npy')

if PlotConfidenceIntervalString == 'WithConfInt':  
    
    NumRandom = 1000
    RunNum = 0    

    # NoModel_20180609All_vis_A_ConfInt = LoadOrdersForConfidenceInterval(NumRandom,RunNum,'NoModel','20180609All','vis','A',0,Night20180609AllVisSysremIts,ModelForXCor,InjectionScalingFactorFor20180609All,Night20180609AllVisSysremItsForWeights,VisOverrideWeights)
    # NoModel_20180609All_nir_A_ConfInt = LoadOrdersForConfidenceInterval(NumRandom,RunNum,'NoModel','20180609All','nir','A',0,Night20180609AllNirASysremIts,ModelForXCor,InjectionScalingFactorFor20180609All,Night20180609AllNirASysremItsForWeights,NirAOverrideWeights)
    
    
    NoModel_20180618All_vis_A_ConfInt = LoadOrdersForConfidenceInterval(NumRandom,RunNum,'NoModel','20180618All','vis','A',0,Night20180618AllVisSysremIts,ModelForXCor,InjectionScalingFactorFor20180618All,Night20180618AllVisSysremItsForWeights,VisOverrideWeights)
    NoModel_20180618All_nir_A_ConfInt = LoadOrdersForConfidenceInterval(NumRandom,RunNum,'NoModel','20180618All','nir','A',0,Night20180618AllNirASysremIts,ModelForXCor,InjectionScalingFactorFor20180618All,Night20180618AllNirASysremItsForWeights,NirAOverrideWeights)
        
    
    NoModel_20190528All_vis_A_ConfInt = LoadOrdersForConfidenceInterval(NumRandom,RunNum,'NoModel','20190528P2','vis','A',0,Night20190528AllVisSysremIts,ModelForXCor,InjectionScalingFactorFor20190528All,Night20190528AllVisSysremItsForWeights,VisOverrideWeights)
    NoModel_20190528All_nir_A_ConfInt = LoadOrdersForConfidenceInterval(NumRandom,RunNum,'NoModel','20190528P2','nir','A',0,Night20190528AllNirASysremIts,ModelForXCor,InjectionScalingFactorFor20190528All,Night20190528AllNirASysremItsForWeights,NirAOverrideWeights)
    
    
    # NoModel_20190604All_vis_A_ConfInt = LoadOrdersForConfidenceInterval(NumRandom,RunNum,'NoModel','20190604All','vis','A',0,Night20190604AllVisSysremIts,ModelForXCor,InjectionScalingFactorFor20190604All,Night20190604AllVisSysremItsForWeights,VisOverrideWeights)
    # NoModel_20190604All_nir_A_ConfInt = LoadOrdersForConfidenceInterval(NumRandom,RunNum,'NoModel','20190604All','nir','A',0,Night20190604AllNirASysremIts,ModelForXCor,InjectionScalingFactorFor20190604All,Night20190604AllNirASysremItsForWeights,NirAOverrideWeights)


    Nights618And528OrderListConfInt = [NoModel_20180618All_vis_A_ConfInt,
                                NoModel_20180618All_nir_A_ConfInt,                                                      
                                NoModel_20190528All_vis_A_ConfInt,
                                NoModel_20190528All_nir_A_ConfInt]
    Nights618And528OrderWeightsListConfInt = Nights618And528OrderWeightsList
    
    
    ### For TiO that only use 618 and 528 VIS arm 
    # Nights618And528OrderListConfInt = [NoModel_20180618All_vis_A_ConfInt,        
    #                                   NoModel_20190528All_vis_A_ConfInt]
    # Nights618And528OrderWeightsListConfInt = Nights618and528OnlyVISOrderWeightsList

    
    
    # Nights618And528OrderListConfInt = [NoModel_20190528All_vis_A_ConfInt]    
    # Nights618And528OrderWeightsListConfInt = Night528VISOrderWeightsList
    
    
    Nights618And528AllWeigtnedAverageConfInt = CalculateWeightedAverage(Nights618And528OrderListConfInt, Nights618And528OrderWeightsListConfInt)                        
    

    
    
    # NumShuffles = 125

    
    # ### This %d will need to be updated to %.2e for all others 
    # ConfIntLoadPath = 'CalculatedConfidenceIntervals/%s/%s/%s_%d/%s/%s/NumShuffles%d'%(DataProcessingVersion,
    #                                                                                    SysremItDescriptor,
    #                                                                                    TargetForCrossCor,
    #                                                                                    InjectedSignalScalingFactor,
    #                                                                                    ModelForXCor,
    #                                                                                    InjStrForWeights,
    #                                                                                    NumShuffles)
    
    
    # CCFsFromShuffles0 = np.load('%s/Results_of_%d_shuffles_run_0.npy'%(ConfIntLoadPath,NumShuffles))
    # CCFsFromShuffles1 = np.load('%s/Results_of_%d_shuffles_run_1.npy'%(ConfIntLoadPath,NumShuffles))
    # CCFsFromShuffles2 = np.load('%s/Results_of_%d_shuffles_run_2.npy'%(ConfIntLoadPath,NumShuffles))
    # CCFsFromShuffles3 = np.load('%s/Results_of_%d_shuffles_run_3.npy'%(ConfIntLoadPath,NumShuffles))
    # CCFsFromShuffles4 = np.load('%s/Results_of_%d_shuffles_run_4.npy'%(ConfIntLoadPath,NumShuffles))
    # CCFsFromShuffles5 = np.load('%s/Results_of_%d_shuffles_run_5.npy'%(ConfIntLoadPath,NumShuffles))
    # CCFsFromShuffles6 = np.load('%s/Results_of_%d_shuffles_run_6.npy'%(ConfIntLoadPath,NumShuffles))
    # CCFsFromShuffles7 = np.load('%s/Results_of_%d_shuffles_run_7.npy'%(ConfIntLoadPath,NumShuffles))

    # CalculatedConfidenceIntervals/ProcessedData/SysremConfig4/NoModel_0/KELT9b_TiO_all_iso_Plez_0.50_+0.0_0.55_Vrot6.63/750_195_150_750/NumShuffles125/Results_of_125_shuffles_run_0.npy

    # CCFsFromShuffles0 = np.load('CalculatedConfidenceIntervals/SysremConfig1/ProcessedData/%s/NumShuffles125/Run0/TargetNoModel_InjScalingFactor0.000e+00_Model%s_AllShuffleResults_125.npy'%(ModelForXCor,ModelForXCor))
    # CCFsFromShuffles1 = np.load('CalculatedConfidenceIntervals/SysremConfig1/ProcessedData/%s/NumShuffles125/Run1/TargetNoModel_InjScalingFactor0.000e+00_Model%s_AllShuffleResults_125.npy'%(ModelForXCor,ModelForXCor))
    # CCFsFromShuffles2 = np.load('CalculatedConfidenceIntervals/SysremConfig1/ProcessedData/%s/NumShuffles125/Run2/TargetNoModel_InjScalingFactor0.000e+00_Model%s_AllShuffleResults_125.npy'%(ModelForXCor,ModelForXCor))
    # CCFsFromShuffles3 = np.load('CalculatedConfidenceIntervals/SysremConfig1/ProcessedData/%s/NumShuffles125/Run3/TargetNoModel_InjScalingFactor0.000e+00_Model%s_AllShuffleResults_125.npy'%(ModelForXCor,ModelForXCor))
    # CCFsFromShuffles4 = np.load('CalculatedConfidenceIntervals/SysremConfig1/ProcessedData/%s/NumShuffles125/Run4/TargetNoModel_InjScalingFactor0.000e+00_Model%s_AllShuffleResults_125.npy'%(ModelForXCor,ModelForXCor))
    # CCFsFromShuffles5 = np.load('CalculatedConfidenceIntervals/SysremConfig1/ProcessedData/%s/NumShuffles125/Run5/TargetNoModel_InjScalingFactor0.000e+00_Model%s_AllShuffleResults_125.npy'%(ModelForXCor,ModelForXCor))
    # CCFsFromShuffles6 = np.load('CalculatedConfidenceIntervals/SysremConfig1/ProcessedData/%s/NumShuffles125/Run6/TargetNoModel_InjScalingFactor0.000e+00_Model%s_AllShuffleResults_125.npy'%(ModelForXCor,ModelForXCor))
    # CCFsFromShuffles7 = np.load('CalculatedConfidenceIntervals/SysremConfig1/ProcessedData/%s/NumShuffles125/Run7/TargetNoModel_InjScalingFactor0.000e+00_Model%s_AllShuffleResults_125.npy'%(ModelForXCor,ModelForXCor))
    
    # CCFsFromShuffles = np.vstack((CCFsFromShuffles0,
    #                               CCFsFromShuffles1,
    #                               CCFsFromShuffles2,
    #                               CCFsFromShuffles3,
    #                               CCFsFromShuffles4,
    #                               CCFsFromShuffles5,
    #                               CCFsFromShuffles6,
    #                               CCFsFromShuffles7))


    StdConfInt = np.std(Nights618And528AllWeigtnedAverageConfInt,axis=0)
    StdConfIntToPlot = StdConfInt[ImShowXLimits[0]:ImShowXLimits[1]]
    
    
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # print('Nights618And528AllWeigtnedAverageConfInt')
    # print(Nights618And528AllWeigtnedAverageConfInt)
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # raise Exception

        
if InjectedSignalScalingFactor == 0:
    # MakePlots(Night609WeigtnedAverage,ModelForXCor,'Night_20180609All_%s'%(ModelForXCor),SaveLoadable=True)
    # MakePlots(Night609VISWeigtnedAverage,ModelForXCor,'Night_20180609VIS_%s'%(ModelForXCor),SaveLoadable=True)
    # MakePlots(Night609NIRWeigtnedAverage,ModelForXCor,'Night_20180609NIR_%s'%(ModelForXCor),SaveLoadable=True)
    
    MakePlots(Night618WeigtnedAverage,ModelForXCor,'Night_20180618All_%s'%(ModelForXCor),SaveLoadable=True)
    MakePlots(Night618VISWeigtnedAverage,ModelForXCor,'Night_20180618VIS_%s'%(ModelForXCor),SaveLoadable=True)
    MakePlots(Night618NIRWeigtnedAverage,ModelForXCor,'Night_20180618NIR_%s'%(ModelForXCor),SaveLoadable=True)    
    
    MakePlots(Night528WeigtnedAverage,ModelForXCor,'Night_20190528All_%s'%(ModelForXCor),SaveLoadable=True)
    MakePlots(Night528VISWeigtnedAverage,ModelForXCor,'Night_20190528VIS_%s'%(ModelForXCor),SaveLoadable=True)
    MakePlots(Night528NIRWeigtnedAverage,ModelForXCor,'Night_20190528NIR_%s'%(ModelForXCor),SaveLoadable=True)    
    
    # MakePlots(Night604WeigtnedAverage,ModelForXCor,'Night_20190604All_%s'%(ModelForXCor),SaveLoadable=True)
    # MakePlots(Night604VISWeigtnedAverage,ModelForXCor,'Night_20190604VIS_%s'%(ModelForXCor),SaveLoadable=True)
    # MakePlots(Night604NIRWeigtnedAverage,ModelForXCor,'Night_20190604NIR_%s'%(ModelForXCor),SaveLoadable=True)    
    
    # MakePlots(NightAllWeigtnedAverage,ModelForXCor,'AllNights_%s'%(ModelForXCor),SaveLoadable=True)    
    # MakePlots(NightAllVISWeigtnedAverage,ModelForXCor,'AllNightsVIS_%s'%(ModelForXCor),SaveLoadable=True) 
    # MakePlots(NightAllNIRWeigtnedAverage,ModelForXCor,'AllNightsNIR_%s'%(ModelForXCor),SaveLoadable=True) 
    
    MakePlots(Nights618And528AllWeigtnedAverage,ModelForXCor,'Nights_618And528_VisAndNIR_%s'%(ModelForXCor)) 
    
    MakePlots(Nights618VISAnd528VISandNIRWeigtnedAverage,ModelForXCor,'Nights_618VISAnd528NIRAndVIS_%s'%(ModelForXCor)) 
    
    MakePlots(Nights618and528OnlyVISWeigtnedAverage,ModelForXCor,'Nights_618And528OnlyVIS_%s'%(ModelForXCor)) 
    
    MakePlots(Nights618and528OnlyNIRWeigtnedAverage,ModelForXCor,'Nights_618And528OnlyNIR_%s'%(ModelForXCor)) 

    
else:
    # MakePlots(Night609WeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'Night_20180609All_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)
    # MakePlots(Night609VISWeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'Night_20180609VIS_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)
    # MakePlots(Night609NIRWeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'Night_20180609NIR_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)

    MakePlots(Night618WeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'Night_20180618All_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)
    MakePlots(Night618VISWeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'Night_20180618VIS_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)
    MakePlots(Night618NIRWeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'Night_20180618NIR_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)   
    
    MakePlots(Night528WeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'Night_20190528All_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)
    MakePlots(Night528VISWeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'Night_20190528VIS_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)
    MakePlots(Night528NIRWeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'Night_20190528NIR_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)    
    
    # MakePlots(Night604WeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'Night_20190604All_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)
    # MakePlots(Night604VISWeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'Night_20190604VIS_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)
    # MakePlots(Night604NIRWeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'Night_20190604NIR_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)    
    
    # MakePlots(NightAllWeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'AllNights_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)
    # MakePlots(NightAllVISWeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'AllNightsVIS_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)
    # MakePlots(NightAllNIRWeigtnedAverage,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),'AllNightsNIR_%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor),SaveLoadable=True)

    MakePlots(Nights618And528AllWeigtnedAverage,ModelForXCor,'Nights_618And528_VisAndNIR_%s'%(ModelForXCor)) 
    
    MakePlots(Nights618VISAnd528VISandNIRWeigtnedAverage,ModelForXCor,'Nights_618VISAnd528NIRAndVIS_%s'%(ModelForXCor)) 
    
    MakePlots(Nights618and528OnlyVISWeigtnedAverage,ModelForXCor,'Nights_618And528OnlyVIS_%s'%(ModelForXCor)) 
    
    MakePlots(Nights618and528OnlyNIRWeigtnedAverage,ModelForXCor,'Nights_618And528OnlyNIR_%s'%(ModelForXCor)) 



print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('Shape of Weighted average KpVsys')
print(Night528WeigtnedAverage.shape)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


### Works for the kp vector of 1 to 500
#KpRefIndex = 230

RadVelVectInterpObject = interpolate.interp1d(range(len(RadVelVect)),RadVelVect) 
VsysRefIndex = 970


if KpMode == 'MultiKp':

    KpRefIndex = 141
    KpVectInterpObject = interpolate.interp1d(range(len(KpVect)),KpVect) 
    
    
    
    
    
    
    
    SubArrayForCentroid = Nights618And528AllWeigtnedAverage[KpRefIndex:161,VsysRefIndex:990]
    
    
    ### For TiO
    #SubArrayForCentroid = Nights618and528OnlyVISWeigtnedAverage[KpRefIndex:161,VsysRefIndex:990]
    
    
    
    
    #SubArrayForCentroid = NightAllWeigtnedAverage
    
    plt.figure()
    plt.imshow(SubArrayForCentroid,aspect='auto',interpolation='None',origin='lower')
    plt.title('Subarray for centroid')
    plt.savefig('%s/TestSubarrayForOutput.pdf'%(SavePath))
    
    
    x1, y1 = centroid_com(SubArrayForCentroid)
    x2, y2 = centroid_1dg(SubArrayForCentroid)
    x3, y3 = centroid_2dg(SubArrayForCentroid)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(SubArrayForCentroid, origin='lower', interpolation='nearest')
    marker = '+'
    ms, mew = 15, 2.
    plt.plot(x1, y1, color='black',label='centroid_com', marker=marker, ms=ms, mew=mew)
    plt.plot(x2, y2, color='white',label='centroid_com', marker=marker, ms=ms, mew=mew)
    plt.plot(x3, y3, color='red', marker=marker, ms=ms, mew=mew)
    plt.savefig('%s/LocatedCentroids.pdf'%(SavePath))
    
    maxes = np.where(SubArrayForCentroid==np.max(SubArrayForCentroid))
    
    Kp1 = KpVectInterpObject(KpRefIndex+y1)
    Vsys1 = RadVelVectInterpObject(VsysRefIndex+x1)
    
    Kp2 = KpVectInterpObject(KpRefIndex+y2)
    Vsys2 = RadVelVectInterpObject(VsysRefIndex+x2)
    
    Kp3 = KpVectInterpObject(KpRefIndex+y3)
    Vsys3 = RadVelVectInterpObject(VsysRefIndex+x3)
    
    # Kp4 = KpVectInterpObject(maxes[1]+y3)
    # Vsys4 = RadVelVectInterpObject(maxes[0]+x3)
    
    # if InjectedSignalScalingFactor == 0:
    #     SavePath = 'plots/%s/CombinedPlots/%s/%s'%(DataProcessingVersion,SysremItDescriptor,ModelForXCor)
    
    # else: 
    #     SavePath = 'plots/%s/CombinedPlots/%s/%s'%(DataProcessingVersion,SysremItDescriptor,'%s_InjectStrength%.3e'%(ModelForXCor,InjectedSignalScalingFactor))
    
    
    
    
    print()
    print('Centroid fits:')
    print('centroid_com: Kp = %f, Vsys = %f'%(Kp1,Vsys1))
    print('centroid_1dg: Kp = %f, Vsys = %f'%(Kp2,Vsys2))
    print('centroid_2dg: Kp = %f, Vsys = %f'%(Kp3,Vsys3))
    # print('argmax: Kp = %f, Vsys = %f'%(Kp4,Vsys4))
    
    
    AllFitKps = np.array([Kp1,Kp2,Kp3])
    AllFitVsys = np.array([Vsys1,Vsys2,Vsys3])
    
    MeanKp = np.mean(AllFitKps)
    MeanVsys = np.mean(AllFitVsys)
    
    KpMaxSpread = np.ptp(AllFitKps)
    KpStd = np.std(AllFitKps)
    
    VsysMaxSpread = np.ptp(AllFitVsys)
    VsysStd = np.std(AllFitVsys)
    
    print('Mean: Kp = %f, Vsys = %f'%(MeanKp,MeanVsys)) 
    print('Max spread: Kp = %f, Vsys = %f'%(KpMaxSpread,VsysMaxSpread)) 
    print('Std: Kp = %f, Vsys = %f'%(KpStd,VsysStd)) 
    
    
print('Closest Kp row: %d'%(ClosestKpRow))


SignalRegionIndex=(975,985)
DetSig = MeasureDetectionSNR(Nights618And528AllWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)

#### All Det sigs 
###########################

DetSig_609All = MeasureDetectionSNR(Night609WeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)
DetSig_609VIS = MeasureDetectionSNR(Night609VISWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)
DetSig_609NIR = MeasureDetectionSNR(Night609NIRWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)

DetSig_618All = MeasureDetectionSNR(Night618WeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)
DetSig_618VIS = MeasureDetectionSNR(Night618VISWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)
DetSig_618NIR = MeasureDetectionSNR(Night618NIRWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)

DetSig_528All = MeasureDetectionSNR(Night528WeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)
DetSig_528VIS = MeasureDetectionSNR(Night528VISWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)
DetSig_528NIR = MeasureDetectionSNR(Night528NIRWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)

DetSig_604All = MeasureDetectionSNR(Night604WeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)
DetSig_604VIS = MeasureDetectionSNR(Night604VISWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)
DetSig_604NIR = MeasureDetectionSNR(Night604NIRWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)

# DetSig_AllAll = MeasureDetectionSNR(NightAllWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)
# DetSig_AllVIS = MeasureDetectionSNR(NightAllVISWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)
# DetSig_AllNIR = MeasureDetectionSNR(NightAllNIRWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)

DetSig_Nights618And528 = MeasureDetectionSNR(Nights618And528AllWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)

DetSig_Nights618VISAnd528VISandNIR = MeasureDetectionSNR(Nights618VISAnd528VISandNIRWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)

DetSig_Nights618And528OnlyVIS = MeasureDetectionSNR(Nights618and528OnlyVISWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)

DetSig_Nights618And528OnlyNIR = MeasureDetectionSNR(Nights618and528OnlyNIRWeigtnedAverage[ClosestKpRow,:],SignalRegionIndex = SignalRegionIndices)



############################

print()
print('Detection significance:')
print(DetSig)


f = open('%s/DetSigAndFits.txt'%(SavePath), 'w')
f.write('Total detection significance:\n')
f.write('%.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig[0],DetSig[1],DetSig[2]))
f.write('\n')







###########
###### Fit 1D Gaussians to the closest row and column

ccf = np.zeros((len(RadVelVect),2))
ccf[:,0] = RadVelVect
ccf[:,1] = Nights618And528AllWeigtnedAverage[ClosestKpRow,:]

### For TiO
#ccf[:,1] = Nights618and528OnlyVISWeigtnedAverage[ClosestKpRow,:]



PeakIndex = np.argmax(ccf[SignalRegionIndex[0]:SignalRegionIndex[1],1]) + SignalRegionIndex[0]

HalfWidthRegionToFit = 50
HalfKpWidthToFit = 50

ccf_ToFit = ccf[PeakIndex-HalfWidthRegionToFit:PeakIndex+HalfWidthRegionToFit,:]
RVRegionToFit = (PeakIndex-HalfWidthRegionToFit,PeakIndex+HalfWidthRegionToFit)

if KpMode == 'MultiKp':
    KpRegionToFit = (ClosestKpRow-HalfKpWidthToFit,ClosestKpRow+HalfKpWidthToFit)    
    
    KpVsys = Night528WeigtnedAverage
    
    ccfKp_ToFit = KpVsys[KpRegionToFit[0]:KpRegionToFit[1],ClosestVsysCol]
    
    
    KpVsys_ToFit = KpVsys[KpRegionToFit[0]:KpRegionToFit[1],RVRegionToFit[0]:RVRegionToFit[1]]
    KpVect_ToFit = KpVect[KpRegionToFit[0]:KpRegionToFit[1]]
    
    InitialMeanKp = np.mean(KpVect_ToFit)


RVVect_ToFit = RadVelVect[RVRegionToFit[0]:RVRegionToFit[1]] 


InitialMeanAmp = np.max(ccf_ToFit[:,1])
InitialMean = np.mean(ccf_ToFit[:,0])
InitialSTD = 10



g_init = models.Gaussian1D(amplitude=InitialMeanAmp, mean=InitialMean, stddev=InitialSTD)
fit_g = fitting.LevMarLSQFitter()
g = fit_g(g_init, ccf_ToFit[:,0], ccf_ToFit[:,1])

cov_diag = np.diag(fit_g.fit_info['param_cov'])

##############
### Trying scipy curve_fit 

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,ccf_ToFit[:,0], ccf_ToFit[:,1],p0=[InitialMeanAmp,InitialMean,InitialSTD])

pcov_diag = np.diag(pcov)

print('Scipy fits:')
print('popt')
print(popt)
print()
print('pcov_diag')
print(np.sqrt(pcov_diag))

print()
print('Astroypy')
print(np.sqrt(cov_diag))
print()
print()




#################




print('Gaussian fit to closest Kp row')
print('Amplitude: %.10e +\- %.10e'%(g.amplitude.value, np.sqrt(cov_diag[0])))
print('Mean: %.10f +\- %.10f'%(g.mean.value, np.sqrt(cov_diag[1])))
print('Standard Deviation: %.10f +\- %.10f'%(g.stddev.value, np.sqrt(cov_diag[2])))

raise Exception

f.write('Gaussian fits:\n')
f.write('Gaussian fit to closest Kp row (giving Vsys)\n')
f.write('Amplitude: %.10e +\- %.10e\n'%(g.amplitude.value, np.sqrt(cov_diag[0])))
f.write('Mean: %.10f +\- %.10f\n'%(g.mean.value, np.sqrt(cov_diag[1])))
f.write('Standard Deviation: %.10f +\- %.10f\n'%(g.stddev.value, np.sqrt(cov_diag[2])))
f.write('\n')

#######################
if KpMode == 'MultiKp':

    g_initKp = models.Gaussian1D(amplitude=InitialMeanAmp, mean=InitialMeanKp, stddev=InitialSTD)
    fit_gKp = fitting.LevMarLSQFitter()
    gKp = fit_gKp(g_initKp, KpVect_ToFit, ccfKp_ToFit)
    
    cov_diagKp = np.diag(fit_gKp.fit_info['param_cov'])
    print()
    print('Gaussian fit to closest Vsys column')
    print('Kp Amplitude: %.10e +\- %.10e'%(gKp.amplitude.value, np.sqrt(cov_diagKp[0])))
    print('Kp Mean: %.10f +\- %.10f'%(gKp.mean.value, np.sqrt(cov_diagKp[1])))
    print('Kp Standard Deviation: %.10f +\- %.10f'%(gKp.stddev.value, np.sqrt(cov_diagKp[2])))
    
    
    f.write('Gaussian fit to closest Vsys column (giving Kp)\n')
    f.write('Kp Amplitude: %.10e +\- %.10e\n'%(gKp.amplitude.value, np.sqrt(cov_diagKp[0])))
    f.write('Kp Mean: %.10f +\- %.10f\n'%(gKp.mean.value, np.sqrt(cov_diagKp[1])))
    f.write('Kp Standard Deviation: %.10f +\- %.10f\n'%(gKp.stddev.value, np.sqrt(cov_diagKp[2])))
    f.write('\n')

Gauss1DFitsPdf = PdfPages('%s/Gauss1DFits.pdf'%(SavePath)) 

plt.figure()
plt.plot(ccf_ToFit[:,0],ccf_ToFit[:,1])
plt.plot(ccf_ToFit[:,0],g(ccf_ToFit[:,0]),'r')
plt.title('Gaussian fit to closest Kp row')
plt.xlabel('Vsys (km/s)')
plt.ylabel('cross-correlation value')

Gauss1DFitsPdf.savefig()

if KpMode == 'MultiKp':
    plt.figure()
    plt.plot(KpVect_ToFit,ccfKp_ToFit)
    plt.plot(KpVect_ToFit,gKp(KpVect_ToFit),'r')
    plt.title('Gaussian fit to closest Vsys column')
    plt.xlabel('Kp (km/s)')
    plt.ylabel('cross-correlation value')
    
    Gauss1DFitsPdf.savefig()

Gauss1DFitsPdf.close()

f.write('\n')
f.write('Detection significances of individual arms and nights:\n')

# f.write('20180609All VIS and NIR: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_609All[0],DetSig_609All[1],DetSig_609All[2]))
# f.write('20180609All VIS: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_609VIS[0],DetSig_609VIS[1],DetSig_609VIS[2]))
# f.write('20180609All NIR: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_609NIR[0],DetSig_609NIR[1],DetSig_609NIR[2]))
# f.write('\n')

f.write('20180618All VIS and NIR: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_618All[0],DetSig_618All[1],DetSig_618All[2]))
f.write('20180618All VIS: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_618VIS[0],DetSig_618VIS[1],DetSig_618VIS[2]))
f.write('20180618All NIR: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_618NIR[0],DetSig_618NIR[1],DetSig_618NIR[2]))
f.write('\n')

f.write('20190528All VIS and NIR: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_528All[0],DetSig_528All[1],DetSig_528All[2]))
f.write('20190528All VIS: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_528VIS[0],DetSig_528VIS[1],DetSig_528VIS[2]))
f.write('20190528All NIR: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_528NIR[0],DetSig_528NIR[1],DetSig_528NIR[2]))
f.write('\n')

# f.write('20190604All VIS and NIR: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_604All[0],DetSig_604All[1],DetSig_604All[2]))
# f.write('20190604All VIS: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_604VIS[0],DetSig_604VIS[1],DetSig_604VIS[2]))
# f.write('20190604All NIR: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_604NIR[0],DetSig_604NIR[1],DetSig_604NIR[2]))
# f.write('\n')

# f.write('All nights VIS and NIR: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_AllAll[0],DetSig_AllAll[1],DetSig_AllAll[2]))
# f.write('All nights VIS: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_AllVIS[0],DetSig_AllVIS[1],DetSig_AllVIS[2]))
# f.write('All nights NIR: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_AllNIR[0],DetSig_AllNIR[1],DetSig_AllNIR[2]))
# f.write('\n')

f.write('618 and 528 VIS and NIR: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_Nights618And528[0],DetSig_Nights618And528[1],DetSig_Nights618And528[2]))
f.write('\n')

f.write('618 VIS and 528 VIS and NIR: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_Nights618VISAnd528VISandNIR[0],DetSig_Nights618VISAnd528VISandNIR[1],DetSig_Nights618VISAnd528VISandNIR[2]))
f.write('\n')

f.write('618 and 528 Only VIS: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_Nights618And528OnlyVIS[0],DetSig_Nights618And528OnlyVIS[1],DetSig_Nights618And528OnlyVIS[2]))
f.write('\n')

f.write('618 and 528 Only NIR: %.3e sigma (peak %.3e, noise %.3e)\n'%(DetSig_Nights618And528OnlyNIR[0],DetSig_Nights618And528OnlyNIR[1],DetSig_Nights618And528OnlyNIR[2]))
f.write('\n')


if KpMode == 'MultiKp':
    f.write('\n')
    f.write('1D plots using Kp index %d, corresponding to %.2f km/s\n'%(ClosestKpRow,KpVect[ClosestKpRow]))
    f.write('1D plots using Vsys index %d, corresponding to %.2f km/s\n'%(ClosestVsysCol,RadVelVect[ClosestVsysCol]))

    f.write('\n')
    f.write('Centroid fits to all combined data:\n')
    f.write('centroid_com: Kp = %f, Vsys = %f\n'%(Kp1,Vsys1))
    f.write('centroid_1dg: Kp = %f, Vsys = %f\n'%(Kp2,Vsys2))
    f.write('centroid_2dg: Kp = %f, Vsys = %f\n'%(Kp3,Vsys3))
    # f.write('argmax: Kp = %f, Vsys = %f\n'%(Kp4,Vsys4))

    f.write('Mean: Kp = %f, Vsys = %f\n'%(MeanKp,MeanVsys))
    f.write('Max spread: Kp = %f, Vsys = %f\n'%(KpMaxSpread,VsysMaxSpread))
    f.write('Std: Kp = %f, Vsys = %f\n'%(KpStd,VsysStd))
    




# RadVelsForCCFs528VIS, AllCCFPerOrderArray528VIS = LoadAllCCFs('20190528All',TargetForCrossCor,InjectedSignalScalingFactor,'vis','A',Night20190528AllVisSysremIts,ModelForXCor)
# OrderWeightArray = NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[3]
# AvgCCFs = np.average(AllCCFPerOrderArray528VIS,axis=2,weights=OrderWeightArray)

# with PdfPages('%s/Night528Vis_AverageCCF.pdf'%(SavePath)) as CCFpdf:

##############################################
### Make plots of the 2D CCFs 
phase528 = np.loadtxt('../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/ProcessedData/NoModel/%s/%s/phase.txt'%('20190528P2','vis'))
phase618 = np.loadtxt('../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/ProcessedData/NoModel/%s/%s/phase.txt'%('20180618All','vis'))

#rv528 = np.loadtxt('../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/ProcessedData/NoModel/%s/%s/radv.txt'%('20190528P2','vis'))
#bary528 = np.loadtxt('../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/ProcessedData/NoModel/%s/%s/BarycentricRVcorrection_kms.txt'%('20190528P2','vis'))
netrv528 = np.loadtxt('../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/ProcessedData/NoModel/%s/%s/NettPlanetRV.txt'%('20190528P2','vis'))
netrv618 = np.loadtxt('../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/ProcessedData/NoModel/%s/%s/NettPlanetRV.txt'%('20180618All','vis'))



RadVelsForCCFs528VIS, AllCCFPerOrderArray528VIS = LoadAllCCFs('20190528P2',TargetForCrossCor,InjectedSignalScalingFactor,'vis','A',Night20190528AllVisSysremIts,ModelForXCor)
RadVelsForCCFs528NIRA, AllCCFPerOrderArray528NIRA = LoadAllCCFs('20190528P2',TargetForCrossCor,InjectedSignalScalingFactor,'nir','A',Night20190528AllNirASysremIts,ModelForXCor)
AllCCFPerOrderArray528 = np.dstack((AllCCFPerOrderArray528VIS,AllCCFPerOrderArray528NIRA))
OrderWeightArray528 = np.hstack([NoModel_20190528All_vis_A_0_0_9_KELT9b_FeI_10_9[3], NoModel_20190528All_nir_A_0_0_9_KELT9b_FeI_10_9[3]])


RadVelsForCCFs618VIS, AllCCFPerOrderArray618VIS = LoadAllCCFs('20180618All',TargetForCrossCor,InjectedSignalScalingFactor,'vis','A',Night20180618AllVisSysremIts,ModelForXCor)
RadVelsForCCFs618NIRA, AllCCFPerOrderArray618NIRA = LoadAllCCFs('20180618All',TargetForCrossCor,InjectedSignalScalingFactor,'nir','A',Night20180618AllNirASysremIts,ModelForXCor)
AllCCFPerOrderArray618 = np.dstack((AllCCFPerOrderArray618VIS,AllCCFPerOrderArray618NIRA))
OrderWeightArray618 = np.hstack([NoModel_20180618All_vis_A_0_0_9_KELT9b_FeI_10_9[3], NoModel_20180618All_nir_A_0_0_9_KELT9b_FeI_10_9[3]])


OrderWeightsForTextOutput = np.zeros((len(OrderWeightArray618),2))
OrderWeightsForTextOutput[:,0] = OrderWeightArray618
OrderWeightsForTextOutput[:,1] = OrderWeightArray528
##np.savetxt('%s/%s_OrderWeights618_and_528.txt'%(SavePath,ModelForXCor),OrderWeightsForTextOutput)

ExcludedOrders = []
for i in range(len(OrderWeightArray528)):
    if OrderWeightArray528[i] == 0:
        ExcludedOrders.append(i+1)

ExcludedOrdersHeader = '### These orders are +1 (starting at order 1 instead of zero). VIS orders 1-61. NIR orders 62 - 89'
np.savetxt('%s/Excluded_orders_528_%s.txt'%(SavePath,ModelForXCor),np.array(ExcludedOrders),header=ExcludedOrdersHeader)



StringToWrite = ExcludedOrdersHeader+'\n'

for i in ExcludedOrders:
    StringToWrite += '%d, '%(int(i))


ExcludeOrdersTextFile = open('%s/Excluded_orders_528_text_file_%s.txt'%(SavePath,ModelForXCor), 'w')
ExcludeOrdersTextFile.write(StringToWrite)
ExcludeOrdersTextFile.close()

print('AllCCFPerOrderArray528')
print(AllCCFPerOrderArray528.shape)


print('OrderWeightArray528 shape')
print(OrderWeightArray528.shape)



AvgCCFs528 = np.average(AllCCFPerOrderArray528,axis=2,weights=OrderWeightArray528)
AvgCCFs618 = np.average(AllCCFPerOrderArray618,axis=2,weights=OrderWeightArray618)





#AvgCCFs = np.sum(AllCCFPerOrderArray528,axis=2)



with PdfPages('%s/AverageCCF_Night528VisAndNir.pdf'%(SavePath)) as CCFpdf:
    
    # NumSpecForCCF,NumColForCCF = np.shape(AvgCCFs528)
    
    # CCFPlotExtent = [RadVelsForCCFs528VIS[0],RadVelsForCCFs528VIS[-1],0,NumSpecForCCF]
    
    # mad = median_absolute_deviation(AvgCCFs528)
    # median = np.median(AvgCCFs528)
    # NumMADs = 3
    
    plt.figure()
    #plt.imshow(AvgCCFs,aspect='auto',origin='lower',interpolation='none',extent=CCFPlotExtent,vmin=median-NumMADs*mad,vmax=median+NumMADs*mad)
    #plt.imshow(AvgCCFs,aspect='auto',origin='lower',interpolation='none',extent=CCFPlotExtent)
    
    #plt.imshow(AvgCCFs,aspect='auto',origin='lower',interpolation='none', vmin=median-NumMADs*mad,vmax=median+NumMADs*mad)
    plt.imshow(AvgCCFs528,aspect='auto',origin='lower',interpolation='none')

    

    ax = plt.gca()
    
    ax.set_xticks(np.arange(len(RadVelsForCCFs528VIS))[::200])
    ax.set_xticklabels([int(x) for x in list(RadVelsForCCFs528VIS[::200])])
    
    ax.set_yticks(np.arange(len(phase528))[::5])
    ax.set_yticklabels(['%.2f'%(x) for x in list(phase528[::5])])    
    

    

  
    plt.colorbar()

    plt.xlabel('radial velocity (km/s)')
    plt.ylabel('Spectrum number')       

    NetRVShiftToPixel528 = netrv528 - np.min(RadVelsForCCFs528VIS)    
    PlanetGuideLineOffset = 50    
    plt.plot(NetRVShiftToPixel528-PlanetGuideLineOffset,np.arange(len(NetRVShiftToPixel528)),'w:')
    plt.plot(NetRVShiftToPixel528+PlanetGuideLineOffset,np.arange(len(NetRVShiftToPixel528)),'w:')

    
    
    
    CCFpdf.savefig()
    plt.close()
  
    
    plt.figure()
    #plt.imshow(AvgCCFs,aspect='auto',origin='lower',interpolation='none',extent=CCFPlotExtent,vmin=median-NumMADs*mad,vmax=median+NumMADs*mad)
    #plt.imshow(AvgCCFs,aspect='auto',origin='lower',interpolation='none',extent=CCFPlotExtent)
    
    #plt.imshow(AvgCCFs,aspect='auto',origin='lower',interpolation='none', vmin=median-NumMADs*mad,vmax=median+NumMADs*mad)
    plt.imshow(AvgCCFs618,aspect='auto',origin='lower',interpolation='none')

    ax = plt.gca()
    
    ax.set_xticks(np.arange(len(RadVelsForCCFs618VIS))[::200])
    ax.set_xticklabels([int(x) for x in list(RadVelsForCCFs618VIS[::200])])
    
    ax.set_yticks(np.arange(len(phase618))[::5])
    ax.set_yticklabels(['%.2f'%(x) for x in list(phase618[::5])])        
    plt.colorbar()

    plt.xlabel('radial velocity (km/s)')
    plt.ylabel('Spectrum number')       

    NetRVShiftToPixel618 = netrv618 - np.min(RadVelsForCCFs618VIS)    
    PlanetGuideLineOffset = 50    
    plt.plot(NetRVShiftToPixel618-PlanetGuideLineOffset,np.arange(len(NetRVShiftToPixel618)),'w:')
    plt.plot(NetRVShiftToPixel618+PlanetGuideLineOffset,np.arange(len(NetRVShiftToPixel618)),'w:')

    CCFpdf.savefig()
    plt.close()
    
################ subplots 
    
plt.figure()
plt.subplot(1,2,1)
plt.imshow(AvgCCFs528*1e3,aspect='auto',origin='lower',interpolation='none')
ax = plt.gca()   
ax.set_xticks(np.arange(len(RadVelsForCCFs528VIS))[::100])
ax.set_xticklabels([int(x) for x in list(RadVelsForCCFs528VIS[::100])])
ax.set_yticks(np.arange(len(phase528))[::5])
ax.set_yticklabels(['%.2f'%(x) for x in list(phase528[::5])])    
plt.colorbar()
plt.xlabel('radial velocity (km/s)')
plt.ylabel('phase')       
NetRVShiftToPixel528 = netrv528 - np.min(RadVelsForCCFs528VIS)    
PlanetGuideLineOffset = 50    
plt.plot(NetRVShiftToPixel528-PlanetGuideLineOffset,np.arange(len(NetRVShiftToPixel528)),'w:')
plt.plot(NetRVShiftToPixel528+PlanetGuideLineOffset,np.arange(len(NetRVShiftToPixel528)),'w:')
plt.xlim((600,1000))
plt.title('2019-05-28')  

plt.subplot(1,2,2)
plt.imshow(AvgCCFs618*1e3,aspect='auto',origin='lower',interpolation='none')
ax = plt.gca()    
ax.set_xticks(np.arange(len(RadVelsForCCFs618VIS))[::100])
ax.set_xticklabels([int(x) for x in list(RadVelsForCCFs618VIS[::100])])    
ax.set_yticks(np.arange(len(phase618))[::4])
ax.set_yticklabels(['%.2f'%(x) for x in list(phase618[::4])])        
plt.colorbar()
plt.xlabel('radial velocity (km/s)')
#plt.ylabel('phase')       
NetRVShiftToPixel618 = netrv618 - np.min(RadVelsForCCFs618VIS)    
PlanetGuideLineOffset = 50    
plt.plot(NetRVShiftToPixel618-PlanetGuideLineOffset,np.arange(len(NetRVShiftToPixel618)),'w:')
plt.plot(NetRVShiftToPixel618+PlanetGuideLineOffset,np.arange(len(NetRVShiftToPixel618)),'w:')
plt.xlim((950,1350))
plt.title('2018-06-18')

plt.subplots_adjust(wspace=0.35)

plt.savefig('%s/CCFs_%s.pdf'%(SavePath,ModelForXCor))
plt.close()

    
    
#     for OrderIndex in range(len(OrderWeightArray)):
        
#         plt.figure()
#         plt.imshow(AllCCFPerOrderArray528VIS[:,:,OrderIndex],aspect='auto',origin='lower',interpolation='none',extent=CCFPlotExtent)
#         plt.title('CCFs from order %d'%(OrderIndex))
#         plt.colorbar()
        
#         plt.xlabel('radial velocity (km/s)')
#         plt.ylabel('Spectrum number')    
        
#         CCFpdf.savefig()
#         plt.close()


#### End making plots of the 2D CCFs
###############################################


#################################################################
### plot the order weights in one graph where the order number is VIS, NIR (with NIRs as NIR A 0, NIR B 0, NIR A 1, NIR B 1 ect....)
        
AllOrderWeightsList528 = list(VIS_DetSigs[2,:])
AllOrderWeightsList618 = list(VIS_DetSigs[1,:])

for i in range(28):
    
    AllOrderWeightsList618.append(NIRA_DetSigs[1,i])
    AllOrderWeightsList618.append(NIRB_DetSigs[1,i])
    
    AllOrderWeightsList528.append(NIRA_DetSigs[2,i])
    AllOrderWeightsList528.append(NIRB_DetSigs[2,i])
        

    
with PdfPages('%s/Recovered_order_weights_in_one_plot.pdf'%(SavePath)) as OrderWeightsPDFInOnePlot:

    
    plt.figure()
    plt.plot(AllOrderWeightsList618)
    plt.ylabel('order weight')
    plt.xlabel('Order (vis then nir (Nir A 0, Nir B 0, ...))')
    plt.title('Night 618')
    OrderWeightsPDFInOnePlot.savefig()
        
    
    plt.figure()
    plt.plot(AllOrderWeightsList528)
    plt.ylabel('order weight')
    plt.xlabel('Order (vis then nir (Nir A 0, Nir B 0, ...))')
    plt.title('Night 528')
    OrderWeightsPDFInOnePlot.savefig()
        

sigToFWHM = 2*(2*np.log(2))**0.5

#StringForOverleafTable = '& $-$21.91 $\pm$ 0.34  & 5.49 $\pm$ 0.34 & 238.17 $\pm$ 0.32 & 10.58 $\pm$ 0.32 & 6'

print('')
print('For Overleaf table')

StringForOverleafTable = '& $%.2f$ $\pm$ $%.2f$ & $%.2f$ $\pm$ $%.2f$ & $%.2f$ $\pm$ $%.2f$ & $%.2f$ $\pm$ $%.2f$ & $%.1f$'%(g.mean.value, 
                                                                                                                              np.sqrt(cov_diag[1]),
                                                                                                                              g.stddev.value*sigToFWHM, 
                                                                                                                              np.sqrt(cov_diag[2])*sigToFWHM,
                                                                                                                              gKp.mean.value, 
                                                                                                                              np.sqrt(cov_diagKp[1]),
                                                                                                                              gKp.stddev.value*sigToFWHM, 
                                                                                                                              np.sqrt(cov_diagKp[2])*sigToFWHM,
                                                                                                                              DetSig[0])

print(StringForOverleafTable)

f.write('\n')
f.write('For Overleaf table:\n')
f.write(StringForOverleafTable)
f.close()




#################

# data = SubArrayForCentroid

# #med = np.median(data) 
# # data = data - med
# # data = data[0,0,:,:] # NOT SURE THIS IS NEEDED!

# fit_w = fitting.LevMarLSQFitter()

# y0, x0 = np.unravel_index(np.argmax(data), data.shape)
# sigma = np.std(data)
# amp = np.max(data)

# w = models.Gaussian2D(amp, x0, y0, None, None)

# yi, xi = np.indices(data.shape)

# g = fit_w(w, xi, yi, data)
# print(w)
# model_data = g(xi, yi)

# fig, ax = plt.subplots()
# #eps = np.min(model_data[model_data > 0]) / 10.0
# # use logarithmic scale for sharp Gaussians
# #ax.imshow(np.log(eps + model_data), label='Gaussian') 
# ax.imshow(model_data, label='Gaussian') 


# circle = plt.Circle((g.x_mean.value, g.y_mean.value),
#                 g.x_stddev.value, facecolor ='none',
#                 edgecolor = 'red', linewidth = 1)

# ax.add_patch(circle)
# #plt.show()


