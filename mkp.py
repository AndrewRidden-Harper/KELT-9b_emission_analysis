#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:15:30 2020

@author: ariddenharper
"""

import matplotlib as mpl
mpl.use('Agg')


import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
####from HD209458bJDToPhaseFunc import JDtophase_radial_vel   #### For HD209458b 
from KELT9bJDToPhaseFunc import JDtophase_radial_vel
from scipy.interpolate import interp1d
import os 
from MakePlanetSpecOverStarSpec import LambertSphere
import astropy.io.fits as pyfits
from scipy.signal import find_peaks 
import pandas as pd
import time

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

def MeasureDetectionSNR(ccfNoInj,ccf,SignalRegionIndex=(975,985),Adjust=False):
    
    NumberOfLeadingAndTrailingPointsToExclude = 500 ## Good for -1000 to 1000 in steps of 1 
    
    #MiddleIndex = int(len(ccf)/2)
    
    maxindex = SignalRegionIndex[0] + np.argmax(ccf[SignalRegionIndex[0]:SignalRegionIndex[1]])
    SignalStrength = ccf[maxindex]
    
    ExcludeSignalMask = np.zeros_like(ccf)
    ExcludeSignalMask[SignalRegionIndex[0]:SignalRegionIndex[1]] = 1
    
    ### For an RV range of -1000 to + 1000 in steps of 1 (2001 points)
    ###HalfExcludeForNoiseRegion = 400  was initially 400 but I think it would be better to include more baseline and this still excludes the first and last 500 with 1000 each side (total 2000)
    # HalfExcludeForNoiseRegion = 500
    # ExcludeSignalMask[0:MiddleIndex-HalfExcludeForNoiseRegion] = 1
    # ExcludeSignalMask[MiddleIndex+HalfExcludeForNoiseRegion:] = 1    
    
    # HalfExcludeForNoiseRegion = 300
    # ExcludeSignalMask[0:MiddleIndex-HalfExcludeForNoiseRegion] = 1
    # ExcludeSignalMask[MiddleIndex+HalfExcludeForNoiseRegion:] = 1  
    
    ### Simpler way to exclude the first and last N points from being considered in the standard deviation 
    ExcludeSignalMask[0:NumberOfLeadingAndTrailingPointsToExclude] = 1
    ExcludeSignalMask[-NumberOfLeadingAndTrailingPointsToExclude:] = 1
    

    
    ma = np.ma.MaskedArray(ccfNoInj, mask=ExcludeSignalMask)
    
    # if OrderIndex == 15:
    #     np.savetxt('InjSig.txt',ccf)
    #     np.savetxt('NoInjForNoise.txt',ccfNoInj)
    
    noise = ma.std()    
    
    if Adjust:
        SNR = (SignalStrength-noise)/noise
    
    if not Adjust:
        SNR = SignalStrength/noise
    
    return SNR, SignalStrength, noise

def AssignWeightFromSNR(ccfNoInj,ccf,NumberOfInjectedLines,SignalRegionIndex=(975,985)):
        
    if NumberOfInjectedLines == 0:
        return 0
    
    else:    
        SNR, SignalStrength, noise = MeasureDetectionSNR(ccfNoInj,ccf,SignalRegionIndex=SignalRegionIndex,Adjust=False)
        
        CuttOffSNR = 2.5
        
        if SNR < CuttOffSNR:
            
            return 0
    
        if SNR >= CuttOffSNR:
            return SNR 
        
        
# def SplitIntoOddAndEven(a,b):    
    
#     NumRows = len(b)
    
#     Odda = []
#     Oddb = []
    
#     Evena = []
#     Evenb = []
    
#     for i in range(NumRows):
#         if (i % 2) == 0: 
#             ## Is even 
#             Evena.append(a[i,:])
#             Evenb.append(b[i])
        
#         if (i % 2) != 0: 
#             Odda.append(a[i,:])
#             Oddb.append(b[i])
            
#     Odda = np.array(Odda)
#     oddb = np.array(Oddb)
        
#     Evena = np.array(Evena)
#     Evenb = np.array(Evenb)
    
#     return (Odda,oddb,Evena,Evenb)
    
    
StartTime = time.time()

FirstPartOfLoadPath = '../CrossCorrelationDataAndProcessing'
#FirstPartOfLoadPath = 'F:'


SystemicVelocity = -20.567  ## kms for KELT-9b from NASA Exoplanet Archive
PlanetInclination = 87.2*np.pi/180   ## for KELT-9b, Ahlers et al. 2020 



DataOrigin = 'KELT-9b_CARMENES_emission_data'



##############################


# ModelForCrossCor = 'KELT9b_Al_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = 'KELT9b_Al_0.50_+0.0_0.55_Vrot6.63'
# #### 300, 120, 75, 450
# #ModelScalingFactorList = [300, 120, 75, 450]
# ModelScalingFactorList = [1,1,1,1]

# WriteWeights = True 





# ModelForCrossCor = 'KELT9b_Ca_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = ModelForCrossCor
# #ModelScalingFactorList = [500, 130, 100, 600]
# ModelScalingFactorList = [1,1,1,1]
# WriteWeights = True 



# ModelForCrossCor = 'KELT9b_CaII_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = ModelForCrossCor
# #ModelScalingFactorList = [1000, 270, 200, 1000]
# ModelScalingFactorList = [1,1,1,1]
# WriteWeights = True 


# ModelForCrossCor = 'KELT9b_Cr_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = ModelForCrossCor
# #ModelScalingFactorList = [200, 50, 40, 200]
# ModelScalingFactorList = [1,1,1,1]
# WriteWeights = True 




# ModelForCrossCor = 'KELT9b_Fe_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = ModelForCrossCor
# ModelScalingFactorList = [150, 40, 30, 150]
# WriteWeights = True 

# ModelForCrossCor = 'KELT9b_FeH_main_iso_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = 'KELT9b_FeH_main_iso_0.50_+0.0_0.55_Vrot6.63'
# ### 750, 200, 150, 750
# #ModelScalingFactorList = [750, 200, 150, 750]
# ModelScalingFactorList = [1,1,1,1]
# WriteWeights = True 


# ModelForCrossCor = 'KELT9b_FeII_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = 'KELT9b_FeII_0.50_+0.0_0.55_Vrot6.63'
# ### 1000, 270, 200, 1000
# ### 1500, 400, 300, 1500
# #ModelScalingFactor = 300
# #ModelScalingFactor = 1
# #ModelScalingFactorList = [1500, 400, 300, 1500]
# ModelScalingFactorList = [1,1,1,1]
# WriteWeights = True 


# ModelForCrossCor = 'KELT9b_FeII_UsingFeI_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = 'KELT9b_FeII_UsingFeI_0.50_+0.0_0.55_Vrot6.63'
# ### 1000, 270, 200, 1000
# ### 1500, 400, 300, 1500
# #ModelScalingFactor = 300
# #ModelScalingFactor = 1
# ModelScalingFactorList = [1500, 400, 300, 1500]
# #ModelScalingFactorList = [1,1,1,1]
# WriteWeights = True 






# ModelForCrossCor = 'KELT9b_K_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = 'KELT9b_K_0.50_+0.0_0.55_Vrot6.63'
# ## 500, 130, 100, 500
# #ModelScalingFactorList = [500, 130, 100, 500]
# ModelScalingFactorList = [1,1,1,1]
# WriteWeights = True 


# ModelForCrossCor = 'KELT9b_Mg_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = ModelForCrossCor
# #ModelScalingFactorList = [400, 100, 80, 400]
# ModelScalingFactorList = [1,1,1,1]
# WriteWeights = True 




# ModelForCrossCor = 'KELT9b_Na_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = 'KELT9b_Na_0.50_+0.0_0.55_Vrot6.63'
# # ###  500, 130, 100, 500
# #ModelScalingFactorList = [500, 130, 100, 500]
# ModelScalingFactorList = [1,1,1,1]
# WriteWeights = True 





# ModelForCrossCor = 'KELT9b_Ti_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = ModelForCrossCor
# #ModelScalingFactorList = [150, 45, 25, 150]
# ModelScalingFactorList = [1,1,1,1]
# WriteWeights = True 




# ModelForCrossCor = 'KELT9b_TiO_all_iso_Plez_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = ModelForCrossCor
# ## 750, 195, 150, 750
# ### 1000, 300, 230, 1000  ## These weights gave a slightly worse detection of TiO

# #### 1600, 400, 300, 1800

# ModelScalingFactorList = [750, 195, 150, 750]
# WriteWeights = True 



# ModelForCrossCor = 'KELT9b_TiO_48_Exomol_McKemmish_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = ModelForCrossCor
# #ModelScalingFactorList = [750, 195, 150, 750]
# ModelScalingFactorList = [1,1,1,1]
# ##ModelScalingFactorList = [3]
# WriteWeights = True 

# ModelForCrossCor = 'KELT9b_TiO_48_Plez_0.50_+0.0_0.55_Vrot6.63'
# TargetForCrossCor = ModelForCrossCor
# ModelScalingFactorList = [750, 195, 150, 750]
# WriteWeights = True 



# # ModelForCrossCor = 'KELT9b_Al_0.50_+0.0_0.55_Vrot6.63'
# # # ModelForCrossCor = 'KELT9b_Al_0.50_+2.3_0.55_Vrot6.63'
# # #ModelForCrossCor = 'KELT9b_Ca_0.50_+0.0_0.55_Vrot6.63'
# # #ModelForCrossCor = 'KELT9b_CaII_0.50_+0.0_0.55_Vrot6.63'
# # #ModelForCrossCor = 'KELT9b_Cr_0.50_+0.0_0.55_Vrot6.63'
# # #ModelForCrossCor = 'KELT9b_Fe_0.50_+0.0_0.55_Vrot6.63'
# #ModelForCrossCor = 'KELT9b_FeH_main_iso_0.50_+0.0_0.55_Vrot6.63'
# #ModelForCrossCor = 'KELT9b_FeII_0.50_+0.0_0.55_Vrot6.63'
ModelForCrossCor = 'KELT9b_FeII_UsingFeI_0.50_+0.0_0.55_Vrot6.63'



#ModelForCrossCor = 'KELT9b_K_0.50_+0.0_0.55_Vrot6.63'
# #ModelForCrossCor = 'KELT9b_Mg_0.50_+0.0_0.55_Vrot6.63'
#ModelForCrossCor = 'KELT9b_Na_0.50_+0.0_0.55_Vrot6.63'
# #ModelForCrossCor = 'KELT9b_Ti_0.50_+0.0_0.55_Vrot6.63'
# #ModelForCrossCor = 'KELT9b_TiO_all_iso_Plez_0.50_+0.0_0.55_Vrot6.63'
# #ModelForCrossCor = 'KELT9b_TiO_48_Exomol_McKemmish_0.50_+0.0_0.55_Vrot6.63'
# #ModelForCrossCor = 'KELT9b_TiO_48_Plez_0.50_+0.0_0.55_Vrot6.63'

#ModelForCrossCor = '6000_J23245+578_flatter'
#ModelForCrossCor = '6000_J07274+052_flatter'
#ModelForCrossCor = '6000_J22565+165_flatter'


TargetForCrossCor = 'NoModel'
ModelScalingFactorList = [0,0,0,0]
####ModelScalingFactorList = [0]
WriteWeights = False 



#CCFMedSub = 0 ## Unmodified CCFs without subtracting the median 
CCFMedSub = 1 ## Subtract from each CCF its median 
#CCFMedSub = 2 ## Subtract from each CCF its meidan then subtract the median from every column 



#KpMode = 'SingleKp'
KpMode = 'MultiKp'

#SpecWeightConfig = 1 ### Weighting spec by Square of average error and Lambert  
#SpecWeightConfig = 2 ### Weighting spec by average error and Lambert 
SpecWeightConfig = 3 ### No spec weighting  
#SpecWeightConfig = 4 ### Weighting spec by Lambert sphere 

###############################

AbsorptionOrEmission = 'Emission'

UpToComponents = 10
# if TargetForCrossCor == 'NoModel':
#     UpToComponents = 10    

### ^^^
#NightList = ['20180609All']
#NightList = ['20180618All']
#NightList = ['20190528All']
#NightList = ['20190604All']

### NightList = ['20180609All','20180618All','20190528All','20190604All']
NightList = ['20180609All','20180618All','20190528P2','20190604All']


#ArmSubpartList = [['vis','A'],['nir','A'],['nir','B']]
ArmSubpartList = [['vis','A'],['nir','A']]
#ArmSubpartList = [['nir','A'],['nir','B']]


#ArmSubpartList = [['vis','A']]
#ArmSubpartList = [['nir','A']]
#ArmSubpartList = [['nir','B']]

ExcludeLastNInTransit = 0
ModelShift_kms = 0

#SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig1.csv')
#SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig2.csv')
#SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig3.csv')
#SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig4.csv')
SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig6.csv')

MakePlots = False

OrbitalVelocity_kms = 2*np.pi*(0.035*1.496e8)/(1.4811235*24*3600)   ### For KELT-9b from exomast 
OrbitalInclination = 87.2*np.pi/180

RealKp = 241 ### 
BestSys = -20

#KpVect = np.arange(1,501,1)

if KpMode == 'SingleKp':
    KpVect = np.array([RealKp])

if KpMode == 'MultiKp':
    KpVect = np.arange(90,391,1)
    ###KpVect = np.array([240,241,242])

    ####KpVect = np.linspace(1,500,10)

ClosestKpRow = np.argmin(np.abs(KpVect-RealKp))

OrbVelVect = KpVect/np.sin(OrbitalInclination)   

PlanetSpectrum = np.load('%s/KELT-9b_CARMENES_emission_data/ModelSpectra/%s_CarmenesRes_pRT_flux_per_Hz.npy'%(FirstPartOfLoadPath,ModelForCrossCor))
PlanetSpectrumWave = PlanetSpectrum[:,0]
PlanetSpectrumFlux = PlanetSpectrum[:,1]

PlanetSpecInterpObject = interp1d(PlanetSpectrumWave,PlanetSpectrumFlux)            
 

#for NightIndex in range(len(NightList)):
for NightIndex in [1,2]:
    
    night = NightList[NightIndex]
    ModelScalingFactor = ModelScalingFactorList[NightIndex]

    for ArmSubpartIndex in range(len(ArmSubpartList)):
        
        arm = ArmSubpartList[ArmSubpartIndex][0]
        arm_subpart = ArmSubpartList[ArmSubpartIndex][1]    
    
        FullOutput = True 
        
        PCA_algorithm = 'sysrem'
        
        if DataOrigin == 'KELT-9b_CARMENES_emission_data':
            
            if arm == 'nir':
                FileNameString = 'nir_A'
                NumOrders = 28
                #NumberOfSysremIterationsVect = np.ones((NumOrders))*6
                #NumOrders = 1
            if arm == 'vis':
                FileNameString = 'vis_A'         
                NumOrders = 61
                #NumberOfSysremIterationsVect = np.ones((NumOrders))*6
        print()
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('Lookup key')
        print('%s_%s_%s'%(night,arm,arm_subpart))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print()
        NumberOfSysremIterationsVect = SysremItConfig_df['%s_%s_%s'%(night,arm,arm_subpart)].values            
        #NumberOfSysremIterationsVect = np.ones(NumOrders)*6
        
        OrderListToProcess = range(NumOrders)
    
        mjdLoadDirectory = '%s/%s/ProcessedData/%s/%s/%s'%(FirstPartOfLoadPath,DataOrigin,TargetForCrossCor,night,arm)
        
        planetrv = np.loadtxt('%s/radv.txt'%(mjdLoadDirectory))
        phase =  np.loadtxt('%s/phase.txt'%(mjdLoadDirectory))
        baryrv = np.loadtxt('%s/BarycentricRVcorrection_kms.txt'%(mjdLoadDirectory))
        
        AvgBaryRV = np.mean(baryrv)
        #AvgPlanetRV = np.mean(planetrv)
        AvgPlanetRV = np.median(planetrv)
        
        
        BaseRV = SystemicVelocity - AvgBaryRV + AvgPlanetRV
        
        DetectionSignificanceList = []
        
    
            
        if WriteWeights:
            OrderWeights = np.ones((NumOrders))
        if not WriteWeights:
            
            ### A quick hack for it to not try to load the weights for the average here. Instead just do the Kps 
            #OrderWeights = np.loadtxt('../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/ProcessedData/%s/%s/%s/Part%s/ModelScalingFactor%.5e/ModelShift_0.000e+00_kms/XCorr/%s/sysremed_%d/KpVsys/OrderWeightsFromInjectedSignal.txt'%(ModelForCrossCor,night,arm,arm_subpart,InjectedScalingFactorForWeights,ModelForCrossCor,SysRemComponentsRemovedForWeights))**2
            OrderWeights = np.ones((NumOrders))    
        
        UncertaintyMatrixPerOrderList = []
        NumberOfLinesPerOrderList = []           
        
        mjd = np.loadtxt('%s/%s/ProcessedData/%s/%s/%s/mjd.txt'%(FirstPartOfLoadPath,DataOrigin,TargetForCrossCor,night,arm))
        BarycentricRVcorrection = np.loadtxt('%s/%s/ProcessedData/%s/%s/%s/BarycentricRVcorrection_kms.txt'%(FirstPartOfLoadPath,DataOrigin,TargetForCrossCor,night,arm))
        
        ### ***
        ListOfOrdersToDo = range(NumOrders)
        #ListOfOrdersToDo = range(30,NumOrders)
        #ListOfOrdersToDo = range(30)
        
        for OrderIndex in ListOfOrdersToDo:
        #for OrderIndex in range(NumOrders): 
            
            print('Transforming %s %s %s order %d'%(night,arm,arm_subpart,OrderIndex))
            
        #for OrderIndex in [0]: 
            
            NumberOfSysremIterations = NumberOfSysremIterationsVect[OrderIndex]
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # print('NumberOfSysremIterations: %d'%(NumberOfSysremIterations))
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        
            LoadDirectory = '%s/KELT-9b_CARMENES_emission_data/XCorr/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d'%(FirstPartOfLoadPath,night,TargetForCrossCor,ModelForCrossCor,ModelScalingFactor,arm,arm_subpart,OrderIndex,NumberOfSysremIterations)
            NoInjLoadDirectory = '%s/KELT-9b_CARMENES_emission_data/XCorr/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d'%(FirstPartOfLoadPath,night,'NoModel',ModelForCrossCor,0,arm,arm_subpart,OrderIndex,NumberOfSysremIterations)
            #NoInjLoadDirectory = '%s/KELT-9b_CARMENES_emission_data/XCorr_ActualKp/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d'%(FirstPartOfLoadPath,night,'NoModel',ModelForCrossCor,0,arm,arm_subpart,OrderIndex,NumberOfSysremIterations)

            
            #OutputDirectory = '%s/KpVsys'%(LoadDirectory)
            OutputDirectory = '%s/KELT-9b_CARMENES_emission_data/%s/SpecWeightConfig%d/CCFMedSub%d/%s/%s/%s/ModelScalingFactor%.5e/%s_%s/Order%d/Sysrem%d'%(FirstPartOfLoadPath,
                                                                                                                                                KpMode,
                                                                                                                                                SpecWeightConfig,
                                                                                                                                                CCFMedSub,
                                                                                                                                                night,
                                                                                                                                                TargetForCrossCor,
                                                                                                                                                ModelForCrossCor,
                                                                                                                                                ModelScalingFactor,
                                                                                                                                                arm,
                                                                                                                                                arm_subpart,
                                                                                                                                                OrderIndex,
                                                                                                                                                NumberOfSysremIterations)       
            
            
            if not os.path.exists(OutputDirectory):
                os.makedirs(OutputDirectory)       
            
            dPerOrder = np.load('%s/Xcorr_order%d_Sysrem%d.npy'%(LoadDirectory,OrderIndex,NumberOfSysremIterations))        
            
            if WriteWeights:
                dNoInjPerOrder = np.load('%s/Xcorr_order%d_Sysrem%d.npy'%(NoInjLoadDirectory,OrderIndex,NumberOfSysremIterations))
                
            if not WriteWeights: 
                dNoInjPerOrder = dPerOrder
        
            RadVelVect = np.loadtxt('%s/sysremed_%d_XCorrRadVelVect.txt'%(LoadDirectory,NumberOfSysremIterations))
            
            NumRadVelPoints = len(RadVelVect)    
            SignalRegionIndices = ((int(NumRadVelPoints/2)-25,int(NumRadVelPoints/2)-15))
        
            ### Make a plot of the injected spectrum over this order 
            if MakePlots:
                InjSpecLinesPDF = PdfPages('%s/NumberOfLinesPerOrder.pdf'%(OutputDirectory))
        
            
            SpectrumUncertaintyLoadDirectory = '%s/%s/ProcessedData/%s/%s/%s/Part%s/ModelScalingFactor%0.5e/ModelShift_%.3e_kms/Order%d/'%(FirstPartOfLoadPath,DataOrigin,TargetForCrossCor,night,arm,arm_subpart,ModelScalingFactor,ModelShift_kms,OrderIndex)      
            
            
            UncertaintyMatrixPerOrderList.append(pyfits.getdata('%s/CroppedBlazeCorrectedUncArray.fits'%(SpectrumUncertaintyLoadDirectory)))        
            
            WaveMatrixPerOrder = pyfits.getdata('%s/Wave4_ToSysrem.fits'%(SpectrumUncertaintyLoadDirectory))
            WavePerOrder = WaveMatrixPerOrder[0,:]/1e4                
            
            PlanetSpecFluxPerOrder = PlanetSpecInterpObject(WavePerOrder)                
        
            peaks,properties = find_peaks(PlanetSpecFluxPerOrder)
            
            NumberOfInjectedLines = len(peaks)            
            NumberOfLinesPerOrderList.append(NumberOfInjectedLines)
            
            if MakePlots:
                plt.figure()
                plt.plot(PlanetSpecFluxPerOrder)
                plt.plot(peaks,PlanetSpecFluxPerOrder[peaks],'x')
                plt.title('Order %d has %d lines'%(OrderIndex,NumberOfInjectedLines))
                InjSpecLinesPDF.savefig()        
                InjSpecLinesPDF.close()
        
        
        
        ##PlanetFrameRVvect = RadVelVect - BaseRV
        
            plphase,plradv,transitlims,num_orbits = JDtophase_radial_vel(mjd+2400000.5)
        
            numspec,numradv = np.shape(dPerOrder)
            
            NotNormalizedCrossCor = np.copy(dPerOrder)    
        
            #### Some CCFs has overall systematic offsets so this normalizes them all to be around 0
            ### Seeing if this is actually needed for this data
            
    
            if CCFMedSub == 1:
                dPerOrder -= np.median(dPerOrder,axis=1,keepdims=True)
                dNoInjPerOrder -= np.median(dNoInjPerOrder,axis=1,keepdims=True)       
                
            
            if CCFMedSub == 2: 
                dPerOrder -= np.median(dPerOrder,axis=1,keepdims=True)
                dNoInjPerOrder -= np.median(dNoInjPerOrder,axis=1,keepdims=True)                   
    
                dPerOrder -= np.median(dPerOrder,axis=0)
                dNoInjPerOrder -= np.median(dNoInjPerOrder,axis=0)      
            
    
            if MakePlots:
                ### Make some PDF plots     
                CCFPRVlotLims = (-600,600)    
                RadVPlotLimits = (-250,250)   
                
                ImShowXLimits = int(NumRadVelPoints/2) + np.array([-200,200])    
                KpVsysXaxis = RadVelVect[ImShowXLimits[0]:ImShowXLimits[1]]
                
                # CCFPRVlotLims = (-375,25)    
                # RadVPlotLimits = (-400,400)   
                
                # ImShowXLimits = 1000 + np.array([-400,400])    
                # KpVsysXaxis = RadVelVect[ImShowXLimits[0]:ImShowXLimits[1]]
                
                # ImShowXLimits = [0,-1]    
                # KpVsysXaxis = RadVelVect[ImShowXLimits[0]:ImShowXLimits[1]]
            
            
                with PdfPages('%s/XCorrPlot.pdf'%(OutputDirectory)) as pdf:         
                    
                    plt.figure()
                    plt.title('CCF of order %d'%(OrderIndex))
                    plt.imshow(dPerOrder[:,ImShowXLimits[0]:ImShowXLimits[1]],interpolation='None',origin='lower',aspect='auto',extent=(RadVelVect[0],RadVelVect[-1],0,numspec))        
            
                    plt.colorbar()
                    plt.xlabel('radial velocity of template for CCF (km/s)')
                    plt.ylabel('spectrum number')
            
                    plt.xlim(CCFPRVlotLims[0],CCFPRVlotLims[-1])
                    plt.tight_layout()
            
                    pdf.savefig()      
                    plt.close()
    
    
            ### Find the closest column to the literature value 
            ClosestVsysCol = np.argmin(np.abs(BestSys-RadVelVect))
        
        # testrv = np.zeros((28))
        # testrv[0] = 100
    
            KpResultsPerOrder = np.zeros((len(KpVect),numradv))
            KpResultsPerOrderNoInj = np.zeros_like(KpResultsPerOrder)
            
            KpResultsPerOrderOdd = np.zeros_like(KpResultsPerOrder)
            #KpResultsPerOrderNoInjOdd = np.zeros_like(KpResultsPerOrder)
            
            KpResultsPerOrderEven = np.zeros_like(KpResultsPerOrder)
            #KpResultsPerOrderNoInjEven = np.zeros_like(KpResultsPerOrder)
            
            
            SpecUncWeightingsPerOrder = np.empty((numspec))
            SpecTotalWeightingsPerOrder = np.empty_like(SpecUncWeightingsPerOrder)    
        
            #################################################################
        
            #### Additions to save the shifted CCFs 
            if MakePlots:
                ShiftedCCFpdf = PdfPages('%s/CCFsInPlanetFrame.pdf'%(OutputDirectory))    
            
            if SpecWeightConfig == 1:
                SpecNoiseWeights = (1/np.nanmedian(UncertaintyMatrixPerOrderList[OrderIndex],axis=1))**2  ## This was initally squared thinking it should be like photon noise statisics but I think now that the CARACAL pipeline should already take that into account so it is probably more correct to not square these uncertainties. I think it shouldn't make much difference 
                WeightsFromPhase = LambertSphere(phase,i=PlanetInclination)
                
            if SpecWeightConfig == 2:
                SpecNoiseWeights = (1/np.nanmedian(UncertaintyMatrixPerOrderList[OrderIndex],axis=1))
                WeightsFromPhase = LambertSphere(phase,i=PlanetInclination)
                
            if SpecWeightConfig == 3:
                SpecNoiseWeights = np.ones_like(phase)
                WeightsFromPhase = np.copy(SpecNoiseWeights)         
                
            if SpecWeightConfig == 4:
                SpecNoiseWeights = np.ones_like(phase)
                WeightsFromPhase = LambertSphere(phase,i=PlanetInclination)
            
                
            #SpecNoiseWeights = (1/np.nanmedian(UncertaintyMatrixPerOrderList[OrderIndex],axis=1))  ## Trying without squaring the uncertainties from CARACAL (see comment on line above 
    
            ScaledSpecNoiseWeights = SpecNoiseWeights/np.max(SpecNoiseWeights)
            SpectrumWeightings = WeightsFromPhase*ScaledSpecNoiseWeights
            
            SpecUncWeightingsPerOrder[:] = ScaledSpecNoiseWeights
            SpecTotalWeightingsPerOrder[:] = SpectrumWeightings    
            
            
            
            dNumRows,dNumCols = np.shape(dPerOrder[:,:])
            
            ShiftedXCorr = np.zeros((dNumRows,dNumCols,len(KpVect)))
            ShiftedXCorrNoInj = np.copy(ShiftedXCorr)
            
            ## Do the shifts in a fast way without defining the interpolation object every time 
            for SpecIndex in range(numspec):            
                
                SpecInterpObj = interp1d(RadVelVect, dPerOrder[SpecIndex,:], kind='linear',bounds_error=False,fill_value=np.nan) 
                SpecNoInjInterpObj = interp1d(RadVelVect, dNoInjPerOrder[SpecIndex,:], kind='linear',bounds_error=False,fill_value=np.nan) 
                
                for KpIndex in range(len(KpVect)):                
                                        
                    OrbVel = OrbVelVect[KpIndex]
                    
                    plphase,plradv,transitlims,num_orbits = JDtophase_radial_vel(mjd+2400000.5,vorbital=OrbVel)
                    
                    NewRadVelVect = RadVelVect + plradv[SpecIndex] - BarycentricRVcorrection[SpecIndex] ## Trialled + and - to get the aligned signal at the corret systemic velocity. They are basically the opposite sign of what they were in the inital way of defining the interpolation object on the shifted radial velocities so it makes sense 
    
                    #### For investigating how the barycentric correction affects the apparent velocity of the signal      
                    ##NewRadVelVect = RadVelVect + plradv[SpecIndex]                     
                    
                    ShiftedXCorr[SpecIndex,:,KpIndex] = SpecInterpObj(NewRadVelVect)
                    ShiftedXCorrNoInj[SpecIndex,:,KpIndex] = SpecNoInjInterpObj(NewRadVelVect)
                    
            
            ### Now that CCFs are aligned, sum over them for each Kp 
            for KpIndex in range(len(KpVect)):                  

                if AbsorptionOrEmission == 'Emission':
    
                    KpResultsPerOrder[KpIndex,:] = WeightedNanMeanOverRows(ShiftedXCorr[:,:,KpIndex],SpectrumWeightings)
                    KpResultsPerOrderNoInj[KpIndex,:] = WeightedNanMeanOverRows(ShiftedXCorrNoInj[:,:,KpIndex],SpectrumWeightings)
                    
                    
                    OddShiftedXCorr = ShiftedXCorr[1::2,:,:]
                    OddSpectrumWeightings = SpectrumWeightings[1::2]
                    
                    EvenShiftedXCorr = ShiftedXCorr[::2,:,:]
                    EvenSpectrumWeightings = SpectrumWeightings[::2]
                    
                    
                    #ShiftedXCorrAndWeightsSplitIntoOddAndEven = SplitIntoOddAndEven(ShiftedXCorr,SpectrumWeightings)
                    #ShiftedXCorrAndWeightsNoInjSplitIntoOddAndEven = SplitIntoOddAndEven(ShiftedXCorrNoInj,SpectrumWeightings)
        
                    
                    KpResultsPerOrderOdd[KpIndex,:] = WeightedNanMeanOverRows(OddShiftedXCorr[:,:,KpIndex],OddSpectrumWeightings)
                    #KpResultsPerOrderNoInjOdd[KpIndex,:] = WeightedNanMeanOverRows(ShiftedXCorrAndWeightsNoInjSplitIntoOddAndEven[0],ShiftedXCorrAndWeightsNoInjSplitIntoOddAndEven[1])
                    
                    
                    KpResultsPerOrderEven[KpIndex,:] = WeightedNanMeanOverRows(EvenShiftedXCorr[:,:,KpIndex],EvenSpectrumWeightings)
                    #KpResultsPerOrderNoInjEven[KpIndex,:] = WeightedNanMeanOverRows(ShiftedXCorrAndWeightsNoInjSplitIntoOddAndEven[2],ShiftedXCorrAndWeightsNoInjSplitIntoOddAndEven[3])
                
            ###############################################
            
            
            ##### Original way with spec and Kp loops orders reversed so that the interpolation object had to be in each new RV from Kp  
            # for KpIndex in range(len(KpVect)):
            #     # if (KpIndex % 100) == 0:
            #     #     print('Transforming order %d of %d with KpIndex %d of %d'%(OrderIndex,NumOrders,KpIndex,len(KpVect)))        
                
            #     ShiftedXCorr = np.zeros_like(dPerOrder[:,:])
            #     ShiftedXCorrNoInj = np.zeros_like(ShiftedXCorr)
                        
            #     OrbVel = OrbVelVect[KpIndex]
                
            #     plphase,plradv,transitlims,num_orbits = JDtophase_radial_vel(mjd+2400000.5,vorbital=OrbVel)
                
            #     for SpecIndex in range(numspec):
            #         if AbsorptionOrEmission == 'Emission':
            #             NewRadVelVect = RadVelVect - plradv[SpecIndex]
                        
            #         ShiftedAsFunc = interp1d(NewRadVelVect, dPerOrder[SpecIndex,:], kind='linear',bounds_error=False,fill_value=np.nan) 
            #         ShiftedXCorr[SpecIndex,:] = ShiftedAsFunc(RadVelVect)
                    
            #         ShiftedAsFuncNoInj = interp1d(NewRadVelVect, dNoInjPerOrder[SpecIndex,:], kind='linear',bounds_error=False,fill_value=np.nan) 
            #         ShiftedXCorrNoInj[SpecIndex,:] = ShiftedAsFuncNoInj(RadVelVect)
        
    
            #         if AbsorptionOrEmission == 'Absorption':
            #             KpResultsPerOrder[KpIndex,:] = np.nanmean(ShiftedXCorr[transitlims[0][0]:transitlims[0][-1]-ExcludeLastNInTransit,:],axis=0)
            #         if AbsorptionOrEmission == 'Emission':
            #             KpResultsPerOrder[KpIndex,:] = WeightedNanMeanOverRows(ShiftedXCorr,SpectrumWeightings)
            #             KpResultsPerOrderNoInj[KpIndex,:] = WeightedNanMeanOverRows(ShiftedXCorrNoInj,SpectrumWeightings)
                        
                        
            #             OddShiftedXCorr = ShiftedXCorr[1::2,:]
            #             OddSpectrumWeightings = SpectrumWeightings[1::2]
                        
            #             EvenShiftedXCorr = ShiftedXCorr[::2,:]
            #             EvenSpectrumWeightings = SpectrumWeightings[::2]
                        
                        
            #             #ShiftedXCorrAndWeightsSplitIntoOddAndEven = SplitIntoOddAndEven(ShiftedXCorr,SpectrumWeightings)
            #             #ShiftedXCorrAndWeightsNoInjSplitIntoOddAndEven = SplitIntoOddAndEven(ShiftedXCorrNoInj,SpectrumWeightings)
    
                        
            #             KpResultsPerOrderOdd[KpIndex,:] = WeightedNanMeanOverRows(OddShiftedXCorr,OddSpectrumWeightings)
            #             #KpResultsPerOrderNoInjOdd[KpIndex,:] = WeightedNanMeanOverRows(ShiftedXCorrAndWeightsNoInjSplitIntoOddAndEven[0],ShiftedXCorrAndWeightsNoInjSplitIntoOddAndEven[1])
                        
                        
            #             KpResultsPerOrderEven[KpIndex,:] = WeightedNanMeanOverRows(EvenShiftedXCorr,EvenSpectrumWeightings)
                        #KpResultsPerOrderNoInjEven[KpIndex,:] = WeightedNanMeanOverRows(ShiftedXCorrAndWeightsNoInjSplitIntoOddAndEven[2],ShiftedXCorrAndWeightsNoInjSplitIntoOddAndEven[3])
                        
                        
                        
                if MakePlots:    
                    if KpIndex == ClosestKpRow:
                        plt.figure()
                        plt.title('CCF of order %d\n(stellar frame)'%(OrderIndex))
                        #plt.imshow(d[:,ImShowXLimits[0]:ImShowXLimits[1],OrderIndex],interpolation='None',origin='lower',aspect='auto',extent=(RadVelVect[0],RadVelVect[-1],0,numspec))                        
                        plt.imshow(dPerOrder[:,:],interpolation='None',origin='lower',aspect='auto',extent=(RadVelVect[0],RadVelVect[-1],0,numspec))                        
                        plt.colorbar()
                        plt.xlabel('radial velocity of template for CCF (km/s)')
                        plt.ylabel('spectrum number')                
                        #plt.xlim(CCFPRVlotLims[0],CCFPRVlotLims[-1])
                        plt.tight_layout()                
                        ShiftedCCFpdf.savefig()      
                        plt.close()
                        
                        plt.figure()
                        plt.title('Planet frame CCF of order %d\nusing Kp = %f (index %d)'%(OrderIndex,KpVect[ClosestKpRow],ClosestKpRow))
                        #plt.imshow(ShiftedXCorr[:,600:1400],interpolation='None',origin='lower',aspect='auto',extent=(RadVelVect[600],RadVelVect[1400],0,numspec))                        
                        plt.imshow(ShiftedXCorr,interpolation='None',origin='lower',aspect='auto',extent=(RadVelVect[0],RadVelVect[-1],0,numspec))    
                        plt.colorbar()
                        plt.xlabel('radial velocity of template for CCF (km/s)')
                        plt.ylabel('spectrum number')                
                        #plt.xlim(CCFPRVlotLims[0],CCFPRVlotLims[-1])
                        plt.tight_layout()                
                        ShiftedCCFpdf.savefig()      
                        plt.close()         
                        
            if MakePlots:            
                ShiftedCCFpdf.close()                
          
        #########################
        
    
        
        
        # NetSystemic = SystemicVelocity #- AvgBaryRV       
        
        # BestKpForGivenVsys = KpVect[np.argmax(SelectiveAverage[:,ClosestVsysCol])]
            
                
            np.save('%s/KpXCorrOrder%d.npy'%(OutputDirectory,OrderIndex),KpResultsPerOrder)
            np.savetxt('%s/KpVect.txt'%(OutputDirectory),KpVect)    
            
            np.save('%s/KpXCorrOrder%d_OnlyOddSpectra.npy'%(OutputDirectory,OrderIndex),KpResultsPerOrderOdd)
            np.save('%s/KpXCorrOrder%d_OnlyEvenSpectra.npy'%(OutputDirectory,OrderIndex),KpResultsPerOrderEven)
        
        
            np.savetxt('%s/SpecUncertaintyWeightingsPerOrder.txt'%(OutputDirectory),SpecUncWeightingsPerOrder)
            np.savetxt('%s/SpecTotalWeightingsPerOrder.txt'%(OutputDirectory),SpecTotalWeightingsPerOrder)
            np.savetxt('%s/SpecWeightingsFromPhase.txt'%(OutputDirectory),WeightsFromPhase)      
        
            if WriteWeights:
                WeightFromInjectedSignalPerOrder = np.empty((1))
                DetSigFromInjectedSignal = np.empty_like(WeightFromInjectedSignalPerOrder)            
                     
                WeightFromInjectedSignalPerOrder[0] = AssignWeightFromSNR(KpResultsPerOrderNoInj[ClosestKpRow,:],KpResultsPerOrder[ClosestKpRow,:],NumberOfLinesPerOrderList[OrderIndex], SignalRegionIndex = SignalRegionIndices)
                DetSigFromInjectedSignal[0] = MeasureDetectionSNR(KpResultsPerOrderNoInj[ClosestKpRow,:],KpResultsPerOrder[ClosestKpRow,:], SignalRegionIndex = SignalRegionIndices)[0]
            
            if WriteWeights:
                # print('Trying to write weights to:')
                # print('%s/OrderWeightsFromInjectedSignal.txt'%(OutputDirectory))
                np.savetxt('%s/OrderWeightsFromInjectedSignal.txt'%(OutputDirectory),WeightFromInjectedSignalPerOrder)
                
    ###################################
            
            if MakePlots:
                with PdfPages('%s/Kp_plots_Per_order.pdf'%(OutputDirectory)) as pdf:     
                    
                    plt.figure()
                    plt.title('Order %d'%(OrderIndex))
                    plt.imshow(KpResultsPerOrder[:,ImShowXLimits[0]:ImShowXLimits[1]],interpolation='None',origin='lower',aspect='auto',extent=(KpVsysXaxis[0],KpVsysXaxis[-1],KpVect[0],KpVect[-1]))
                    plt.xlabel('Systemic velocity (km s$^{-1}$)')
                    plt.ylabel(r'K$_p$ (km s$^{-1}$)')        
                    plt.colorbar()
                    plt.plot([KpVsysXaxis[0],KpVsysXaxis[-1]],[RealKp,RealKp],'w:')
                    plt.plot([BestSys,BestSys],[KpVect[0],KpVect[-1]],'w:')
                    #plt.xlim(RadVPlotLimits[0],RadVPlotLimits[-1])
                    pdf.savefig()
                    plt.close()
                    
                    plt.figure()
                    plt.plot(KpVsysXaxis,KpResultsPerOrder[ClosestKpRow,ImShowXLimits[0]:ImShowXLimits[1]])
                    plt.xlabel('Systemic velocity (km s$^{-1}$)')
                    plt.ylabel('CCF')
                    #plt.plot([BestSys,BestSys],[-0.001,1.2*np.max(SelectiveAverage[ClosestKpRow,:])],'k:')
                    #plt.xlim(RadVPlotLimits[0],RadVPlotLimits[-1])
                    
                    SNRValuesForTitle = MeasureDetectionSNR(KpResultsPerOrderNoInj[ClosestKpRow,:],KpResultsPerOrder[ClosestKpRow,:], SignalRegionIndex = SignalRegionIndices)
                    
                    if WriteWeights:
                        plt.title('Order %d CCF slice at best Kp of %f\nDet sig %.3e, assigned weight %.3e (%d lines)\nMeasured from plotted: SNR %.3e, S %.3e, N %.3e'%(OrderIndex,KpVect[ClosestKpRow],DetSigFromInjectedSignal[0],WeightFromInjectedSignalPerOrder[0],NumberOfLinesPerOrderList[OrderIndex],SNRValuesForTitle[0],SNRValuesForTitle[1],SNRValuesForTitle[2]))
                    if not WriteWeights:
                        plt.title('Order %d CCF slice at best Kp of %f '%(OrderIndex,KpVect[ClosestKpRow]))
                    
                    plt.tight_layout()        
                    pdf.savefig()     
                    plt.close()
                    
                    
                    #################################
                    plt.figure()
                    plt.title('No injected signal used for noise Order %d'%(OrderIndex))
                    plt.imshow(KpResultsPerOrderNoInj[:,ImShowXLimits[0]:ImShowXLimits[1]],interpolation='None',origin='lower',aspect='auto',extent=(KpVsysXaxis[0],KpVsysXaxis[-1],KpVect[0],KpVect[-1]))
                    plt.xlabel('Systemic velocity (km s$^{-1}$)')
                    plt.ylabel(r'K$_p$ (km s$^{-1}$)')        
                    plt.colorbar()
                    plt.plot([KpVsysXaxis[0],KpVsysXaxis[-1]],[RealKp,RealKp],'w:')
                    plt.plot([BestSys,BestSys],[KpVect[0],KpVect[-1]],'w:')
                    #plt.xlim(RadVPlotLimits[0],RadVPlotLimits[-1])
                    pdf.savefig()
                    plt.close()
                    
                    plt.figure()
                    plt.plot(KpVsysXaxis,KpResultsPerOrderNoInj[ClosestKpRow,ImShowXLimits[0]:ImShowXLimits[1]])
                    plt.xlabel('Systemic velocity (km s$^{-1}$)')
                    plt.ylabel('CCF')
                    #plt.plot([BestSys,BestSys],[-0.001,1.2*np.max(SelectiveAverage[ClosestKpRow,:])],'k:')
                    #plt.xlim(RadVPlotLimits[0],RadVPlotLimits[-1])
                    
                    
                    if WriteWeights:
                        plt.title('No injected signal used for noise\nOrder %d CCF slice at best Kp of %f\nDet sig %f, assigned weight %f\nMeasured from plotted: SNR %.5f, S %.5f, N %.5f'%(OrderIndex,KpVect[ClosestKpRow],DetSigFromInjectedSignal[0],WeightFromInjectedSignalPerOrder[0],SNRValuesForTitle[0],SNRValuesForTitle[1],SNRValuesForTitle[2]))
                    if not WriteWeights:
                        plt.title('No injected signal used for noise Order %d CCF slice at best Kp of %f '%(OrderIndex,KpVect[ClosestKpRow]))
                    
                    plt.tight_layout()        
                    pdf.savefig()     
                    plt.close()
                
                #############################
                
                
            print()
            print('!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Order %d, det sig %s'%(OrderIndex,MeasureDetectionSNR(KpResultsPerOrderNoInj[ClosestKpRow,:],KpResultsPerOrder[ClosestKpRow,:], SignalRegionIndex = SignalRegionIndices)))
            print('!!!!!!!!!!!!!!!!!!!!!!!!')

EndTime = time.time()

print()
print('Elapsed time: %f hours'%((EndTime-StartTime)/3600))