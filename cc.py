#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:54:23 2020

@author: ariddenharper
"""

import numpy as np 
import matplotlib.pyplot as plt 
import astropy.io.fits as pyfits 
from scipy import interpolate
import time
import os 
#from spectres import spectres 
from MakePlanetSpecOverStarSpec import MakePlanetOverStellarSpecFast,SubtractLowerEnvelope
from astropy import units as u
import pandas as pd
#from numba import jit 

StartTime = time.time()

#from PyAstronomy.pyasl import crosscorrRV 


def crosscor(a,v):
    '''
    A properly normalised version of the numpy cross-correlation function    
    '''

    a = (a - np.mean(a)) / (np.std(a) * len(a))
    v = (v - np.mean(v)) /  np.std(v)
    
    return np.correlate(a,v)

def LambertSphere(PhaseOffset,i=90*np.pi/180):    
    
    '''
    Taken from Herman et al. (2020), equation 1.
    For scaling the emission spectrum according to its orbital phase.
    Orbital phase 0 or 1 has scaling factor of 0 and orbital phase 0.5
    has scaling factor of 1.    
    
    '''
    
    Phase = 0
    
    cosz = -np.sin(i)*np.cos(2*np.pi*(Phase-PhaseOffset))
    
    z = np.arccos(cosz)   

    sinz = np.sin(z)
    
    Fp = (sinz + (np.pi - z)*cosz)/np.pi
    
    return Fp


# def SubtractEnvelopeFromBins(wave,depth,numbins=50,PolyOrder=2):
    
#     envFromBin = FindEnvelopeWithBins(wave,depth,numbins=50,PolyOrder=2)
    
#     return depth - envFromBin

def GetAllResiduals(TargetForCrossCor,night,arm,arm_subpart,ModelScalingFactor,PCA_algorithm,UpToComponents,SysremComponentsRemovedVect):    
    
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
        ##DataFile = '%s/Flux5b_cropped_%sTo%d_ColWeighted.fits'%(LoadDirectory,PCA_algorithm,UpToComponents)
        #DataFile = '%s/Flux5_BaryFrame_%sTo%d_ColWeighted.fits'%(LoadDirectory,PCA_algorithm,UpToComponents)       
        #DataFile = '%s/Flux5_Corrected_BaryFrame_sysremedTo%d_ColWeighted.fits'%(LoadDirectory,UpToComponents)       
        DataFile = '%s/Flux8a_TelluricFrame_sysremedTo%d_ColWeighted_var.fits'%(LoadDirectory,UpToComponents)       

           
        #DataFileWaveArray = '%s/CroppedWave.fits'%(LoadDirectory)
        #DataFileWaveArray = '%s/Wave_PostSysrem_ShiftedForBaryCorrection.fits'%(LoadDirectory)
        DataFileWaveArray = '%s/Wave4_ToSysrem.fits'%(LoadDirectory)
        
        
        residualsAllSysremsFile = pyfits.open(DataFile)
        
        residuals = residualsAllSysremsFile[SysremComponentsRemoved].data   
        
        ## Just in case some nans have slipped through
        residuals[np.isnan(residuals)] = 0.0
    
        ##pyfits.writeto('QuickOutput.fits',residuals,overwrite='True')      
    
        wave = pyfits.getdata(DataFileWaveArray)
        
        ListOfResiduals.append((wave[0,:],residuals))
        
        
    AllOutput = (AncillaryData,ListOfResiduals)        
        
    return AllOutput



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


FirstPartOfLoadPath = '../CrossCorrelationDataAndProcessing'
#FirstPartOfLoadPath = 'F:'


### First define what to load 

c=299792.458 #speed of light (km/s)

SystemicVelocity = -20.567  ## kms for KELT-9b from NASA Exoplanet Archive. Gaudi et al. 2017 
PlanetInclination = 87.2*np.pi/180   ## for KELT-9b, Ahlers et al. 2020 
Rplanet = 1.891*u.Rjup 
Rstar = 2.36*u.Rsun

RplanetOverRstarSquared = ((Rplanet/Rstar)**2).decompose().value


#DataOrigin = 'CARMENES'
#ModelToInject = 'H2O'
#ModelToInject = 'NH3'




DataOrigin = 'KELT-9b_CARMENES_emission_data'



# TargetForCrossCor = 'KELT9b_Al_0.50_+0.0_0.55_Vrot6.63'
# ### 300	120	75	450
# ##ModelScalingFactor = 75
# #ModelScalingFactorList = [300, 120, 75, 450]
# ModelScalingFactorList = [1,1,1,1]
# ModelForCrossCor = 'KELT9b_Al_0.50_+0.0_0.55_Vrot6.63'

# TargetForCrossCor = 'KELT9b_Al_0.50_+2.3_0.55_Vrot6.63'
# ModelScalingFactor = 1
# ModelForCrossCor = 'KELT9b_Al_0.50_+2.3_0.55_Vrot6.63'

# TargetForCrossCor = 'KELT9b_Al_0.50_+1.7_0.55_Vrot6.63'
# ModelScalingFactor = 1
# ModelForCrossCor = 'KELT9b_Al_0.50_+1.7_0.55_Vrot6.63'


# TargetForCrossCor = 'KELT9b_Ca_0.50_+0.0_0.55_Vrot6.63'
# ModelForCrossCor = TargetForCrossCor
# ##ModelScalingFactorList = [500, 130, 100, 600]
# ModelScalingFactorList = [1,1,1,1]



# TargetForCrossCor = 'KELT9b_CaII_0.50_+0.0_0.55_Vrot6.63'
# ModelForCrossCor = TargetForCrossCor
# #ModelScalingFactorList = [1000, 270, 200, 1000]
# ModelScalingFactorList = [1,1,1,1]
# #ModelScalingFactorList = [3,3,3,3]


# TargetForCrossCor = 'KELT9b_Cr_0.50_+0.0_0.55_Vrot6.63'
# #ModelScalingFactorList = [200, 50, 40, 200]
# ModelScalingFactorList = [1,1,1,1]
# ModelForCrossCor = TargetForCrossCor

# TargetForCrossCor = 'KELT9b_Fe_0.50_+0.0_0.55_Vrot6.63'
# ## 150,40,30,150
# #ModelScalingFactor = 30
# #ModelScalingFactorList = [3,3,3,3]
# ModelScalingFactorList = [150,40,30,150]
# ModelForCrossCor = TargetForCrossCor

# TargetForCrossCor = 'KELT9b_FeII_0.50_+0.0_0.55_Vrot6.63'
# ### 1000, 270, 200, 1000 ## an inital test 
# ### 1500, 400, 300, 1500
# #ModelScalingFactor = 300
# ##ModelScalingFactorList = [1500, 400, 300, 1500]
# ModelScalingFactorList = [1,1,1,1]
# ModelForCrossCor = TargetForCrossCor

# TargetForCrossCor = 'KELT9b_FeII_UsingFeI_0.50_+0.0_0.55_Vrot6.63'
# ### 1000, 270, 200, 1000 ## an inital test 
# ### 1500, 400, 300, 1500
# #ModelScalingFactor = 300
# ModelScalingFactorList = [1500, 400, 300, 1500]
# #ModelScalingFactorList = [1,1,1,1]
# ModelForCrossCor = TargetForCrossCor



# TargetForCrossCor = 'KELT9b_FeH_main_iso_0.50_+0.0_0.55_Vrot6.63'
# ### 750, 200, 150, 750
# ModelScalingFactorList = [750, 200, 150, 750]
# #ModelScalingFactorList = [1,1,1,1]
# ModelForCrossCor = TargetForCrossCor


# TargetForCrossCor = 'KELT9b_K_0.50_+0.0_0.55_Vrot6.63'
# ### 500, 130, 100, 500
# #ModelScalingFactorList = [500, 130, 100, 500]
# ModelScalingFactorList = [1,1,1,1]
# ModelForCrossCor = TargetForCrossCor




# TargetForCrossCor = 'KELT9b_Mg_0.50_+0.0_0.55_Vrot6.63'
# #ModelScalingFactorList = [400, 100, 80, 400]
# ModelScalingFactorList = [1,1,1,1]
# ModelForCrossCor = TargetForCrossCor


# TargetForCrossCor = 'KELT9b_Na_0.50_+0.0_0.55_Vrot6.63'
# ###  500, 130, 100, 500
# #ModelScalingFactorList = [500, 130, 100, 500]
# ModelScalingFactorList = [1, 1, 1, 1]
# ModelForCrossCor =  TargetForCrossCor



# TargetForCrossCor = 'KELT9b_Ti_0.50_+0.0_0.55_Vrot6.63'
# #ModelScalingFactorList = [150, 45, 25, 150]
# ModelScalingFactorList = [1,1,1,1]
# ModelForCrossCor = TargetForCrossCor





# TargetForCrossCor = 'KELT9b_TiO_all_iso_Plez_0.50_+0.0_0.55_Vrot6.63'
# ## 750, 195, 150, 750
# ## 1000, 300, 230, 1000  ## resulted in slightly worse detection with Sysrem config 1  
# ### 1600, 400, 300, 1800
# #ModelScalingFactorList = [750, 195, 150, 750]
# ModelScalingFactorList = [3,3,3,3]
# ModelForCrossCor = TargetForCrossCor





# TargetForCrossCor = 'KELT9b_TiO_48_Exomol_McKemmish_0.50_+0.0_0.55_Vrot6.63'
# #ModelScalingFactorList = [750, 195, 150, 750]
# #ModelScalingFactorList = [3,3,3,3]
# ModelScalingFactorList = [1,1,1,1]
# ModelForCrossCor = TargetForCrossCor


# TargetForCrossCor = 'KELT9b_TiO_48_Plez_0.50_+0.0_0.55_Vrot6.63'
# ModelScalingFactorList = [750, 195, 150, 750]
# ModelForCrossCor = TargetForCrossCor



TargetForCrossCor = 'NoModel'
ModelScalingFactorList = [0,0,0,0]
ModelForCrossCor = 'KELT9b_FeII_UsingFeI_0.50_+0.0_0.55_Vrot6.63'



AbsorptionOrEmission = 'Emission'


#arm = 'vis'
# arm = 'nir' 
# arm_subpart = 'B' ## A or B 
### ^^^
#NightList = ['20180609All']
#NightList = ['20180618All'[]
#NightList = ['20190528All']
#NightList = ['20190604All']  

###NightList = ['20180609All','20180618All','20190528All','20190604All']
NightList = ['20180609All','20180618All','20190528P2','20190604All']



##ArmSubpartList = [['vis','A'],['nir','A'],['nir','B']]
ArmSubpartList = [['vis','A'],['nir','A']]

#ArmSubpartList = [['vis','A']]
#ArmSubpartList = [['nir','A']]
#ArmSubpartList = [['nir','B']]

#ArmSubpartList = [['nir','A'],['nir','B']]

#
#ArmSubpartList = [['nir','A']]
#ArmSubpartList = [['nir','B']]

##################################
### Parameters for subtracting the continuum from the the injected spectrum 
        
### Old parameters used for all models 
# ModelSpectrumFlattening_PointsPerBin = 400
# ## InjectedSpectrumFlattening_PolyOrder = 50 ### good if fitting a minimum envelope over the entire spectral range but drastically overfits the basically straight line over a single order 
# InjectedSpectrumFlattening_PolyOrder = 2


if ModelForCrossCor == 'KELT9b_Cr_0.50_+0.0_0.55_Vrot6.63':
    ModelSpectrumFlattening_PointsPerBin = 400
    ModelSpectrumFlattening_PolyOrder = 2
    
    SecondarySubList = [['vis','A'],['nir','A'],['nir','B']]


elif ModelForCrossCor == 'KELT9b_Fe_0.50_+0.0_0.55_Vrot6.63':
    ModelSpectrumFlattening_PointsPerBin = 400
    ModelSpectrumFlattening_PolyOrder = 2
    
    SecondarySubList = []
    
elif ModelForCrossCor == 'KELT9b_Ti_0.50_+0.0_0.55_Vrot6.63':
    ModelSpectrumFlattening_PointsPerBin = 400
    ModelSpectrumFlattening_PolyOrder = 2
    
    SecondarySubList = [['vis','A'],['nir','A'],['nir','B']]        

    
elif ModelForCrossCor == 'KELT9b_TiO_all_iso_Plez_0.50_+0.0_0.55_Vrot6.63':
    ModelSpectrumFlattening_PointsPerBin = 50
    ModelSpectrumFlattening_PolyOrder = 10      
    
    SecondarySubList = [['nir','A'],['nir','B']]
    
elif ModelForCrossCor == 'KELT9b_TiO_48_Exomol_McKemmish_0.50_+0.0_0.55_Vrot6.63':
    ModelSpectrumFlattening_PointsPerBin = 50
    ModelSpectrumFlattening_PolyOrder = 10      
    
    SecondarySubList = [['nir','A'],['nir','B']]

elif ModelForCrossCor == 'KELT9b_TiO_48_Plez_0.50_+0.0_0.55_Vrot6.63':
    ModelSpectrumFlattening_PointsPerBin = 50
    ModelSpectrumFlattening_PolyOrder = 10      
    
    SecondarySubList = [['nir','A'],['nir','B']]    
    
    
else: 
    ModelSpectrumFlattening_PointsPerBin = 50
    ModelSpectrumFlattening_PolyOrder = 10   
    
    SecondarySubList = [['vis','A'],['nir','A'],['nir','B']]
    
#####################################################

#SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig1.csv')
#SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig2.csv')

#SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig3.csv')
    
#SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig4.csv')
    
SysremItConfig_df = pd.read_csv('SysremItsConfigFiles/CAMENES_KELT-9b_SysremItConfig6.csv')




# print('!!!!!!!!!!!!!!!!!!!!!!!!!')
# print('The keys')
# print(SysremItConfig_df.keys())
# print('!!!!!!!!!!!!!!!!!!!!!!!!!')


###ArmSubpartList = [['nir','A'],['nir','B']]

#for NightIndex in range(len(NightList)):
for NightIndex in [1,2]:
##for NightIndex in [2]:
    
    night = NightList[NightIndex]
    ModelScalingFactor = ModelScalingFactorList[NightIndex]

    for ArmSubpartIndex in range(len(ArmSubpartList)):
        
        arm = ArmSubpartList[ArmSubpartIndex][0]
        arm_subpart = ArmSubpartList[ArmSubpartIndex][1]
    
        FullOutput = True 
        
        
        if DataOrigin == 'CARMENES':
            RemoveCosmicsAndFlatten = False
            
            if arm == 'nir':
                FileNameString = 'nir_A'
            if arm == 'vis':
                FileNameString = 'vis_A'         
        
        if arm == 'nir':
            numorders = 28
            #NumberOfSysremIterationsVect = np.ones((numorders))*6
    
        if arm == 'vis':
            numorders = 61
            #NumberOfSysremIterationsVect = np.ones((numorders))*6
        
        ### ***
        #ListOfOrdersToDo = range(30,numorders)
        #ListOfOrdersToDo = range(50,numorders)
        #ListOfOrdersToDo = range(20,30)
        #ListOfOrdersToDo = range(0,30)
        #ListOfOrdersToDo = range(30,numorders)
            
        ListOfOrdersToDo = range(numorders)
        ## Sets the number of SYSREM iterations to use 
        NumberOfSysremIterationsVect = SysremItConfig_df['%s_%s_%s'%(night,arm,arm_subpart)].values        
        #NumberOfSysremIterationsVect = np.ones((numorders))*6
        
    
        ModelShift_kms = 0  ## Refers to the model scaling factor used when injecting the model spectrum 
        
        InputModelScalingFactor = 1  ## The scaling factor used for the cross-correlation template 
        
        #PCA_algorithm = 'pcasub'   
        PCA_algorithm = 'sysremed'
        #UpToComponents = 15
        UpToComponents = 10
        
        if TargetForCrossCor == 'NoModel':
            #UpToComponents = 30
            #UpToComponents = 20
            UpToComponents = 10
        
        #SysremComponentsToRemoveList = range(1,UpToComponents+1)
        
        #SysremComponentsRemoved = 15
        #SysremComponentsRemoved = 7
       
        
        #for SysremComponentsRemoved in SysremComponentsToRemoveList:
        #for SysremComponentsRemoved in [1,2]:
    
        #### End defining what to load 
        ####################    
        
        visA = GetAllResiduals(TargetForCrossCor,night,arm,arm_subpart,ModelScalingFactor,PCA_algorithm,UpToComponents,NumberOfSysremIterationsVect)
        
        # plt.figure()
        # plt.imshow(visA[1][0][1],interpolation='None',origin='lower',aspect='auto')
        # plt.title('residuals')
        # plt.colorbar()
        
        
        AvgBaryRV = np.mean(visA[0][2])
        #AvgPlanetRV = np.mean(visA[0][1])
        AvgPlanetRV = np.median(visA[0][1])
        
        
        PlanetPhase = visA[0][0]
        
        
        
        # BaseRV = SystemicVelocity - AvgBaryRV + AvgPlanetRV
        
        # startccrv = BaseRV-DesiredCrossCorRange_kms/2
        # endccrv = BaseRV+DesiredCrossCorRange_kms/2
        
        
        
        
        #NumRVpoints = 50
        #NumRVpoints = 10
        #RVVectForCrossCorrelation = np.linspace(startccrv,endccrv,NumRVpoints)
        #RVVectForCrossCorrelation = np.arange(startccrv,endccrv+1,1)
        
        # DesiredCrossCorRange_kms = 1000
        # RVVectForCrossCorrelation = np.arange(-DesiredCrossCorRange_kms/2,DesiredCrossCorRange_kms/2 + 1,2)
        
        DesiredCrossCorRange_kms = 2000
        RVVectForCrossCorrelation = np.arange(-DesiredCrossCorRange_kms/2,DesiredCrossCorRange_kms/2 + 1,1)
        
        # DesiredCrossCorRange_kms = 1200
        # RVVectForCrossCorrelation = np.arange(-DesiredCrossCorRange_kms/2,DesiredCrossCorRange_kms/2 + 1,1)
    
        
        radv_vect = RVVectForCrossCorrelation    
            
        if AbsorptionOrEmission == 'Emission':
            
            ####StellarWaveFlux = np.load('../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/ModelSpectra/FluxDensityPerHzStellarBlackBodySpec.npy')
            #StellarWaveFlux = np.load('../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/ModelSpectra/KELT-9b_R94600_vsini111.80_epsilon0.60_lte10200-4.00-0.0.PHOENIX_erg_per_s_per_cm2_per_Hz.npy')         
            #StellarWaveFlux = np.load('%s/KELT-9b_CARMENES_emission_data/ModelSpectra/BB_10170K_R1e6_erg_Per_s_Per_cm2_Per_Hz.npy'%(FirstPartOfLoadPath))         
            StellarWaveFlux = np.load('%s/KELT-9b_CARMENES_emission_data/ModelSpectra/BB_10170K_CarmenesRes_erg_Per_s_Per_cm2_Per_Hz.npy'%(FirstPartOfLoadPath))         
    
            
    
            
            if ModelForCrossCor == 'KELT9bEmission':
                PlanetWaveFlux = np.load('../CrossCorrelationDataAndProcessing/KELT-9b_CARMENES_emission_data/ModelSpectra/pRT_flux_per_Hz.npy')
                
            else:
                PlanetWaveFlux = np.load('%s/KELT-9b_CARMENES_emission_data/ModelSpectra/%s_CarmenesRes_pRT_flux_per_Hz.npy'%(FirstPartOfLoadPath,ModelForCrossCor))
    
                
            ### Just make sure they're sorted in order of increasing wavelength  
            PlanetWaveFlux = PlanetWaveFlux[np.argsort(PlanetWaveFlux[:,0])]
            StellarWaveFlux = StellarWaveFlux[np.argsort(StellarWaveFlux[:,0])]
                
            #### Convert the microns to A 
            PlanetWaveFlux[:,0] *= 1e4 
            StellarWaveFlux[:,0] *= 1e4
        
        test_residual = visA[1][0][1]
        
        numrows,numcols = np.shape(test_residual)
        
        CrossCorrResults = np.zeros((numrows,len(radv_vect),numorders))
        #CrossCorrResultsPerOrder = np.zeros((numrows,len(radv_vect)))
        
    
        
        ### Make a model stellar spectrum for all spectra by just using the average barycentric 
        ### velocity of the night. This is valid since it changes by fraction of a pixel. 
        NettStellarRV = SystemicVelocity - AvgBaryRV
        
        StellarSpecWave = StellarWaveFlux[:,0]
        StellarSpecFlux = StellarWaveFlux[:,1]
        
        ShiftedStellarSpecWave = StellarSpecWave*(1+(NettStellarRV/c))    
        StellarInterpObject = interpolate.interp1d(ShiftedStellarSpecWave, StellarSpecFlux, bounds_error=False, fill_value='extrapolate')
        
        PlanetSpecWave = PlanetWaveFlux[:,0]        
        BasePlanetSpecFlux = PlanetWaveFlux[:,1]*InputModelScalingFactor    
        PlanetInterpObject = interpolate.interp1d(PlanetSpecWave, BasePlanetSpecFlux, bounds_error=False, fill_value='extrapolate')
    
        
        for radvcount in range(len(radv_vect)):
            
            if (radvcount % 10) == 0:
                print('Doing RV index %d of %d'%(radvcount,len(radv_vect)))
        
            radv = radv_vect[radvcount]
            
            for OrderIndex in ListOfOrdersToDo:
            ##for OrderIndex in range(numorders):    
                
                NumberOfSysremIterations = int(NumberOfSysremIterationsVect[OrderIndex])           
                
                #print('Doing doing order index %d of %d (for RV index %d of %d) '%(OrderIndex,numorders,radvcount,len(radv_vect)))
                
                w = visA[1][OrderIndex][0]
                residuals = visA[1][OrderIndex][1]
                
                ShiftedPlanetFlux = PlanetInterpObject(w*(1-(radv/c))) ## 1- when using in the reverse way with a shifted observed wavelength scale   
                ShiftedStellarFlux = StellarInterpObject(w)  ## this shift is only done once so there won't be any advantage to putting it outside the loop like for the planet model           
                    
                ratio = (RplanetOverRstarSquared)*(ShiftedPlanetFlux/ShiftedStellarFlux)
                
                # QuickEmissionRatioOutput = np.empty((len(w),2))
                # QuickEmissionRatioOutput[:,0] = w
                # QuickEmissionRatioOutput[:,1] = ratio
                
                # np.save('QuickEmissionRatio.npy',QuickEmissionRatioOutput)
                
                ### Old flattening method
                ###FlattenedRatio = SubtractEnvelopeFromBins(w,ratio,50,2) + 1
                
                xIndexArray = np.arange(len(ratio))
                
                env,binmiddles,binnedmins = FindEnvelopeWithBins(xIndexArray,ratio,PointsPerBin=ModelSpectrumFlattening_PointsPerBin, numbins=None, PolyOrder=ModelSpectrumFlattening_PolyOrder)
    
                FlattenedRatio = ratio - env 
                
                if [arm,arm_subpart] in SecondarySubList:
                    FlattenedRatio -= np.sort(FlattenedRatio)[100]
                    
                FlattenedRatio += 1.0
                
                ##################################
                    
                ####FlattenedRatio = SubtractLowerEnvelope(ratio,'lower',100) + 1
                
                for SpecIndex in range(numrows):        
                    
                    # ShiftedPlanetFlux *= LambertSphere(PlanetPhase[SpecIndex],i=PlanetInclination)
                    
                    # ratio = 1 + (RplanetOverRstarSquared)*(ShiftedPlanetFlux/ShiftedStellarFlux)
                    
                    # FlattenedRatio = SubtractLowerEnvelope(ratio,'lower',100) + 1
                    
                    ResidualSpec = residuals[SpecIndex]
                    
                    CrossCorrResults[SpecIndex,radvcount,OrderIndex] = crosscor(FlattenedRatio,ResidualSpec)             
                    
        ####### Save the orders here 
        #XCorSaveDirectoryLoadDirectory = '../CrossCorrelationDataAndProcessing/%s/ProcessedData/%s/%s/%s/Part%s/ModelScalingFactor%0.5e/ModelShift_%.3e_kms/XCorr2/%s/%s_%d'%(DataOrigin,TargetForCrossCor,night,arm,arm_subpart,ModelScalingFactor,ModelShift_kms,ModelForCrossCor,PCA_algorithm,SysremComponentsRemoved)
        #XCorSaveDirectoryLoadDirectory = '%s/%s/XCorr/%s/%s/%s/Part%s/ModelScalingFactor%0.5e/ModelShift_%.3e_kms/XCorr2/%s/%s_%d'%(DataOrigin,TargetForCrossCor,night,arm,arm_subpart,ModelScalingFactor,ModelShift_kms,ModelForCrossCor,PCA_algorithm,SysremComponentsRemoved)
        
        for OrderIndex in ListOfOrdersToDo:
        #for OrderIndex in range(numorders): 
            
            NumberOfSysremIterations = int(NumberOfSysremIterationsVect[OrderIndex])   
        
            XCorSaveDirectoryLoadDirectory = '%s/KELT-9b_CARMENES_emission_data/XCorr/%s/%s/%s/ModelScalingFactor%0.5e/%s_%s/Order%d/Sysrem%d'%(FirstPartOfLoadPath,night,TargetForCrossCor,ModelForCrossCor,ModelScalingFactor,arm,arm_subpart,OrderIndex,NumberOfSysremIterations) 
            if not os.path.exists(XCorSaveDirectoryLoadDirectory):
                os.makedirs(XCorSaveDirectoryLoadDirectory)    
            
            np.savetxt('%s/%s_%d_XCorrRadVelVect.txt'%(XCorSaveDirectoryLoadDirectory,PCA_algorithm,NumberOfSysremIterations),radv_vect)            
            
            np.save('%s/Xcorr_order%d_Sysrem%d.npy'%(XCorSaveDirectoryLoadDirectory,OrderIndex,NumberOfSysremIterations),CrossCorrResults[:,:,OrderIndex])

#FirstPart/XCorr/night/target for CrossCor/Model for CrossCor/ModelScalingFactor%0.5e/arm_arm_subpart/order/





EndTime = time.time()
TotalTime_hours = (EndTime-StartTime)/3600

print('Time taken to Cross Correlate %d orders each with %d spectra with %d RVs: %f hours'%(numorders,numrows,len(radv_vect),TotalTime_hours))


            


