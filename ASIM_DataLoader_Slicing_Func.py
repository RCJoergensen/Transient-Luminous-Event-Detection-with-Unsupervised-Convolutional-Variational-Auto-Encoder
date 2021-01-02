#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  13 10:49:00 2020

@author: rcj
"""
import os
from ReadMMIAfits_OCRE import Read_MMIA_Observation
import torch
import numpy as np

def npToTensor(npList):
    return torch.from_numpy(np.flip(npList,axis=0).copy())

def ASIM_DataLoader(dataDir, loadStatusUpdate,minval, maxval, p, sliceSize):
    if ('P1D' and 'P2D' and 'P3D') not in locals():
        ErroneousFiles = []
        workingFiles = []
        fileList = []
        for root, dirs, files in os.walk(dataDir):
            fileList.append([os.path.join(root, x) for x in files]) #Get all file paths
        
        fileListFlatten = [item for items in fileList for item in items] # Flatten the list to 1D list
        fileListFlatten = np.random.choice(fileListFlatten, int(len(fileListFlatten)*p), replace=False)
        fileListFlatten = [str(x) for x in fileListFlatten]
        # Define empty data vecors for each photometer
        P1D = []
        P2D = []
        P3D = []
        Time = []
        #Obs_ID = []
        #FrameTime = []
        for count, file in enumerate(fileListFlatten):
            if count % loadStatusUpdate == 0: # Print loading progress.
                print('Files loaded: ',count, '/', len(fileListFlatten))
            try:    # Try to load the data
            # RCJ Note: ASIM level 1 raw data is not "nice" to read. Instead of using Pytorch build in
            # load() function, we will use OC's load function and simply only save what we need.
                Observation_ID,Dat,Dat0,FrameTimeC,FrameTimeP,TimeError,Nframe,P1Exist,P2Exist,P3Exist,CHU1Meta_exists,CHU2Meta_exists,CHU1Exist,CHU2Exist, \
                DPU_Count,DPU_PreReset,Priority,Frame_Number,Priority,Frame_Number,PHOT1_Trig,PHOT2_Trig,PHOT3_Trig,CHU1_TrigR,CHU1_TrigC,CHU1_Trig, \
                CHU2_TrigR,CHU2_TrigC,CHU2_Trig,MXT,T,TT,BT,PHOT1_Data,PHOT2_Data,PHOT3_Data,PhotSizeTot,PhotSize,mR,MR,mC,MC,CHU1Meta,CHU2Meta, \
                CHU1,CHU2,LON,LAT,ALT,PosX,PosY,PosZ,PosVX,PosVY,PosVZ,PosYaw,PosPitch,PosRoll,CHU1_row_peak_value,CHU1_row_integral_value, \
                CHU1_column_peak_value,CHU1_column_integral_value,CHU2_row_peak_value,CHU2_row_integral_value,CHU2_column_peak_value, \
                CHU2_column_integral_value,PHOT1_peak_value,PHOT1_integral_value,PHOT2_peak_value,PHOT2_integral_value,PHOT3_peak_value, \
                PHOT3_integral_value,PHOT1_temp,PHOT2_temp,PHOT3_temp,CHU1_temp,CHU2_temp,CHU1_20v_45v,CHU2_20v_45v = \
                Read_MMIA_Observation(file,False,1) 
                workingFiles.append(fileListFlatten[count])
    
                if(P1Exist): #If the datafile contains photometer data, store it
          #          Obs_ID.append(Observation_ID[0])
                    currentSlice = 0
                    previousSlice = 0
                    while currentSlice < len(PHOT1_Data) - sliceSize:
                        currentSlice += sliceSize
                        P1D.append(npToTensor(PHOT1_Data[previousSlice:currentSlice]))
                        P2D.append(npToTensor(PHOT2_Data[previousSlice:currentSlice]))
                        P3D.append(npToTensor(PHOT3_Data[previousSlice:currentSlice]))
                        Time.append(npToTensor(T[previousSlice:currentSlice]))
                        previousSlice = currentSlice
         #           FrameTime.append(FrameTimeP)
            # Some files appear to be erroneous, skip these. Create list with names for further investigation
            # RCJ Personal note: Ask OC what is going on with erroneous files.
            except: 
                #print('Erroneous dataset:', fileListFlatten[count])
                ErroneousFiles.append(fileListFlatten[count])
    else:
        print('Files already loaded, skipping reloading')
    
    class Dataset(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, P1D, P2D, P3D, Time):
            'Initialization'
            #self.Obs_ID = npToTensor(Obs_ID)
            self.P1D = P1D
            self.P2D = P2D
            self.P3D = P3D
            self.Time = Time
            #self.FrameTime = FrameTime
    
    
      def __len__(self):
            'Denotes the total number of samples'
            return len(self.P1D)
    
      def __getitem__(self, idx):
            'Generates one sample of data'
            # Select sample
            #ID = self.Obs_ID[idx]
            P1 = self.P1D[idx]
            P2 = self.P2D[idx]
            P3 = self.P3D[idx]
            Time = self.Time[idx]
            #FrameTime = self.FrameTime[idx]
            #sample = {'P1D': P1, 'P2D': P2, 'P3D': P3, 'Time': Time}
            return P1, P2, P3, Time
        
    class NormalizedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, minval, maxval):
            super().__init__()
            self.dataset = dataset
            self.minval = minval
            self.maxval = maxval
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            return torch.clamp((self.dataset[idx] - self.minval) / (self.maxval - self.minval), 0.000001, 0.99999)

    CollectedData = Dataset(NormalizedDataset(P1D,minval,maxval), NormalizedDataset(P2D,minval,maxval), NormalizedDataset(P3D,minval,maxval), Time)
    return CollectedData