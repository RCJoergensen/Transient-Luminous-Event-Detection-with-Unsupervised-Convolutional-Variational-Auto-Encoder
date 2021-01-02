#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""ReadMMIAFits.py: ReadMMIAFits"""
__author__      = "Olivier Chanrion"
__copyright__   = "Copyright 2018, ASDC, DTU Space"
__credits__     = ["NumPy, MatplotLib, VisVis, Datetime, Spacepy, Lightningplot"]
__license__     = "XXX"
__version__     = "0.0.2"
__maintainer__  = "Olivier Chanrion"
__email__       = "chanrion@space.dtu.dk"
__status__      = "Working version"

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from PIL import Image
from   astropy.io import fits
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import get_sun,get_moon
#from ISS_Orbit_Predict import ISS_Pos
import scipy.signal as sig
import pyproj
from sgp4.earth_gravity import wgs84
# RCJ Note: Please redirect to your own share folder. A problem with the latest pyproj update has made it invisible for some reason
# A manual redirection appears to fix it, like shown below.
os.environ['PROJ_LIB'] = r'C:\Users\rasse\Anaconda3\pkgs\proj-7.2.0-h3e70539_0\Library\share'
from mpl_toolkits.basemap import Basemap
import time as ti

def Read_MMIA_Observation(filename,Frame_Time_Correction,level):

    ecefP = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    llaP = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    hdul=fits.open(filename)
    
    A=hdul[1]
    for i in np.linspace(1,np.size(A.columns[:]),np.size(A.columns[:]),dtype=int):
    # PHOT1Data_exists, PHOT2Data_exists, PHOT3Data_exists, CHU1Meta_exists, CHU1Data_exists, 
    # CHU2Meta_exists, CHU2Data_exist
        if (A.columns[i-1].name=='PHOT1DataTrigger'):
            trig=True
        #print(A.columns[i-1].name)

    data=hdul[1].data
    P1Exist=data.field('PHOT1Data_exists')
    
    P1Exist=P1Exist[0]
    CHU1Meta_exists=data.field('CHU1Meta_exists')
    CHU1Meta_exists=CHU1Meta_exists[0]
    CHU1Exist=data.field('CHU1Data_exists')
    CHU1Exist=CHU1Exist[0]
    P2Exist=data.field('PHOT2Data_exists')
    P2Exist=P2Exist[0]
    CHU2Meta_exists=data.field('CHU2Meta_exists')
    CHU2Meta_exists=CHU2Meta_exists[0]
    CHU2Exist=data.field('CHU2Data_exists')
    CHU2Exist=CHU2Exist[0]
    P3Exist=data.field('PHOT3Data_exists')
    P3Exist=P3Exist[0]
    
    Observation_ID=data.field('observation_id')
    Nframe=Observation_ID.size

    DPU_Count=data.field('dpu_counter_current')
    DPU_PreReset=data.field('dpu_timer_pre_reset')

    if (trig):
        Priority=data.field('priority')
            
    Frame_Number=data.field('frame_number')
            
    if (trig):
        CHU1_TrigR=data.field('CHU1RowMetaTrigger')
        CHU1_TrigC=data.field('CHU1ColumnMetaTrigger')
        CHU1_Trig=CHU1_TrigR & CHU1_TrigC

    if (trig):
        CHU2_TrigR=data.field('CHU2RowMetaTrigger')
        CHU2_TrigC=data.field('CHU2ColumnMetaTrigger')
        CHU2_Trig=CHU2_TrigR & CHU2_TrigC
            
    if (trig):
        MXT=data.field('MXGSTrigger')
        
    if (trig):
        PHOT1_Trig=data.field('PHOT1DataTrigger')
        PHOT2_Trig=data.field('PHOT2DataTrigger')
        PHOT3_Trig=data.field('PHOT3DataTrigger')
                
    Dat0=data.field('raw_datetime')
    if (Frame_Time_Correction):
        Dat0=datetime.datetime.strptime(Dat0[0],'%Y-%m-%dT%H:%M:%S.%f+00:00')-datetime.timedelta(seconds=50/1000000.0*np.floor(DPU_Count[0]/83250.0));#-datetime.timedelta(hours=2);         
    #elif Frame_Time is None:
        #Dat0 = datetime.datetime.strptime(Dat0[0], '%s')
    else:
        Dat0=datetime.datetime.strptime(Dat0[0],'%Y-%m-%dT%H:%M:%S.%f+00:00');
    
    if (level==1):
        Dat=data.field('corrected_datetime_level1')
        if (Frame_Time_Correction):
            Dat=datetime.datetime.strptime(Dat[0],'%Y-%m-%dT%H:%M:%S.%f+00:00')-datetime.timedelta(seconds=50/1000000.0*np.floor(DPU_Count[0]/83250.0));#-datetime.timedelta(hours=2);
        else:
            Dat=datetime.datetime.strptime(Dat[0],'%Y-%m-%dT%H:%M:%S.%f+00:00')
    else:
        Dat=Dat0

    TimeError=Dat-Dat0
    
    LON=data.field('longitude')
    LAT=data.field('latitude')
    ALT=np.zeros(LAT.shape)
    
# Put comment on CTRS (Earth Centered Earth Fixed)
                        
    PosX=data.field('bad_gnc_ctrs_pos_x')*0.3048
    PosY=data.field('bad_gnc_ctrs_pos_y')*0.3048
    PosZ=data.field('bad_gnc_ctrs_pos_z')*0.3048
        
    PosVX=data.field('bad_gnc_ctrs_velocity_x')*0.3048
    PosVY=data.field('bad_gnc_ctrs_velocity_y')*0.3048
    PosVZ=data.field('bad_gnc_ctrs_velocity_z')*0.3048
        
    PosYaw=data.field('iss_yaw')[0]
    PosPitch=data.field('iss_pitch')[0]
    PosRoll=data.field('iss_roll')[0]
    
#   FrameTime is the level_1 time corresponding to the center of the
#   timeInterval coresponding to the frame. 
    if (level==1):
        if (P1Exist):
            FrameTimP=data.field('frame_time_phot')

            FrameTimeP=[]
            for i in range(0,Nframe):
                if (Frame_Time_Correction):
                    FrameTimeP.append(datetime.datetime.strptime(FrameTimP[i],'%Y-%m-%dT%H:%M:%S.%f+00:00')-datetime.timedelta(seconds=50/1000000.0*np.floor(DPU_Count[0]/83250.0)))
                else:
                    FrameTimeP.append(datetime.datetime.strptime(FrameTimP[i],'%Y-%m-%dT%H:%M:%S.%f+00:00'))
        else:
            FrameTimeP=Dat0 #??? 
        
        if (CHU1Exist):
            FrameTimC=data.field('frame_time_chu')
            
            FrameTimeC=[]
            for i in range(0,Nframe):
                if (Frame_Time_Correction):
                    FrameTimeC.append(datetime.datetime.strptime(FrameTimC[i],'%Y-%m-%dT%H:%M:%S.%f+00:00')-datetime.timedelta(seconds=50/1000000.0*np.floor(DPU_Count[0]/83250.0)))
                else:
                    FrameTimeC.append(datetime.datetime.strptime(FrameTimC[i],'%Y-%m-%dT%H:%M:%S.%f+00:00'))
        else:
            FrameTimeC=Dat0 #??? 
        
#   Check for level 0, is frame_time there ? NO!
    else:
        FrameTimeP=Dat0 #??? 
        FrameTimeC=Dat0 #??? 

        PosX[i]=PosX[i]+(FrameTimeC[i]-Dat).total_seconds()*PosVX[0]
        PosY[i]=PosY[i]+(FrameTimeC[i]-Dat).total_seconds()*PosVY[0]
        PosZ[i]=PosZ[i]+(FrameTimeC[i]-Dat).total_seconds()*PosVZ[0]
        LON[i],LAT[i],ALT[i]= pyproj.transform(ecefP,llaP,PosX[i],PosY[i],PosZ[i],radians=False)

    TT=[]

    TT=np.zeros((3*(Nframe+1)),dtype='double');
    BT=np.zeros((3*(Nframe+1)),dtype='double');
        
    if (P1Exist):
        if (level==1):
# Read data in (mu) W / m^2
            PHOT1_Data=np.array(data.field('PHOT1_photon_flux'))#*1e6
            PHOT2_Data=np.array(data.field('PHOT2_photon_flux'))#*1e6
            PHOT3_Data=np.array(data.field('PHOT3_photon_flux'))#*1e6
        else:
            PHOT1_Data=np.array(data.field('PHOT1Data'))
            PHOT2_Data=np.array(data.field('PHOT2Data'))
            PHOT3_Data=np.array(data.field('PHOT3Data'))
            
        PhotSize=[]
        CumSize=[0]
        PhotSizeTot=0
        for i in range(0,Nframe):
            PhotSize.append(PHOT1_Data[i].shape[0])
            PhotSizeTot=PhotSizeTot+PhotSize[i]
            CumSize.append(PhotSizeTot)

        T=np.array(range(0,PhotSizeTot))/100.0 # to ms

        TT=np.zeros((3*(Nframe+1)),dtype='double');
        BT=np.zeros((3*(Nframe+1)),dtype='double');
        ST=0;
        TT[0]=ST/100.0;
        TT[1]=ST/100.0;
        TT[2]=ST/100.0;
                
        BT[0]=0;
        BT[1]=1.0;
        BT[2]=0;
        for ii in np.linspace(1,Nframe,Nframe,dtype='int'):
            ST=ST+PhotSize[ii-1];
            TT[3*ii]=ST/100.0;
            TT[3*ii+1]=ST/100.0;
            TT[3*ii+2]=ST/100.0;
            BT[3*ii]=0;
            BT[3*ii+1]=1.0;
            BT[3*ii+2]=0;

        PP1=np.zeros((np.sum(PhotSize)),dtype=PHOT1_Data[0].dtype)
        PP2=np.zeros((np.sum(PhotSize)),dtype=PHOT2_Data[0].dtype)
        PP3=np.zeros((np.sum(PhotSize)),dtype=PHOT3_Data[0].dtype)
        for i in range(0,Nframe):
            PP1[CumSize[i]:CumSize[i+1]]=PHOT1_Data[i]
            PP2[CumSize[i]:CumSize[i+1]]=PHOT2_Data[i]
            PP3[CumSize[i]:CumSize[i+1]]=PHOT3_Data[i]
        PHOT1_Data=PP1
        PHOT2_Data=PP2
        PHOT3_Data=PP3
    else:
        T=np.array([])
        PHOT1_Data=[]
        PHOT2_Data=[]
        PHOT3_Data=[]
        PhotSizeTot=0
        PhotSize=0

    if (not(trig)):
        mR=0
        MR=1055
        mC=0
        MC=1025
    if (trig):
        mR=data.field('chu_minimum_row')
        MR=data.field('chu_maximum_row')
        mC=data.field('chu_minimum_column')
        MC=data.field('chu_maximum_column')

    if (CHU1Meta_exists):
        #print 'CHU Meta data'
        if (level==1):
            CHU1Meta=np.array(data.field('CHU1Meta_photon_flux'))
            CHU2Meta=np.array(data.field('CHU2Meta_photon_flux'))
        else:
            CHU1Meta=np.array(data.field('CHU1Meta'))
            CHU2Meta=np.array(data.field('CHU2Meta'))
    else:
        CHU1Meta=[]
        CHU2Meta=[]

    #print 'TEST',CHU1Exist,data.field("CHU1Data_exists")
    if (CHU1Exist):
        if (level==1):
            CHU1=np.array(data.field('CHU1_photon_flux'))
            CHU2=np.array(data.field('CHU2_photon_flux'))
        else:
            CHU1=np.array(data.field('CHU1Data'))
            CHU2=np.array(data.field('CHU2Data'))
    else:
        CHU1=[]
        CHU2=[]

    if (trig):
        
        CHU1_row_peak_value=data.field('CHU1_row_peak_value')
        CHU1_row_integral_value=data.field('CHU1_row_integral_value')
        CHU1_column_peak_value=data.field('CHU1_column_peak_value')
        CHU1_column_integral_value=data.field('CHU1_column_integral_value')
        
        CHU2_row_peak_value=data.field('CHU2_row_peak_value')
        CHU2_row_integral_value=data.field('CHU2_row_integral_value')
        CHU2_column_peak_value=data.field('CHU2_column_peak_value')
        CHU2_column_integral_value=data.field('CHU2_column_integral_value')
        
        PHOT1_peak_value=data.field('PHOT1_peak_value')
        PHOT1_integral_value=data.field('PHOT1_integral_value')
                
        PHOT2_peak_value=data.field('PHOT2_peak_value')
        PHOT2_integral_value=data.field('PHOT2_integral_value')
            
        PHOT3_peak_value=data.field('PHOT3_peak_value')
        PHOT3_integral_value=data.field('PHOT3_integral_value')
        
        PHOT1_temp=data.field('phot1_temp')[0]
        PHOT2_temp=data.field('phot2_temp')[0]
        PHOT3_temp=data.field('phot3_temp')[0]
        CHU1_temp=data.field('chu1_temp')[0]
        CHU2_temp=data.field('chu2_temp')[0]
        CHU1_20v_45v=data.field('chu1_20v_45v')[0]
        CHU2_20v_45v=data.field('chu2_20v_45v')[0]

    MR=MR[0]
    mR=mR[0]
    MC=MC[0]
    mC=mC[0]
    
    return Observation_ID,Dat,Dat0,FrameTimeC,FrameTimeP,TimeError,Nframe,P1Exist,P2Exist,P3Exist,CHU1Meta_exists,CHU2Meta_exists,CHU1Exist,CHU2Exist,DPU_Count,DPU_PreReset,Priority,Frame_Number,Priority,Frame_Number,PHOT1_Trig,PHOT2_Trig,PHOT3_Trig,CHU1_TrigR,CHU1_TrigC,CHU1_Trig,CHU2_TrigR,CHU2_TrigC,CHU2_Trig,MXT,T,TT,BT,PHOT1_Data,PHOT2_Data,PHOT3_Data,PhotSizeTot,PhotSize,mR,MR,mC,MC,CHU1Meta,CHU2Meta,CHU1,CHU2,LON,LAT,ALT,PosX,PosY,PosZ,PosVX,PosVY,PosVZ,PosYaw,PosPitch,PosRoll,CHU1_row_peak_value,CHU1_row_integral_value,CHU1_column_peak_value,CHU1_column_integral_value,CHU2_row_peak_value,CHU2_row_integral_value,CHU2_column_peak_value,CHU2_column_integral_value,PHOT1_peak_value,PHOT1_integral_value,PHOT2_peak_value,PHOT2_integral_value,PHOT3_peak_value,PHOT3_integral_value,PHOT1_temp,PHOT2_temp,PHOT3_temp,CHU1_temp,CHU2_temp,CHU1_20v_45v,CHU2_20v_45v
