# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:03:45 2017

@author: user
"""
#Quesion 1////////////////////////////////////////////////////////////////

import numpy as np   
import matplotlib.pyplot as plt
import matplotlib.pylab as matplt
import scipy.signal as signal

data=np.loadtxt('ecg_180sec.dat')
data_time=data[:,0]
value1=data[:,1]
value2=data[:,2]
value3=data[:,3]
#比較maximum大小
if (np.max(value1)>np.max(value2)):
    if(np.max(value1)>np.max(value3)):
        value=value1
    else:
        value=value3
else:
    if(np.max(value2)>np.max(value3)):
        value=value2
    else:
        value=value3
#Question 1
#manually detect
#From the previous question
#Convert it into mV
for i in range(len(value)):
    value[i]=(value[i]*2-4096)/2000
#use the first question's bandstop
fs=1000
M=200
k0=int(0.5/fs*M)
k1=int(45/fs*M)
k2=int(55/fs*M)
X=np.ones(M)
#highpass filter to get rid the baseline
X[0:k0+1]=0
X[M-k0:M+1]=0
X[k1:k2+1]=0
X[M-k2:M-k1+1]=0
x=np.fft.ifft(X)
x=np.real(x)
#switch the position of the x
h2=np.zeros(M)
h2[0:int(M/2)]=x[int(M/2):M]
h2[int(M/2):M]=x[0:int(M/2)]
#plt.plot(h2)
value_filtered=signal.lfilter(h2,1,value)
#filter it
template=value_filtered[51700:52380]
fir_coeff=template[::-1]
det=signal.lfilter(fir_coeff,1,value_filtered)
det=det*det
#cutoff the unwanted signal at the start
det_cutoff=det[3000:len(det)]
#plt.plot(det_cutoff)
#set the threshold manually first, since heart is around 3.33Hz to 0.5 Hz, fs is 1000Hz.
#detect is for detecting the start and finish time 
#restriction to drop unwanted noise,index must exceed 300 and lower than 2000
threshold=150
time=np.array([])
detect=0
pre_centertime=[0]
for i in range(len(det_cutoff)):
    if(det_cutoff[i]>threshold and detect==0):
        starttime=i
        detect=1
    if(det_cutoff[i]<threshold and detect==1):
        finishtime=i
        detect=0
        centertime=[int((starttime+finishtime)/2)]
        #add restriction
        if((centertime[0]-pre_centertime[0])>300 and (centertime[0]-pre_centertime[0])<2000):
            time=np.concatenate((time,centertime))
            pre_centertime=centertime
#Then calculate the distance of the two points to get the duration, then turn it into bpm
bpm=np.array([])
for i in range(len(time)-1):
    temp=[fs*60/(time[i+1]-time[i])]
    bpm=np.concatenate((bpm,temp))
#plt.plot(bpm)
#establish the step bpm:time_sec
step_bpm=np.zeros(len(det_cutoff))
checkpoint=0       
for i in range(len(det_cutoff)):
    if (i<time[0]):
        step_bpm[i]=bpm[0]
    elif(i>=time[len(time)-1]):
        step_bpm[i]=bpm[len(bpm)-1]
    elif(i==time[checkpoint]):
        step_bpm[i]=bpm[checkpoint]
        checkpoint=checkpoint+1
    else:
        step_bpm[i]=bpm[checkpoint-1]
time_sec=data_time[3000:len(data_time)]/fs
matplt.title('momentary heart rate')
matplt.xlabel('time(sec)')
matplt.ylabel('heart rate')
plt.plot(time_sec,step_bpm)



        
    
        
        
    
        





























