# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 09:49:13 2017

@author: user
"""
#Quesion 1////////////////////////////////////////////////////////////////
import numpy as np   
import matplotlib.pyplot as plt
import matplotlib.pylab as matplt

data=np.loadtxt('ecg_5sec.dat')
time=data[:,0]
value1=data[:,1]
value2=data[:,2]
value3=data[:,3]
#compare maximum
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
#first, regain the the amplified voltage
value_mV=np.zeros(len(value))
for i in range(len(value)):
    value_mV[i]=(value[i]*2-4096)/2000
#print(value_mV)
#Fs=1000Hz,use Inkscape to plot vector pic
time_ms=np.linspace(0,len(value_mV),len(value_mV))
plt.subplot(311)
#plt.figure(1)
matplt.title('Time Domain')
matplt.xlabel('time(ms)')
matplt.ylabel('amplitude(mV)')
plt.plot(time_ms,value_mV)
#Fourier transform
"""
valuef=np.fft.fft(value_mV)
faxis=np.linspace(0,1000,len(valuef))
matplt.title('Frequency Domain')
matplt.xlabel('frequency(Hz)')
matplt.ylabel('amplitude')
plt.plot(faxis,abs(valuef))
"""
#Question 2/////////////////////////////////////////////////////////////////
#use class to create the definition of fir
class FIR_filter:
    def __init__(self,coefficients,buffer):
        self.myCoeff=coefficients  
        self.buffer=buffer
    def filter(self,v):
        #taps=numbers of coefficients
        #v is the real time lastest signal
        #shift the delay time
        index = len(self.myCoeff)-1
        while index > 0:
            self.buffer[index]=self.buffer[index-1]
            index=index-1
        self.buffer[0]=v
        output=0
        for i in range(len(self.myCoeff)):
            output=self.buffer[i]*self.myCoeff[i]+output
        return output
#Question 3/////////////////////////////////////////////////////////////////
#use the tutorial method to find the coefficients 
#Calculate the coefficients of an FIR filter analytially
#First set the M taps as 200, same as the tutorial,use sinc function to construct h1
#in Python,np.sinc(x)=sin(np.pi*x)/np.pi*x
M=200
fs=1000
f1=45/fs
f2=55/fs
n=np.arange(int(-M/2),int(M/2),1)
h1=2*f1*np.sinc(2*f1*n)-2*f2*np.sinc(2*f2*n)
h1[int(M/2)]=1-(2*np.pi)*(f2-f1)/(np.pi)
#establish buffer
buffer=np.zeros(M)
FIR=FIR_filter(h1,buffer)
#filter it
output1=np.zeros(len(value_mV))
for i in range(len(value_mV)):
    output1[i]=FIR.filter(value_mV[i])
plt.subplot(312)
#plt.figure(2)
matplt.title('50Hz norch with sinc')
matplt.xlabel('time(ms)')
matplt.ylabel('amplitude(mV)')
plt.plot(time_ms,output1)
#Question 4/////////////////////////////////////////////////////////////////    
#remove the baseline shift and 50Hz of the ECG
#https://martinos.org/mne/stable/auto_tutorials/plot_background_filtering.html#high-pass-problems
#try the 1 Hz highpass filter to get rid the baseline
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
#plt.plot(h), h is correct
#establish buffer
buffer=np.zeros(M)
FIR=FIR_filter(h2,buffer)
#filter it
output2=np.zeros(len(value_mV))
for i in range(len(value_mV)):
    output2[i]=FIR.filter(value_mV[i])
plt.subplot(313)
#plt.figure(3)
matplt.title('50 norch + baseline correction')
matplt.xlabel('time(ms)')
matplt.ylabel('amplitude(mV)')
plt.plot(time_ms,output2)












