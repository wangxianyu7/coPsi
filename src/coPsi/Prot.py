#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:00:23 2022

@author: emil
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from astropy.timeseries import LombScargle
import statsmodels.api as sm
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle
from .Phot import Data

class Rotator(Data):
	
	def __init__(self,file=None,x=np.array([]),y=np.array([]),dy=None,cadence=None):
		Data.__init__(self,file,x,y,dy,cadence)
  		#self.x = x
		#self.y = y
		#self.dy = dy
		#self.cadence = cadence


	def ACF(self,lags=np.array([]),**kwargs):
		
		if not len(lags):
			lags = np.arange(0,len(self.x)/2,1)
		
		self.acf = sm.tsa.acf(self.y, nlags = len(lags)-1,**kwargs)
		self.tau = lags*self.cadence

	def smoothACF(self,window=401,poly=3):
		self.acf = savgol_filter(self.acf,window,poly)



	def periodogram(self,samples_per_peak=15,**kwargs):
		
		LS = LombScargle(self.tau,self.acf)
		self.frequency, self.power = LS.autopower(samples_per_peak=samples_per_peak,**kwargs)
		
	def gauss(self,x,a,x0,sigma):
		return a*np.exp(-(x-x0)**2/(2*sigma**2))

	def fitProt(self,p0=[],noise=0.2,print_pars=True):

		
		time = 1/self.frequency
		power = self.power.copy()		

		if not len(p0):
			real = time > noise
			idx = np.argmax(power[real])
			power /= power[idx]
			mu = time[idx]
			p0 = [1.0,mu,0.1]
		
		popt,pcov = curve_fit(self.gauss,time,power,p0=p0)
		if print_pars:
			print(popt)
		self.fitPars = popt
		self.fitCov = pcov
		#return popt

	def getProt(self,prep=True,timeWindow=8001,timePoly=3,
				gap=0.5,yfill=None,cadence=None,smooth=True,
				freqWindow=401,freqPoly=3):
		
		if prep:
			self.prepData(window=timeWindow,poly=timePoly,gap=gap,yfill=yfill,cadence=cadence)
		self.ACF()
		if smooth:
			self.smoothACF(window=freqWindow,poly=freqPoly)
		self.periodogram()
		self.fitProt()


if __name__ == '__main__':
	dat = np.loadtxt('phot/KELT-11_cad_120sec_rotation.txt')
	x, y = dat[:,0], dat[:,1]
	
	yhat = filterData(y)
	nx, ny = fillGaps(x,yhat)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(x-min(x),y,color='C1',marker='.',ls='none')
	ax.plot(nx,ny,color='C0',marker='.',ls='none')
	
	cc, tt = ACF(ny,cadence=getCadence(nx))
	sc = filterData(cc,401,return_trend=1)
	
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(tt,cc,color='k',alpha=0.2)
	ax.plot(tt,sc,color='k')
	
	ff, pp = periodogram(tt,sc)
	pars = getProt(ff,pp)
	yy = gauss(1/ff,*pars)#*np.amax(power)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(1/ff,pp)
	ax.plot(1/ff,yy)
	ax.set_xlim(0,max(tt))
	
