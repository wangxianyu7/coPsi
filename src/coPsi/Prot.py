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
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
from .Phot import Data

class Rotator(Data):
	'''Rotation from light curve

	This class inherits from :py:class:`coPsi.Data()` and is initialized in the same way.

	



	'''


	def __init__(self,file=None,x=np.array([]),y=np.array([]),dy=None,cadence=None):
		Data.__init__(self,file,x,y,dy,cadence)


	def ACF(self,lags=np.array([]),**kwargs):
		'''Autocorrelation function

		Use the ACF to find rotation period as in :cite:t:`McQuillan2013`.

		Calculated using :py:class:`statsmodels.api.tsa.acf()`.

		:param lags: Lags for which to calculate ACF. Optional, default `ǹumpy.array([])`.
		:type lags: array

	

		'''
		if not len(lags):
			lags = np.arange(0,len(self.x)/2,1)
		
		self.acf = sm.tsa.acf(self.y, nlags = len(lags)-1,**kwargs)
		self.tau = lags*self.cadence

	def smoothACF(self,window=401,poly=3):
		'''Smooth ACF
	
		Smoothes the ACF using a Savitsky-Golay filter from :py:class:`scipy.signal.savgol_filter()`.

		:param window: Length of window for filter. Optional, default 401.
		:type window: int

		:param poly: Degree for polynomial. Optional, default 3.
		:type poly: int	

		'''

		self.racf = self.acf
		self.acf = savgol_filter(self.acf,window,poly)



	def periodogram(self,samples_per_peak=15,**kwargs):
		'''Periodogram

		Calculated using :py:class:`astropy.timeseries.LombScargle()`.


		:param samples_per_peak: Number of samples (resolution). Optional, default 15.
		:type samples_per_peak: int

		'''
		LS = LombScargle(self.tau,self.acf)
		self.frequency, self.power = LS.autopower(samples_per_peak=samples_per_peak,**kwargs)
		
	def gauss(self,x,a,x0,sigma):
		'''Gaussian

		.. math:: 
			f (x) = \\frac{A}{\sqrt{2 \pi}\sigma} \exp \\left (-\\frac{(x - \mu)^2}{2 \sigma^2} \\right) \, .

		:param x: :math:`x`.
		:type x: float, array
		
		:param x0: :math:`\mu`.
		:type x0: float

		:param sigma: :math:`\sigma`.
		:type sigma: float

		:param a: :math:`A`.
		:type a: float


		:returns: :math:`f(x)`
		:rtype: float

		'''
		return a*np.exp(-(x-x0)**2/(2*sigma**2))

	def fitProt(self,p0=[],noise=0.2,print_pars=True):
		'''Fit rotation period.

		This function will fit a Gaussian to the periodogram to find the rotation period. Good guesses for the location helps.


		:param p0: Starting guesses for amplitude,location, and width ([:math:`A`, :math:`\mu`, :math:`\sigma`] in :py:class:`gauss`). Optional, default ``[]``. If empty will try to find where the maximum is.
		:type p0: list
	
		:param noise: Value in time (days/lags) under which to consider noise. Will search for peaks above noise for p0. Optional, default 0.2.
		:type noise: float

		:param print_pars: Whether to print the parameters from the fit. Optional, default ``True``.
		:type print_pars: bool

		'''
		
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
			print('Fit Gaussian to periodogram:')
			print('Prot = {:0.4f}+/-{:0.4f} d'.format(popt[1],popt[2]))
		#self.fitPars = popt
		#self.fitCov = pcov
		self.per = popt[1]
		self.sper = popt[2]
		#return popt

	def fromPeaks(self,prominence=(0.3,1.1),poly=False,plot=True,print_pars=True,font=12,usetex=False,**kwargs):
		'''Rotation period from ACF peaks

		Find peaks in ACF using :py:class:`scipy.signal.find_peaks()`. 

		Either following to :cite:t:`McQuillan2013` or :cite:t:`Hjorth2021`.

		:param prominence: The prominence of the peaks. Optional, default (0.3,1.1). See :py:class:`scipy.signal.find_peaks()`.
		:type prominence: tuple

		:param poly: If ``True`` peaks are found from linear fit :cite:p:`Hjorth2021`, else from median of differences :cite:p:`McQuillan2013`. Optional, default ``False``.
		:type poly: bool

		:param plot: Whether to plot ACF with peaks. Optional, default ``True``.
		:type plot: bool

		:param print_pars: Whether to print the parameters from the fit. Optional, default ``True``.
		:type print_pars: bool

		:param font: Fontsize for labels. Optional, default 12.
		:type font: float

		:param usetex: Whether to use LaTeX in plots. Optional, default ``False``.
		:type usetex: bool

		:params kwargs: Extra keywords are sent to :py:class:`scipy.signal.find_peaks()`.
		:type kwargs: dict


		'''
		
		peaks, _ = find_peaks(self.acf, prominence=prominence,**kwargs)
		peaks = peaks[:10] # Only take up to 10 peaks, see McQuilllan2013

		if plot:
			self.plotACF(peaks=peaks,usetex=usetex)

		if poly:
			pars, cov = np.polyfit(np.arange(1,len(peaks)+1),self.tau[peaks],1,cov=True)
			per = pars[0]
			sd = np.sqrt(np.diag(cov))[0]
			print('From linear fit and covariance:')
			if plot:
				plt.rc('text',usetex=usetex)
				fig = plt.figure()
				ax = fig.add_subplot(111)
				xs = np.linspace(0,len(peaks)+1,100)
				ax.plot(xs,xs*pars[0]+pars[1],ls='-',color='k',lw=2.0)
				ax.plot(xs,xs*pars[0]+pars[1],ls='-',color='C7',lw=1.0)
				ax.plot(np.arange(1,len(peaks)+1),self.tau[peaks],marker='x',color='C1',ls='none')
				ax.tick_params(axis='both',labelsize=font)
				ax.set_xlabel(r'$\rm Cycle \ number$',fontsize=font)
				ax.set_ylabel(r'$\rm Peak \ in \ ACF \ (days)$',fontsize=font)

		else:
			rr = np.diff(self.tau[peaks])
			MAD = np.nanmedian(np.abs(rr-np.nanmedian(rr)))*1.4826
			sd = MAD/np.sqrt(len(peaks)-1)
			per = np.median(rr)
			print('From median and MAD:')

		self.per = per
		self.sper = sd

		if print_pars:
			print('Prot = {:0.4f}+/-{:0.4f} d'.format(per,sd))



	def getProt(self,prep=True,timeWindow=8001,timePoly=3,
				gap=0.5,yfill=None,cadence=None,smooth=True,
				freqWindow=401,freqPoly=3):
		'''Get rotation period.

		Collection of calls to :py:class:`coPsi.Phot.Data.prepData()`, :py:class:`ACF()`, :py:class:`smoothACF()`, :py:class:`periodogram()`, and :py:class:`fitProt()`.


		:param prep: Prepare data before calculating, see :py:class:`coPsi.Phot.Data.prepData()`. Optional, default ``True``.
		:type prep: bool

		:param smooth: Smooth ACF, see :py:class:`smoothACF()`. Optional, default ``True``.
		:type smooth: bool


		'''



		if prep:
			self.prepData(window=timeWindow,poly=timePoly,gap=gap,yfill=yfill,cadence=cadence)
		self.ACF()
		if smooth:
			self.smoothACF(window=freqWindow,poly=freqPoly)
		self.periodogram()
		self.fitProt()


	def plotACF(self,ax=None,font=12,ymax=0.75,usetex=False,return_ax=0,peaks=np.array([])):

		'''Plot the periodogram

		:param ax: Axis in which to plot the KDE. Optional, default ``None``, figure and axis will be created.
		:type ax: :py:class:`matplotlib.axes._subplots.AxesSubplot`

		:param ymax: Maximum value for y-axis. Optional, default 0.7.
		:type ymax: float

		:param font: Fontsize for labels. Optional, default 12.
		:type font: float

		:param usetex: Whether to use LaTeX in plots. Optional, default ``False``.
		:type usetex: bool

		:param return_ax: Whether to return `ax`. Optional, default 0.
		:type return_ax: bool

		:param peaks: Peaks in the ACF. Optional, default empty.
		:type peaks: array


		'''
		if not ax:
			plt.rc('text',usetex=usetex)
			fig = plt.figure()
			ax = fig.add_subplot(111)
		ax.plot(self.tau,self.racf,color='k',alpha=0.3,label=r'$\rm Raw \ ACF$')
		ax.plot(self.tau,self.acf,color='k',label=r'$\rm Smoothed$')
		ax.axhline(0.0,ls='--',color='C7',zorder=-5,lw=0.5)
		ax.set_xlim(min(self.tau),max(self.tau))
		ax.set_ylim(ymax=ymax)
		ax.set_xlabel(r'$\tau_k \ \rm (days)$',fontsize=font)
		ax.set_ylabel(r'$\rm ACF$',fontsize=font)
		ax.tick_params(axis='both',labelsize=font)

		for ii, peak in enumerate(peaks):
			if not ii:
				ax.plot(self.tau[peak],self.acf[peak],marker='x',color='C1',markersize=10,label=r'$\rm Peak$',ls='none')
			else:
				ax.plot(self.tau[peak],self.acf[peak],marker='x',color='C1',markersize=10)

		ax.legend()

		if return_ax: return ax

	def plotPeriodogram(self,ax=None,xmax=None,font=12,usetex=False):
		'''Plot the periodogram

		:param ax: Axis in which to plot the KDE. Optional, default ``None``, figure and axis will be created.
		:type ax: :py:class:`matplotlib.axes._subplots.AxesSubplot`

		:param xmax: Maximum time for axis. Optional, default ``None``, will be set to half the duration of the timeseries.
		:type xmax: float

	
		:param font: Fontsize for labels. Optional, default 12.
		:type font: float

		:param usetex: Whether to use LaTeX in plots. Optional, default ``False``.
		:type usetex: bool

		'''

		if not ax:
			plt.rc('text',usetex=usetex)
			fig = plt.figure()
			ax = fig.add_subplot(111)
		time = 1/self.frequency
		ax.plot(time,self.power,lw=3.0,color='k')
		ax.plot(time,self.power,lw=1.5,color='C1',label=r'$\rm Periodogram$')

		if not xmax:
			xmax = (np.amax(self.x)-np.amin(self.x))*0.5
		ax.set_xlim(0.0,xmax)
		
		try:
			mu = self.fitPars[1]
			sigma = self.fitPars[2]
			low = mu-sigma
			high = mu+sigma

			cc = (time < high) & (time > low)
			yy = self.gauss(time,*self.fitPars)*np.amax(self.power)
			ax.fill_between(time[cc],yy[cc],color='C0',alpha=0.6,zorder=-1)#,transform=ax.transAxes)
			ax.plot(time,yy,lw=2.0,color='k')
			ax.plot(time,yy,lw=1.0,color='C0',label=r'$\rm Gaussian \ fit$')
			ax.axvline(mu,linestyle='-',color='C7')
		except AttributeError:
			pass
		
		ax.set_xlabel(r'$\rm Period \ (days)$',fontsize=font)
		ax.set_ylabel(r'$\rm Periodogram$',fontsize=font)
		ax.axhline(0.0,linestyle='-',color='k',zorder=-1)
		ax.tick_params(axis='both',labelsize=font)
		ax.legend()

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
	
