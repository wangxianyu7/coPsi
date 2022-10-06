#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

class Data(object):
	'''Photometric data.
	
	Read and prepare photometric data for which to calculate the stellar rotation period.

	Data constructor initialized with either filename for data in a .txt file, or by setting x (time) and y (flux) directly (and dy, error).


	:param file: Filename. Optional, default ``None``.
	:type file: str

	:param x: Time.
	:type x: array

	:param y: Flux.
	:type y: array

	:param dy: Flux error.
	:type dy: array

	:param cadence: Cadence for observations. Optional, default ``None``. If ``None`` calculated from :py:class:`getCadence()`
	:param cadence: float
	
	'''

	def __init__(self,file=None,x=np.array([]),y=np.array([]),dy=None,cadence=None):
		'''Constructor

		'''
		if file != None:
			self.readData(file)
			self.getCadence()
		else:
			self.x = x
			self.y = y
			self.dy = dy
			if cadence is None:
				self.getCadence()
			else:
				self.cadence = cadence

		self.ii = 0

	def readData(self,file):
		'''Data from .txt file

		t1 fl1 efl1
		t2 fl2 efl2
		...........
		tn fln efln

		:param file: Filename.
		:type file: str

		'''
		arr = np.loadtxt(file)
		self.x = arr[:,0]
		self.y = arr[:,1]
		ncols = arr.shape[1]
		if ncols > 2:
			self.dy = arr[:,2]
		else:
			self.dy = None

	def filterData(self,window=5001,poly=3):
		'''Savitsky-Golay filter

		Using :py:class:`scipy.signal.savgol_filter()`.

		:param window: Length of window for filter. Optional, default 5001.
		:type window: int

		:param poly: Degree for polynomial. Optional, default 3.
		:type poly: int

		'''
		self.yhat = savgol_filter(self.y,window,poly)
		self.y = self.y/self.yhat


	def getCadence(self):
		'''Calculate cadence

		Calculated as the median of the difference between timestamps.

		'''
		self.cadence = np.median(np.diff(self.x))

	def fillGaps(self,gap=0.5,yfill=None,cadence=None):
		'''Fill gaps

		Fill the gaps in the data by simply putting in the median of the observations by default.

		:param gap: Value to consider a gap in days. Optional, default 0.5.
		:type gap: float

		:param yfill: Values to put in gaps. Optional, default ``None``. If ``None`` median is used.
		:type yfill: float

		:param cadence: Cadence for observations. Optional, default ``None``. If ``None`` calculated from :py:class:`getCadence()`
		:param cadence: float		

		'''
		gaps = np.where(np.diff(self.x) > gap)[0]
		
		if yfill == None:
			yfill = np.median(self.y)
		
		if not cadence:
			self.getCadence()
			cadence = self.cadence
		
		nx = self.x.copy()#np.array([])
		ny = self.y.copy()#np.array([])
		for g in gaps:
			new_x = np.arange(self.x[g],self.x[g+1],cadence)
			new_y = np.zeros_like(new_x) + yfill
			
			nx = np.append(nx,new_x[1:])
			ny = np.append(ny,new_y[1:])
		
		ss = np.argsort(nx)
		nx = nx[ss]
		ny = ny[ss]
		nx -= min(self.x)
		self.x = nx
		self.y = ny


	def prepData(self,file=None,window=8001,poly=3,gap=0.5,yfill=None,cadence=None):
		'''Prepare data.

		Collection of calls to :py:class:`readData()`, :py:class:`filterData()`, and :py:class:`fillGaps()`

		'''
		if file != None:
			self.readData(file)
		self.filterData(window=window,poly=poly)
		self.fillGaps(gap=gap,yfill=yfill,cadence=cadence)
		#self.filterData(**kwargs)

	def plotData(self,ax=None,dots=1,return_ax=0,font=12,usetex=False):
		'''Plot the data

		:param ax: Axis in which to plot the KDE. Optional, default ``None``, figure and axis will be created.
		:type ax: :py:class:`matplotlib.axes._subplots.AxesSubplot`

		:param dots: Whether to use dots instead of lines in plot. Optional, default 1.
		:type dots: bool

		:param return_ax: Whether to return `ax`. Optional, default 0.
		:type return_ax: bool
	
		:param font: Fontsize for labels. Optional, default 12.
		:type font: float

		:param usetex: Whether to use LaTeX in plots. Optional, default ``False``.
		:type usetex: bool

		'''
		if not ax:
			plt.rc('text',usetex=usetex)
			fig = plt.figure()
			ax = fig.add_subplot(111)
		if dots:
			ax.plot(self.x-min(self.x),self.y,marker='o',markersize=2,color='k',ls='none')
			ax.plot(self.x-min(self.x),self.y,marker='o',markersize=1,color='C{}'.format(self.ii),ls='none')
		else:
			ax.plot(self.x,self.y,color='k',lw=0.5)

		ax.set_xlabel(r'$\rm Time \ (days)$',fontsize=font)
		ax.set_ylabel(r'$\rm Relative \ brightness$',fontsize=font)
		ax.tick_params(axis='both',labelsize=font)

		self.ii += 1
		if return_ax: return ax

	# def to_rotator(self):
	# 	pass
		