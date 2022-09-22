#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

class Data(object):
	
	def __init__(self,file=None,x=np.array([]),y=np.array([]),dy=None,cadence=None):
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
		arr = np.loadtxt(file)
		self.x = arr[:,0]
		self.y = arr[:,1]
		ncols = arr.shape[1]
		if ncols > 2:
			self.dy = arr[:,2]
		else:
			self.dy = None

	def filterData(self,window=5001,poly=3):		
		self.yhat = savgol_filter(self.y,window,poly)
		self.y = self.y/self.yhat


	def getCadence(self):
		self.cadence = np.median(np.diff(self.x))

	def fillGaps(self,gap=0.5,yfill=None,cadence=None):
		
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

	# def nyquist(cadence=None,Hz=False):
	# 	if not cadence:
	# 		getCadence()
	# 	if Hz:
	# 		self.nyq = 1/(2*86400*cadence) #Hz
	# 	else:		
 	# 		self.nyq = 1/(2*cadence)


	def prepData(self,file=None,window=8001,poly=3,gap=0.5,yfill=None,cadence=None):
		if file != None:
			self.readData(file)
		self.filterData(window=window,poly=poly)
		self.fillGaps(gap=gap,yfill=yfill,cadence=cadence)
		#self.filterData(**kwargs)

	def plot(self,ax=None,dots=1,return_ax=0,font=12,usetex=False):

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
		