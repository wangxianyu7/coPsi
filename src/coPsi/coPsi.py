#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:00:12 2022

@author: emil
"""
import numpy as np
import astropy.units as u
import statsmodels.api as sm
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd
import emcee
dayfac = 86400
rsun = 1*u.R_sun
rsunfac = rsun.to_value('km')



from .priors import *


class iStar(object):

	parameters = [#'incs', 
				'inco', 
				'lam', 
				'Prot', 
				'Rs', 
				'vsini',
				'Teff'
               ]

	stepParameters = [
				'Rs',
				'Prot',
				'cosi'
                   ]

	labels = {
			'Rs'    : r'$R_\star \ \ (R_\odot)$',
			'Prot'  : r'$P_{\rm rot}\ \rm (days)$',
			'cosi'  : r'$\cos i_\star$',
			'incs'  : r'$i_\star \ \rm (deg)$',
			'v'     : r'$v \ \rm (km/s)$',
			'vsini' : r'$v \sin i_\star \ \rm (km/s)$',
			}

	def __init__(self,
              inco = (85.,0.5,0.0,90.,'gauss'),
              lam = (0,0.5,-180.,180.,'gauss'),
              Prot = (3.5,0.5,0.0,10.,'gauss'),
              Rs = (1.0,0.1,0.5,2.0,'gauss'),
              vsini = (4.5,0.5,0.0,10.,'gauss'),
              cosi = (0,0.1,-1.0,1.0,'uni'),
              Teff = (6250,100,3000,9000,'gauss'),
              ):
		'''
  

		Parameters
		----------
		inco : float, array
			Orbital inclination (degrees).
		lam : float, array
			Projected obliquity (degrees).  
		Prot : float, array
			Rotation period (seconds).
		Rs : float, array
			Stellar radius (km).
		vsini : float, array
			Stellar projected rotation speed (km/s).

  
		'''
		self.inco = inco
		self.lam = lam
		self.Prot = Prot
		self.Rs = Rs
		self.vsini = vsini
		self.cosi = cosi
		self.Teff = Teff

	#def coPsi(self,inco,incs,lam,return_psi=True):
	def coPsi(self):
		'''Calculates cos(psi)
		
		cos(psi) = sin(incs)*sin(inco)*cos(lam) + cos(incs)*cos(inco)
		
		Parameters
		----------
		inco : float, array
			Orbital inclination (degrees).
		incs : float, array
			Stellar inclination (degrees).
		lam : float, array
			Projected obliquity (degrees).

		Returns
		-------
		float, array
			Return psi, cos(psi) or just cos(psi), depending on return_psi.

		'''
		inco = np.deg2rad(self.dist['inco'])
		incs = np.deg2rad(self.dist['incs'])
		lam = np.deg2rad(self.dist['lam'])
		
		self.dist['cosp'] = np.sin(incs)*np.sin(inco)*np.cos(lam) + np.cos(incs)*np.cos(inco)
		self.dist['psi'] = np.rad2deg(np.arccos(self.dist['cosp']))
		#self.psi = np.rad2deg(np.arccos(self.cosp))
		#self.cosp = np.sin(incs)*np.sin(inco)*np.cos(lam) + np.cos(incs)*np.cos(inco)
  
  

	#def stellarInclination(Prot,Rs,vsini):
	def stellarInclinationDirectly(self,convert=True):
		'''Calculate stellar inclination
		
		incs = Prot*vsini/(2*pi*Rs)

		Parameters
		----------
		Prot : float, array
			Rotation period (seconds).
		Rs : float, array
			Stellar radius (km).
		vsini : float, array
			Stellar projected rotation speed (km/s).

		Returns
		-------
		float, array
			Return Stellar inclination.

		'''
		try:
			self.dist
		except AttributeError:
			print('Distributions not initilialized.\nCalling iStar.createDistributions.')
			self.createDistributions()
		
		if convert:
			self.dist['Rs'] = self.dist['Rs']*u.R_sun
			self.dist['Rs'] = self.dist['Rs'].to_value('km')
			self.dist['Prot'] = self.dist['Prot']*u.d
			self.dist['Prot'] = self.dist['Prot'].to_value('s')


		si = self.dist['Prot']*self.dist['vsini']/(2*np.pi*self.dist['Rs'])
		incs = np.arcsin(si)
		self.dist['incs'] =  np.rad2deg(incs)

	def stellarInclinationLouden(self,relation='two'):
    
		cs = {
			'single' : {
			'c0' : [9.57,0.29],
			'c1' : [8.01,0.54],
			'c2' : [3.30,0.62],
			'sini' : [0.856,0.036]
				},
			'two' : {
			'c0' : [9.44,0.28],
			'c1' : [8.87,0.61],
			'c2' : [4.05,0.62],
			'sini' : [0.794,0.052],
			'sini_down' : [0.928,0.042]
				}
			}

		N = len(self.dist['Teff'])
		c0 = np.random.normal(cs[relation]['c0'][0],cs[relation]['c0'][1],N)
		c1 = np.random.normal(cs[relation]['c1'][0],cs[relation]['c1'][1],N)
		c2 = np.random.normal(cs[relation]['c2'][0],cs[relation]['c2'][1],N)
		tau = (self.dist['Teff']-6250)/300
		v_avg = c0 + c1*tau + c2*tau**2
		vs = self.dist['vsini']#np.random.normal(vsini[0],vsini[1],N)
		si = vs/v_avg
		incs = np.rad2deg(np.arcsin(si))
		incs = incs[np.isfinite(incs)]
		self.dist['incs'] = incs
		#inc_star[ii] = incs
  
		# for ii in range(N):
		# 	t = np.random.normal(teff[0],teff[1])
		# 	tau = (t-6250)/300
		# 	if tau < 0.0:
		# 		sini = cs['two']['sini_down']
		# 	else:
		# 		sini = cs['two']['sini']
			
		# 	#print(tau)
		# 	c0 = np.random.normal(cs['two']['c0'][0],cs['two']['c0'][1])
		# 	c1 = np.random.normal(cs['two']['c1'][0],cs['two']['c1'][1])
		# 	c2 = np.random.normal(cs['two']['c2'][0],cs['two']['c2'][1])
		# 	v_avg = c0 + c1*tau + c2*tau**2
		# 	vs = vsini[ii]#np.random.normal(vsini[0],vsini[1])
		# 	c = v_avg/vs
		# 	si = 1/c
		# 	incs = np.arcsin(si)*180/np.pi
		# 	inc_star[ii] = incs


	def stellarInclination(self,ndraws=10000,nwalkers=100,nproc=1,
                        moves=None,path='./',
                        plot_corner=True,
                        plot_convergence=True):
		'''Calculate stellar inclination
		
		Following Masuda & Winn (2020)
		

		Parameters
		----------
		Prot : float, array
			Rotation period (seconds).
		Rs : float, array
			Stellar radius (km).
		vsini : float, array
			Stellar projected rotation speed (km/s).

		Returns
		-------
		float, array
			Return Stellar inclination.

		'''

		def start_coords(nwalkers,ndim):
			fps = self.stepParameters
			pos = []
			pars = vars(self)
			for ii in range(nwalkers):
				start = np.ndarray(ndim)
				for idx, par in enumerate(fps):
					pri = pars[par][:4]
					dist = pars[par][-1]
					assert dist in ['gauss','tgauss','uni','jeff'], print('{} is not a valid option for the starting distribution.'.format(dist))
					if dist == 'tgauss':
						start[idx] = tgauss_prior_dis(pri[0],pri[1],pri[2],pri[3])
					elif dist == 'gauss':
						start[idx] = gauss_prior_dis(pri[0],pri[1])
					elif dist == 'uni':
						start[idx] = flat_prior_dis(np.random.uniform(),pri[2],pri[3])
					elif dist == 'jeff':
						start[idx] = jeff_prior_dis(np.random.uniform(),pri[2],pri[3])
				pos.append(start)
			return pos




		ndim = len(self.stepParameters)
		## Starting coordinates for walkers
		coords = start_coords(nwalkers,ndim)
		pp = vars(self)
		pars = { 'FPs' : ['Rs','Prot','cosi'],
				'Rs' : pp['Rs'],
				'Prot' : pp['Prot'],
				'cosi' : pp['cosi'],
				'vsini' : pp['vsini'],
            }
		
		with Pool(nproc) as pool:
			sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,
					pool=pool,moves=moves,kwargs=pars)


			print('Number of draws is {}.\n'.format(ndraws))
						
			## We'll track how the average autocorrelation time estimate changes
			index = 0
			autocorr = np.empty(ndraws)
			## This will be useful to testing convergence
			old_tau = np.inf

			for sample in sampler.sample(coords,iterations=ndraws,progress=True):
				## Only check convergence every 100 steps
				if sampler.iteration % 100: continue

				## Compute the autocorrelation time so far
				## Using tol=0 means that we'll always get an estimate even
				## if it isn't trustworthy
				tau = sampler.get_autocorr_time(tol=0)
				autocorr[index] = np.mean(tau)
				index += 1
				## Check convergence
				converged = np.all(tau * 50 < sampler.iteration)
				converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
				if converged: 
					print('\nMCMC converged after {} iterations.'.format(sampler.iteration))
					break
				
				old_tau = tau

		
		## Plot autocorrelation time
		## Converged if: 
		## - chain is longer than 100 times the estimated autocorrelation time 
		## - & this estimate changed by less than 1%.
		if plot_convergence:
			figc = plt.figure()
			axc = figc.add_subplot(111)
			nn, yy = 100*np.arange(1,index+1), autocorr[:index]
			axc.plot(nn,nn/50.,'k--')
			axc.plot(nn,yy,'k-',lw=3.0)
			axc.plot(nn,yy,'-',color='C0',lw=2.0)
			axc.set_xlabel(r'$\rm Step \ number$')
			axc.set_ylabel(r'$\rm \mu(\hat{\tau})$')
			plt.savefig(path+'/autocorr.pdf')


		tau = sampler.get_autocorr_time(tol=0)
		n_auto = 2
		thin = int(0.5 * np.min(tau))
		burnin = int(n_auto * np.max(tau))
		samples = sampler.get_chain(discard=burnin)	
		flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)	
		flat_samples[:,2] = abs(flat_samples[:,2])
		incs = np.rad2deg(np.arccos(flat_samples[:,2]))
		flat_samples = np.concatenate(
							(flat_samples, incs[:,None]), axis=1)

		nom = 2*np.pi*flat_samples[:,0]*rsunfac
		den = flat_samples[:,1]*dayfac
		v = nom/den
		flat_samples = np.concatenate(
							(flat_samples, v[:,None]), axis=1)

		flat_samples = np.concatenate(
							(flat_samples, v[:,None]*np.sin(np.deg2rad(incs[:,None]))), axis=1)

		log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

		results = {}
		results['Parameter'] = ['Median','Lower','Upper']
		medians = []
		pps = pars['FPs'] + ['incs','v','vsini']

		for ii, fp in enumerate(pps):
			val,lower,upper = self.getConfidence(flat_samples[:,ii])
			results[fp] = [val,lower,upper]
			medians.append(val)
		labs = [self.labels[par] for par in pps]
		#labels = [ r'$R_\star$',r'$P_{\rm rot}$',r'$\cos i_\star$',r'$i_\star$',r'$v$',r'$v \sin i_\star$',r'$\ln \mathcal{L}$']
		labs.append(r'$\ln \mathcal{L}$')
		res_df = pd.DataFrame(results)
		res_df.to_csv(path+'results.csv')
		if plot_corner:
			import corner
			all_samples = np.concatenate(
				(flat_samples, log_prob_samples[:, None]), axis=1)
			medians.append(np.amax(log_prob_samples[:, None]))
			plt.figure()
			corner.corner(all_samples, labels=labs, truths=None)
			plt.savefig(path+'corner.pdf')


	

	def createDistributions(self,N=2000):#,convert=True):
		pars = vars(self)
		self.dist = {}
		generateDist = []

		for par in pars:
			if par in self.parameters:
				if type(pars[par]) == tuple:
					generateDist.append(par)
				elif type(pars[par]) == np.ndarray:
					self.dist[par] = pars[par]
					N = len(pars[par])
				else:
					print('{} should be either tuple (see __init__) or numpy.ndarray'.format(par))
		print(generateDist)			
		for par in generateDist:
			if pars[par][-1] == 'gauss':
				self.dist[par] = np.random.normal(pars[par][0],pars[par][1],N)
			elif pars[par][-1] == 'uniform':
				self.dist[par] = np.random.uniform(pars[par][2],pars[par][3],N)

		#if convert: self.convertUnits()

	def getKDE(self,z):

		kde = sm.nonparametric.KDEUnivariate(z)
		kde.fit()  # Estimate the densities
		x, y = kde.support, kde.density
		return x, y


		
	def hpd(self,data, lev=0.68) :
		''' The Highest Posterior Density.
		
		The Highest Posterior Density (credible) interval of data at level lev.

		Parameters
		----------
		data : array
			Sequence of real values..
		lev : float, optional
			Level for hpd (0 < lev < 1). The default is 0.68.

		Raises
		------
		RuntimeError
			If insufficient data is supplied.

		Returns
		-------
		TYPE
			DESCRIPTION.
		TYPE
			DESCRIPTION.
		i : TYPE
			DESCRIPTION.
		TYPE
			DESCRIPTION.

		'''
		
		d = list(data)
		d.sort()

		nData = len(data)
		nIn = int(round(lev * nData))
		if nIn < 2 :
			raise RuntimeError("not enough data")
		
		i = 0
		r = d[i+nIn-1] - d[i]
		for k in range(len(d) - (nIn - 1)) :
			rk = d[k+nIn-1] - d[k]
			if rk < r :
				r = rk
				i = k

		assert 0 <= i <= i+nIn-1 < len(d)
		
		return (d[i], d[i+nIn-1], i, i+nIn-1)

	def getConfidence(self,z,lev=0.68):
		'''Calculate confidence level.
		

		Parameters
		----------
		z : array
			Distribution for which to calculate confidence intervals.
		lev : float, optional
			Level of confidence. The default is 0.68.

		Returns
		-------
		val : float
			Median value.
		up : float
			Upper uncertainty.
		low : float
			Lower uncertainty.

		'''
		val = np.median(z)
		conf = self.hpd(z,lev=lev)
		up = conf[1]-val
		low = val-conf[0]

		return val, up, low


	def diagnostics(self,z,par='Parameter',ax=None,lev=0.68):
		print(z)
		if type(z) == str:
			par = z
			z = self.dist[z]

		
		#print(self.dist[z])
		print(z)
		print(lev)
		val, up, low = self.getConfidence(z,lev=lev)
		print('{}={}+{}-{}'.format(par,val,up,low))
		xkde, ykde = self.getKDE(z)
		if ax == None:
			fig = plt.figure()
			ax = fig.add_subplot(111)
		ax.plot(xkde,ykde)

def lnprob(positions,**pars):
	#parameters = master_dict['pars']
	log_prob = 0.0
	#fps = self.stepParameters
	fps = pars['FPs']
	#pars = vars(self)
	for idx, par in enumerate(fps):
		val = positions[idx]
		pri = pars[par][:4]
		ptype = pars[par][-1]
		pval, sigma, lower, upper = pri[0], pri[1], pri[2], pri[3]
		if ptype == 'uni':
			prob = flat_prior(val,lower,upper)
		elif ptype == 'jeff':
			prob = jeff_prior(val,lower,upper)
		elif ptype == 'gauss':
			prob = gauss_prior(val,pval,sigma)
		elif ptype == 'tgauss':
			prob = tgauss_prior(val,pval,sigma,lower,upper)
		if prob != 0.:
			log_prob += np.log(prob)
		else:
			return -np.inf
	nom = 2*np.pi*positions[0]*rsunfac
	den = positions[1]*dayfac
	v = nom/den
	vsini = v*np.sqrt(1-positions[2]**2)

	prob = gauss_prior(vsini,pars['vsini'][0],pars['vsini'][1])
	if prob != 0.:
		return np.log(prob) + log_prob
	else:
		return -np.inf

if __name__ == '__main__':
	N = 2000
	#incs = createDistribution(N,85.3,0.02)
	inco = createDistribution(N,85.3,0.02)#*deg2rad
	
	r = 2.69*u.R_sun
	r = r.to_value('km')
	sr = 0.04*u.R_sun
	sr = sr.to_value('km')
	
	rs = createDistribution(N,r,sr)
	vs = createDistribution(N,1.99,0.07)
	p = 3.36316383*u.d
	p = p.to_value('s')
	sp = 0.32774516*u.d
	sp = sp.to_value('s')
	 
	Prot = createDistribution(N,p,sp)
	
	si = stellarInclination(Prot,rs,vs)#*deg2rad
	lam = createDistribution(N,-77.86,2.3)#*deg2rad
	
	
	psi, _ = coPsi(inco,si,lam)
	
	upsi, _ = coPsi(180-inco,si,lam)
	
	
	v, u, l = getConfidence(psi)
	
	
	xk, yk = getKDE(psi)
	xk2, yk2 = getKDE(upsi)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(xk,yk)
	ax.plot(xk2,yk2)
	uv, uu, ul = getConfidence(upsi)
	
	#r = Rs#.to_value('km')*u.km
	#p = Prot.to_value('s')*u.s
	#vs = vsini[ii]*u.km/u.s

