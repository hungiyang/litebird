
# ////////////////////////////////////////////// #
# * Calculate leaked Q/U maps from temperature 
# - Last Modified: Fri 26 Feb 2016 12:57:13 PM PST
# ////////////////////////////////////////////// #

import argparse
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 1024
#nside = 512
npix  = 12*nside**2
Tcmb  = 2.72e6
#type  = 'dc'
type  = 'dg'
#type  = 'dx'
#type  = 'dw'
fwhm  = .5*np.pi/180.
sigma = fwhm/np.sqrt(8.*np.log(2.))

# read parameters
parser = argparse.ArgumentParser( description='Simulate LiteBird pointing.' )
parser.add_argument( '--fp', required=False, default="bore", help='Allowed values are "bore", "nominal", "mirror", "radial"' )
parser.add_argument( '--genTmap', required=False, default=False, help='Allowed values are True or False' )
args = parser.parse_args()

if args.fp=='bore': detnum = 2

# //// read hit count and alpha //// #
#with open('../data/alpha_obs8760_nominal.bin', 'rb') as f: alps = np.fromfile(f, dtype=np.float64).reshape(-1,8)
with open('../data/angle_obs8760_'+args.fp+'.bin', 'rb') as f: angle  = np.fromfile(f, dtype=np.float64)
with open('../data/pix_obs8760_'+args.fp+'.bin', 'rb')   as f: detpix = np.fromfile(f, dtype=np.int64)
#with open('../data/dg_obs8760.bin', 'rb')           as f: dg     = np.fromfile(f, dtype=np.float64)
#with open('../data/dg_obs8760_fmin10mHz.bin', 'rb')           as f: dg     = np.fromfile(f, dtype=np.float64)
with open('../data/dg_obs8760_fmin10nHz.bin', 'rb')           as f: dg     = np.fromfile(f, dtype=np.float64)
A = np.zeros((npix,8))
detbinned = np.bincount(detpix)*detnum
denom = (detbinned+1e-6)#*np.float(len(args.fp.detectors(freq=freq)))
print(len(detpix),len(dg))
for n in range(4):
  Re = np.bincount(detpix, weights=np.cos((n+1)*angle))/denom
  Im = np.bincount(detpix, weights=np.sin((n+1)*angle))/denom
  A[0:Re.shape[0],2*n]   += Re
  A[0:Im.shape[0],2*n+1] += Im
  print(np.mean(A[:,0]),np.mean(A[:,4]))

#with open('../data/alpha_obs8760_bore.bin', 'rb') as f: alps = np.fromfile(f, dtype=np.float64).reshape(-1,8)
# a: cross link = \alpha
# input data: e^{(n+1)ia} = A[:,2*n] + i A[:,2*n+1]
e1a = A[:,0] + 1j*A[:,1]
e2a = A[:,2] + 1j*A[:,3]
e3a = A[:,4] + 1j*A[:,5]
e4a = A[:,6] + 1j*A[:,7]
D   = 1. - np.abs(e4a)**2
print(np.mean(A[:,2]),np.mean(A[:,1]),np.mean(D),np.mean(A[:,0]))


# //// generate T maps //// #
lmax = 700
L    = np.linspace(0,lmax,lmax+1)

if args.genTmap: 

  l, TT = np.loadtxt('../../../DATAS/cls/fid_P15.dat',usecols=(0,1)).T
  CTT = TT[:lmax-1]/(l[:lmax-1]**2+l[:lmax-1])*2*np.pi
  Tlm = hp.sphtfunc.synalm(CTT,lmax=lmax)
  #Tlm = hp.sphtfunc.smoothalm(Tlm,fwhm=fwhm) # beam convolution

  # map and derivatives
  T, Tt, Tp = hp.sphtfunc.alm2map_der1(Tlm,nside,lmax=lmax)
  d2T = hp.sphtfunc.alm2map(hp.sphtfunc.almxfl(Tlm,L*(L+1)),nside,lmax=lmax)

  with open('../data/T.bin', 'wb') as f: T.tofile(f)

else:
  with open('../data/T.bin', 'rb') as f: T = np.fromfile(f, dtype=np.float64)


# from map to TOD
T_tod = T[detpix]

# //// get leakage /// #

if type == 'dg': 
  Re = np.bincount(detpix, weights=dg*T_tod*np.cos(2*angle))/denom
  Im = np.bincount(detpix, weights=dg*T_tod*np.sin(2*angle))/denom
  #Re = np.bincount(detpix, weights=dg*np.cos(2*angle))/denom
  #Im = np.bincount(detpix, weights=dg*np.sin(2*angle))/denom
  DT = (Re+1j*Im)#*T

if type == 'dx': 
  # d = theta/2
  d = 1./60.*np.pi/180./2
  # < T_- e^{2ia} > = dx * dTdt * < -sin(a) e^{2ia} >      + dx * dTdp * < cos(a) e^{2ia} >
  #                 = dx * dTdt * < i(e^{3ia}-e^{ia})/2 >  + dx * dTdp * < (e^{3ia}+e^{ia})/2 >
  DT = d * .5 * ( Tt*1j*(e3a-e1a) + Tp*(e3a-e1a) )

if type == 'dw': 
  # d = \sigma x \delta\sigma = \sigma^2 x ( \delta\sigma / \sigma )
  d = sigma**2 * 0.01
  DT = d*e2a*d2T

if type == 'dc': 
  # d = \sigma^2 (c_A-c_B)
  d  = sigma**2 * (0.01-(-0.01))
  theta, phi = hp.pix2ang(nside,np.arange(npix))
  tant = np.sin(theta)/np.cos(theta)
  Ttlm = hp.sphtfunc.map2alm(Tt,lmax=lmax)
  Tplm = hp.sphtfunc.map2alm(Tp,lmax=lmax)
  dT, Ttt, Tpt = hp.sphtfunc.alm2map_der1(Ttlm,nside,lmax=lmax)
  dT, Ttp, Tpp = hp.sphtfunc.alm2map_der1(Tplm,nside,lmax=lmax)
  DT = ( -1j*(e4a-1.)*Ttt - (2*e2a+e4a+1)*Ttp + (2*e2a-e4a-1)*Tpt + 1j*(e4a-1)*Tpp - (1j*(e4a-1)*Tt+(2*e2a-e4a-1)*Tp)/tant )*d/4.

# Q/U leakage map
# Note: Q+iU = (2/D)(DT-DT^*e^{4i\alpha}) by Shimon et al 2008
TQU = np.zeros((3,npix))
TQU[1,:] = (2./D) * np.real( DT - np.conjugate(DT)*e4a )
TQU[2,:] = (2./D) * np.imag( DT - np.conjugate(DT)*e4a )
cls = np.array(hp.sphtfunc.anafast(TQU,lmax=lmax))


# //// plot //// #
l, BB = np.loadtxt('../../../DATAS/cls/lensedfid_P15.dat',usecols=(0,3)).T
plt.xlabel("multipole")
plt.ylabel(r"$L(L+1)C_L/2\pi [\mu$K$^2]$")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.xlim(3,500)
plt.ylim(1e-10,1e4)
cls[2,:] = cls[2,:] * L*(L+1)/2/np.pi
plt.plot(L,cls[2,:]*Tcmb**2,label="sim")
plt.plot(l[:lmax-1],BB[:lmax-1]*Tcmb**2,label="lensing B")
#plt.show()
plt.legend(loc=0)
plt.savefig(type+'.png')

#cls = hp.sphtfunc.anafast(Tmap,lmax=lmax)[2]


