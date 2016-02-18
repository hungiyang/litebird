import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

files = ['/global/u2/i/iameric/class/output/test_cl_lensed.dat']
data = []
for data_file in files:
    data = np.loadtxt(data_file)
#1:l     2:TT      3:EE       4:TE      5:BB   6:phiphi       7:TPhi      8:Ephi
names = ['l','TT','EE','BB','TE','phiphi','Tphi','Ephi']

# l starts with 2 in class's output ps
# let Cl=0 when l = 1
def remove_norm(Cl):
    return np.append(0,np.array([2*np.pi*1/l/(l+1)*Cl[l-2] for l in range(2,2+len(Cl))]))
l = data[:,0]
TT = remove_norm(np.insert(data[:,1],0,0))
EE = remove_norm(np.insert(data[:,2],0,0))
BB = remove_norm(np.insert(data[:,3],0,0))
TE = remove_norm(np.insert(data[:,4],0,0))


# map output is an tuple of three maps: T, Q, U
# alm output is an tuple of alm's of the T, Q, U maps
testmap, testalm = hp.synfast((TT,EE,BB,TE), nside = 1024, new = True, alm = True, fwhm = 0.01)
# dermap[0] is the original map, dermap[1], [2] are derivative with respect to theta and phi. (d_phi is devided with sin(theta))
map, d_theta, d_phi = hp.alm2map_der1(testalm[0], nside = 1024)

# create a map of sin(theta), to get rid of the sin(theta) in the d_phi map.
#With this, we can create maps of the second derivative
nside = 1024
sin_theta = np.sin(hp.pix2ang(nside = nside, ipix = range(hp.nside2npix(nside)))[0])
d_phi = d_phi*sin_theta

#create second derivative maps
map, dermap_tt, dermap_tp = hp.alm2map_der1(hp.map2alm(d_theta),nside = nside)
dermap_tp = dermap_tp*sin_theta
dermap_pp = hp.alm2map_der1(hp.map2alm(d_phi), nside = nside)[2]
dermap_pp = dermap_pp*sin_theta

#saves the derivative template to fits file 
#hp.write_map('maps/d_theta.fits',d_theta, fits_IDL = False)
#hp.write_map('maps/d_phi.fits',d_phi,fits_IDL = False)
#hp.write_map('maps/dermap_tt.fits', dermap_tt,fits_IDL = False)
#hp.write_map('maps/dermap_pp,fits', dermap_pp,fits_IDL = False )
#hp.write_map('maps/dermap_tp.fits', dermap_tp, fits_IDL = False)



