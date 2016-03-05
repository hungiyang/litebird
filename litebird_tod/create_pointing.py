
import sys
import os
import time
import argparse

import numpy as np
import healpy as hp
import quaternionarray as qa

import litebird as lb


comm = None
rank = 0
nproc = 1

def lbtime():
    if comm is not None:
        return MPI.Wtime()
    else:
        return time.time()

def boresight_sim(nsim=1000, qprec=None, samplerate=23.0, spinperiod=10.0, spinangle=30.0, precperiod=93.0, precangle=65.0):

    spinrate = 1.0 / (60.0 * spinperiod)
    spinangle = spinangle * np.pi / 180.0
    precrate = 1.0 / (60.0 * precperiod)
    precangle = precangle * np.pi / 180.0

    xaxis = np.array([1,0,0], dtype=np.float64)
    yaxis = np.array([0,1,0], dtype=np.float64)
    zaxis = np.array([0,0,1], dtype=np.float64)

    satrot = None
    if qprec is None:
        satrot = np.tile(qa.rotation(np.array([0.0, 1.0, 0.0]), np.pi/2), nsim).reshape(-1,4)
    elif qprec.flatten().shape[0] == 4:
        satrot = np.tile(qprec, nsim).reshape(-1,4)
    elif qprec.shape == (nsim, 4):
        satrot = qprec
    else:
        raise RuntimeError("qprec has wrong dimensions")

    # Time-varying rotation about precession axis.  
    # Increment per sample is
    # (2pi radians) X (precrate) / (samplerate)
    # Construct quaternion from axis / angle form.
    precang = np.arange(nsim, dtype=np.float64)
    precang *= 2.0 * np.pi * precrate / samplerate

    # (zaxis, precang)
    cang = np.cos(0.5 * precang)
    sang = np.sin(0.5 * precang)
    precaxis = np.multiply(sang.reshape(-1,1), np.tile(zaxis, nsim).reshape(-1,3))
    precrot = np.concatenate((precaxis, cang.reshape(-1,1)), axis=1)

    # Rotation which performs the precession opening angle
    precopen = qa.rotation(np.array([1.0, 0.0, 0.0]), precangle)

    # Time-varying rotation about spin axis.  Increment 
    # per sample is
    # (2pi radians) X (spinrate) / (samplerate)
    # Construct quaternion from axis / angle form.
    spinang = np.arange(nsim, dtype=np.float64)
    spinang *= 2.0 * np.pi * spinrate / samplerate

    cang = np.cos(0.5 * spinang)
    sang = np.sin(0.5 * spinang)
    spinaxis = np.multiply(sang.reshape(-1,1), np.tile(zaxis, nsim).reshape(-1,3))
    spinrot = np.concatenate((spinaxis, cang.reshape(-1,1)), axis=1)

    # Rotation which performs the spin axis opening angle
    spinopen = qa.rotation(np.array([1.0, 0.0, 0.0]), spinangle)

    # compose final rotation
    boresight = qa.mult(satrot, qa.mult(precrot, qa.mult(precopen, qa.mult(spinrot, spinopen))))

    return boresight


def sim2(fp, freq, borequats, hwpang, hits, alps, inpp=None, hwprate=88.0, outdir = ''):

    nsim = borequats.shape[0]
    nhpix = hits.shape[0]
    nside = int(np.sqrt(nhpix / 12))

    if nhpix != 12*nside*nside:
        raise RuntimeError('invalid healpix nside value')
    if hwpang.shape[0] != borequats.shape[0]:
        raise RuntimeError('HWP angle vector must be same length as boresight quaternions')
    if inpp is not None:
        if inpp.shape[0] != nhpix:
            raise RuntimeError('N_pp^-1 number of pixels must match N_hits')
        if inpp.shape[1] != 6:
            raise RuntimeError('N_pp^-1 must have 6 elements per pixel')

    xaxis = np.array([1,0,0], dtype=np.float64)
    yaxis = np.array([0,1,0], dtype=np.float64)
    zaxis = np.array([0,0,1], dtype=np.float64)

    # generate hitcount map and alpha
    for i, det in enumerate(fp.detectors(freq=freq)):

        detrot = qa.mult(borequats, fp.quat(det))
        detdir = qa.rotate(detrot, np.tile(zaxis, nsim).reshape(-1,3))
        dettheta, detphi = hp.vec2ang(detdir)
        detpix = hp.vec2pix(nside, detdir[:,0], detdir[:,1], detdir[:,2])
        detbinned = np.bincount(detpix)
        hits[0:detbinned.shape[0]] += detbinned[:]

        outfile = os.path.join(outdir, 'theta.bin')
        with open(outfile, 'wb') as f:
            dettheta.tofile(f)
        outfile = os.path.join(outdir, 'phi.bin')
        with open(outfile, 'wb') as f:
            detphi.tofile(f)
        outfile = os.path.join(outdir, 'pix.bin')
        with open(outfile, 'wb') as f:
            detpix.tofile(f)

        if np.mod(i,2)!=1: 
            alpdir = qa.rotate(detrot, np.tile(xaxis, nsim).reshape(-1,3))
            x = alpdir[:,0]*detdir[:,1] - alpdir[:,1]*detdir[:,0]
            y = alpdir[:,0]*(-detdir[:,2]*detdir[:,0]) + alpdir[:,1]*(-detdir[:,2]*detdir[:,1]) + alpdir[:,2]*(detdir[:,0]*detdir[:,0]+detdir[:,1]*detdir[:,1])        
            angle = np.arctan2(y,x)

            outfile = os.path.join(outdir, 'angle.bin')
            with open(outfile, 'wb') as f:
                angle.tofile(f)

            #denom = (detbinned+1e-6)*np.float(len(fp.detectors(freq=freq)))
            #for n in range(4):
            #  Re = np.bincount(detpix, weights=np.cos((n+1)*angle))/denom
            #  Im = np.bincount(detpix, weights=np.sin((n+1)*angle))/denom
            #  alps[0:Re.shape[0],2*n]   += Re
            #  alps[0:Im.shape[0],2*n+1] += Im
            #print(np.mean(alps[:,0]),np.mean(alps[:,4]))


def main():

    if rank == 0:
        print("Running with {} processes".format(nproc))

    global_start = lbtime()

    allowed_freq = []
    for type in sorted(lb.pixel_types):
        for f in lb.pixel_to_freq[type]:
            allowed_freq.append(f)
    allowed_str = ", ".join(allowed_freq)
    #allowed_str = ''

    parser = argparse.ArgumentParser( description='Simulate LiteBird pointing.' )
    parser.add_argument( '--frequency', required=False, default='040', help='Frequency as a 3-digit string.  Valid values are {}'.format(allowed_str) )
    parser.add_argument( '--samplerate', required=False, default=1.0, help='Detector sample rate (Hz)' )
    parser.add_argument( '--spinperiod', required=False, default=10.0, help='The period (in minutes) of the rotation about the spin axis' )
    parser.add_argument( '--spinangle', required=False, default=30.0, help='The opening angle (in degrees) of the boresight from the spin axis' )
    parser.add_argument( '--precperiod', required=False, default=90.0, help='The period (in minutes) of the rotation about the precession axis' )
    parser.add_argument( '--precangle', required=False, default=65.0, help='The opening angle (in degrees) of the spin axis from the precession axis' )
    parser.add_argument( '--hwprpm', required=False, default=0.0, help='The rate (in RPM) of the HWP rotation' )
    parser.add_argument( '--hwpstep', required=False, default=None, help='For stepped HWP, the angle in degrees of each step' )
    parser.add_argument( '--hwpsteptime', required=False, default=0.0, help='For stepped HWP, the the time in seconds between steps' )
    parser.add_argument( '--obs', required=False, default=24.0, help='Number of hours in one science observation' )
    parser.add_argument( '--gap', required=False, default=0.0, help='Cycle time in hours between science obs' )
    parser.add_argument( '--numobs', required=False, default=1, help='Number of complete science + gap observations' )
    parser.add_argument( '--nside', required=False, default=1024, help='Healpix NSIDE' )
    parser.add_argument( '--invnpp', required=False, default=False, action='store_true', help='Also compute the block diagonal N_pp^-1 matrix' )
    parser.add_argument( '--fp', required=False, default="boreshift", help='Allowed values are "bore", "nominal", "mirror", "radial"' )
    parser.add_argument( '--wafersize', required=False, default=86.6, help='Wafer width in millimeters' )
    parser.add_argument( '--waferang', required=False, default=3.0, help='Angular size (in degrees) of wafer' )
    parser.add_argument( '--outdir', required=False, default='.', help='Output directory' )
    parser.add_argument( '--debug', required=False, default=False, action='store_true', help='Write focalplane image and other diagnostics' )
    parser.add_argument( '--shiftx', required=False, default=0., help='how much shifting of the detector on the focal plane in the x direction' )
    parser.add_argument( '--shifty', required=False, default=0., help='how much shifting of the detector on the focal plane in the y direction' )
    args = parser.parse_args()

    nside = int(args.nside)
    npix = 12 * nside * nside

    samplerate = float(args.samplerate)
    spinperiod = float(args.spinperiod)
    spinangle = float(args.spinangle)
    precperiod = float(args.precperiod)
    precangle = float(args.precangle)

    wafer_mm = float(args.wafersize)
    wafer_deg = float(args.waferang)

    hwprate = float(args.hwprpm)
    hwpstep = None
    if args.hwpstep is not None:
        hwpstep = float(args.hwpstep)
    hwpsteptime = float(args.hwpsteptime)

    start = lbtime()
    fp = None
    freq = args.frequency
    shiftx = float(args.shiftx)
    shifty = float(args.shifty)

    #create a single detector, and give it a shift on the focal plane
    if args.fp == "boreshift":
        pol = lb.pol_angles_qu(1)
        pixels = ['L1A']
        wafers = []
        wafers.append( (lb.Wafer(pixels=pixels, pol=pol), "{}B".format(''), np.array([shiftx, shifty, 0.0])) )
        fp =  lb.FocalPlane(wafers=wafers)
    
    elif args.fp == "bore":
        fp = lb.create_focalplane_bore()
        freq = allowed_freq[0]
    elif int(args.frequency) > 250:
        fp = lb.create_HFT_nominal(wafer_mm, wafer_deg, margin=8.0)
    else:
        if args.fp == "nominal":
            fp = lb.create_LFT_nominal(wafer_mm, wafer_deg, margin=8.0)
        elif args.fp == "mirror":
            fp = lb.create_LFT_mirror(wafer_mm, wafer_deg, margin=8.0)
        elif args.fp == "radial":
            fp = lb.create_LFT_radial(wafer_mm, wafer_deg, margin=8.0)
        else:
            raise RuntimeError("Unknown focalplane type \"{}\"".format(args.fp))
    stop = lbtime()
    if rank == 0:
        print("Create focalplane:  {:.2f} seconds".format(stop-start))

    if args.outdir != '.':
        if not os.path.isdir(args.outdir):
            os.mkdir(args.outdir)

    if args.debug:
        if rank == 0:
            import matplotlib.pyplot as plt
            import litebird.vis as lbv
            fig = plt.figure( figsize=(36,18), dpi=100 )
            ax = fig.add_subplot(1, 1, 1)
            lbv.view_focalplane(fp, ax, freq=args.frequency)
            outfile = os.path.join(args.outdir, 'focalplane.png')
            plt.savefig(outfile)

    hits = np.zeros(npix)
    alps = np.zeros((npix,8))
    inpp = None
    if args.invnpp:
        inpp = np.zeros((npix, 6), dtype=np.float64)

    obs = 3600.0 * float(args.obs)
    gap = 3600.0 * float(args.gap)
    obssamples = int(obs * samplerate)
    gapsamples = int(gap * samplerate)
    nobs = int(args.numobs)
    simsamples = obssamples + gapsamples

    # compute which observations this process is responsible for

    startobs = 0
    stopobs = nobs

    if comm is not None:
        myobs = nobs // nproc
        leftover = nobs % nproc
        if ( rank < leftover ):
            myobs += 1
            startobs = rank * myobs
        else:
            startobs = ((myobs + 1) * leftover) + (myobs * (rank - leftover))
        stopobs = startobs + myobs
        if myobs == 0:
            print("WARNING: process {} assigned no data and will be idle".format(rank))

    # Assume that we constantly slew the precession axis at one
    # degree per day, regardless of whether we are in a science
    # observation or a cooler cycle.

    # this is the increment per sample
    angincr = (np.pi / 180.0) / (24.0 * 3600.0 * samplerate)

    # this is the increment per complete observation
    obincr = angincr * simsamples

    if comm is not None:
        comm.barrier()
    start = lbtime()

    elapsed_bore = 0
    elapsed_sim = 0

    for ob in range(nobs):

        if (ob < startobs) or (ob >= stopobs):
            continue

        start_bore = lbtime()
        # Compute the time-varying quaternions representing the rotation
        # from the coordinate frame to the precession axis frame.  The
        # angle of rotation is fixed (PI/2), but the axis starts at the Y
        # coordinate axis and sweeps.

        # angle about coordinate z-axis
        satang = np.arange(simsamples, dtype=np.float64)
        satang *= angincr
        satang += ob * obincr + (np.pi / 2)

        # this is the time-varying rotation axis
        # sataxis = [cos(ft+pi/2), sin(ft+pi/2), 0]
        cang = np.cos(satang)
        sang = np.sin(satang)
        sataxis = np.concatenate((cang.reshape(-1,1), sang.reshape(-1,1), np.zeros((simsamples,1))), axis=1)

        # now construct the axis-angle quaternion
        # the rotation about the axis is always pi/2 (in order to change z->x at t=0)
        # satquat = (sataxis, pi/2)
        csatrot = np.cos(0.25 * np.pi)
        ssatrot = np.sin(0.25 * np.pi)
        sataxis = np.multiply(np.repeat(ssatrot, simsamples).reshape(-1,1), sataxis)
        satquat = np.concatenate((sataxis, np.repeat(csatrot, simsamples).reshape(-1,1)), axis=1)

        #borequats = lb.boresight_sim(nsim=simsamples, qprec=satquat, samplerate=samplerate, spinperiod=spinperiod, spinangle=spinangle, precperiod=precperiod, precangle=precangle)
        borequats = boresight_sim(nsim=simsamples, qprec=satquat, samplerate=samplerate, spinperiod=spinperiod, spinangle=spinangle, precperiod=precperiod, precangle=precangle)

        stop_bore = lbtime()
        elapsed_bore += stop_bore - start_bore

        start_sim = lbtime()

        hwpang = lb.hwp_angles(0, simsamples, samplerate, hwprate, hwpstep, hwpsteptime)

        #lb.simulate(fp, args.frequency, borequats, hwpang, hits, inpp=inpp)
        sim2(fp, args.frequency, borequats, hwpang, hits, alps, inpp=inpp, outdir = args.outdir)
        
        stop_sim = lbtime()
        elapsed_sim += stop_sim - start_sim
        

    min_bore = np.zeros(1)
    max_bore = np.zeros(1)
    min_sim = np.zeros(1)
    max_sim = np.zeros(1)

    if comm is not None:
        comm.barrier()
        comm.Reduce(np.array(float(elapsed_bore)), min_bore, op=MPI.MIN, root=0)
        comm.Reduce(np.array(float(elapsed_bore)), max_bore, op=MPI.MAX, root=0)
        comm.Reduce(np.array(float(elapsed_sim)), min_sim, op=MPI.MIN, root=0)
        comm.Reduce(np.array(float(elapsed_sim)), max_sim, op=MPI.MAX, root=0)
    else:
        min_bore[0] = elapsed_bore
        max_bore[0] = elapsed_bore
        min_sim[0] = elapsed_sim
        max_sim[0] = elapsed_sim

    stop = lbtime()

    if rank == 0:
        print("Parallel Simulation:  {:.2f} seconds".format(stop-start))
        print("  Boresight calculation:  min = {:.2f} s, max = {:.2f} s".format(min_bore[0], max_bore[0]))
        print("  Detector pointing and accumulate:  min = {:.2f} s, max = {:.2f} s".format(min_sim[0], max_sim[0]))

    start = lbtime()

    fullhits = None
    fullinpp = None

    if comm is not None:
        if rank == 0:
            fullhits = np.zeros(npix, dtype=np.float64)
            if inpp is not None:
                fullinpp = np.zeros((npix, 6), dtype=np.float64)

        comm.Reduce(hits, fullhits, op=MPI.SUM, root=0)
        if inpp is not None:
            comm.Reduce(inpp, fullinpp, op=MPI.SUM, root=0)
    else:
        fullhits = hits
        fullinpp = inpp

    stop = lbtime()
    if rank == 0:
        print("Reduction:  {:.2f} seconds".format(stop-start))


    start = lbtime()

    if rank == 0:

        #outfile = os.path.join(args.outdir, 'hits.fits')
        #hp.fitsfunc.write_map(outfile,hits)
        outfile = os.path.join(args.outdir, 'hits.bin')
        with open(outfile, 'wb') as f:
            fullhits.tofile(f)

        if fullinpp is not None:
            outfile = os.path.join(args.outdir, 'invnpp.bin')
            with open(outfile, 'wb') as f:
                fullinpp.tofile(f)

    stop = lbtime()
    if rank == 0:
        print("Write hits and N_pp^-1:  {:.2f} seconds".format(stop-start))

    if comm is not None:
        comm.barrier()
    global_stop = lbtime()
    if rank == 0:
        print("Total Time:  {:.2f} seconds".format(global_stop-global_start))

if __name__ == "__main__":
    main()

