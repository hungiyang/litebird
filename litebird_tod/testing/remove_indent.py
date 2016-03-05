
import mpi4py.MPI as MPI

import os
import re
import argparse

import toast
import toast.tod as tt
import toast.map as tm

import litebird as lb
import litebird.toast as lbt



# This is the 2-level toast communicator.  By default,
# there is just one group which spans MPI_COMM_WORLD.
comm = toast.Comm()

if comm.comm_world.rank == 0:
    print("Running with {} processes".format(comm.comm_world.size))

#global_start = MPI.Wtime()

allowed_freq = []
for type in sorted(lb.pixel_types):
    for f in lb.pixel_to_freq[type]:
        allowed_freq.append(f)
allowed_str = ", ".join(allowed_freq)

parser = argparse.ArgumentParser( description='Simulate LiteBIRD data and make a map.' )
parser.add_argument( '--frequency', required=False, default='040', help='Frequency as a 3-digit string.  Valid values are {}'.format(allowed_str) )
parser.add_argument( '--samplerate', required=False, default=23.0, help='Detector sample rate (Hz)' )
parser.add_argument( '--spinperiod', required=False, default=10.0, help='The period (in minutes) of the rotation about the spin axis' )
parser.add_argument( '--spinangle', required=False, default=30.0, help='The opening angle (in degrees) of the boresight from the spin axis' )
parser.add_argument( '--precperiod', required=False, default=93.0, help='The period (in minutes) of the rotation about the precession axis' )
parser.add_argument( '--precangle', required=False, default=65.0, help='The opening angle (in degrees) of the spin axis from the precession axis' )
parser.add_argument( '--hwprpm', required=False, default=88.0, help='The rate (in RPM) of the HWP rotation' )
parser.add_argument( '--hwpstep', required=False, default=None, help='For stepped HWP, the angle in degrees of each step' )
parser.add_argument( '--hwpsteptime', required=False, default=0.0, help='For stepped HWP, the the time in seconds between steps' )

parser.add_argument( '--obs', required=False, default=0.1, help='Number of hours in one science observation' )
parser.add_argument( '--gap', required=False, default=0.0, help='Cycle time in hours between science obs' )
parser.add_argument( '--numobs', required=False, default=1, help='Number of complete observations' )

parser.add_argument( '--fp', required=False, default="mirror", help='Allowed values are "nominal", "mirror", "radial"' )
parser.add_argument( '--wafersize', required=False, default=86.6, help='Wafer width in millimeters' )
parser.add_argument( '--waferang', required=False, default=3.0, help='Angular size (in degrees) of wafer' )

parser.add_argument( '--madampar', required=False, default=None, help='Madam parameter file' )

parser.add_argument( '--outdir', required=False, default='.', help='Output directory' )
parser.add_argument( '--debug', required=False, default=False, action='store_true', help='Write focalplane image and other diagnostics' )
args = parser.parse_args()

samplerate = float(args.samplerate)
spinperiod = float(args.spinperiod)
spinangle = float(args.spinangle)
precperiod = float(args.precperiod)
precangle = float(args.precangle)

wafer_mm = float(args.wafersize)
wafer_deg = float(args.waferang)

hwprpm = float(args.hwprpm)
hwpstep = None
if args.hwpstep is not None:
    hwpstep = float(args.hwpstep)
hwpsteptime = float(args.hwpsteptime)

start = MPI.Wtime()
fp = None
freq = args.frequency

if int(args.frequency) > 250:
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
stop = MPI.Wtime()
elapsed = stop - start
if comm.comm_world.rank == 0:
    print("Create focalplane:  {:.2f} seconds".format(stop-start))
start = stop

if args.outdir != '.':
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

if args.debug:
    if comm.comm_world.rank == 0:
        import matplotlib.pyplot as plt
        import litebird.vis as lbv
        fig = plt.figure( figsize=(36,18), dpi=100 )
        ax = fig.add_subplot(1, 1, 1)
        lbv.view_focalplane(fp, ax, freq=args.frequency)
        outfile = os.path.join(args.outdir, 'focalplane.png')
        plt.savefig(outfile)

# Read in madam parameter file
pars = {}

if comm.comm_world.rank == 0:
    if args.madampar is not None:
        pat = re.compile(r'\s*(\S+)\s*=\s*(\S+)\s*')
        comment = re.compile(r'^#.*')
        with open(args.madampar, 'r') as f:
            for line in f:
                if not comment.match(line):
                    result = pat.match(line)
                    if result:
                        pars[result.group(1)] = result.group(2)
    else:
        pars[ 'temperature_only' ] = 'F'
        pars[ 'force_pol' ] = 'T'
        pars[ 'kfirst' ] = 'T'
        pars[ 'base_first' ] = 360.0
        pars[ 'fsample' ] = samplerate
        pars[ 'nside_map' ] = 1024
        pars[ 'nside_cross' ] = 1024
        pars[ 'nside_submap' ] = 64
        pars[ 'write_map' ] = 'T'
        pars[ 'write_binmap' ] = 'T'
        pars[ 'write_matrix' ] = 'T'
        pars[ 'write_wcov' ] = 'T'
        pars[ 'write_hits' ] = 'T'
        pars[ 'kfilter' ] = 'F'
        pars[ 'run_submap_test' ] = 'T'
        pars[ 'path_output' ] = args.outdir

pars = comm.comm_world.bcast(pars, root=0)

# madam only supports a single observation.  Normally
# we would have multiple observations with some subset
# assigned to each process group.

# The distributed timestream data
data = toast.Data(comm)

# create the TOD for this observation
tod = lbt.SimSimple(
    mpicomm=comm.comm_group, 
    focalplane=fp,
    freq=freq,
    start=0.0,
    samplerate=samplerate, 
    spinperiod=spinperiod, 
    spinangle=spinangle, 
    precperiod=precperiod, 
    precangle=precangle, 
    hwprpm=hwprpm,
    hwpsteptime=hwpsteptime,
    hwpstep=hwpstep,
    obs=float(args.obs),
    gap=float(args.gap),
    numobs=int(args.numobs),
    axisincr=1.0
)

# Create the noise model for this observation

noise = lbt.SimNoise(fp.detectors(freq=freq))

# normally we would get the intervals from somewhere else, but since
# we are using a single observation with a single TOD, we get that
# information from the TOD.

ob = {}
ob['id'] = 'mission'
ob['tod'] = tod
ob['intervals'] = tod.valid_intervals
ob['baselines'] = None
ob['noise'] = noise

data.obs.append(ob)

comm.comm_world.barrier()
stop = MPI.Wtime()
elapsed = stop - start
if comm.comm_world.rank == 0:
    print("Metadata queries took {:.3f} s".format(elapsed))
start = stop

# cache the data in memory

cache = tt.OpCopy()
cache.exec(data)

comm.comm_world.barrier()
stop = MPI.Wtime()
elapsed = stop - start
if comm.comm_world.rank == 0:
    print("Data read and cache took {:.3f} s".format(elapsed))
start = stop

# make a Healpix pointing matrix.  By setting highmem=False,
# we purge the detector quaternion pointing to save memory.
# If we ever change this pipeline in a way that needs this
# pointing at a later stage, we need to set highmem=True and
# run at fewer processes per node.

mode = 'IQU'
if pars['temperature_only'] == 'T':
    mode = 'I'
pointing = lbt.OpPointingSim(nside=int(pars['nside_map']), mode=mode, highmem=False, samplerate=samplerate, hwprpm=hwprpm, hwpstep=hwpstep, hwpsteptime=hwpsteptime)
pointing.exec(data)

comm.comm_world.barrier()
stop = MPI.Wtime()
elapsed = stop - start
if comm.comm_world.rank == 0:
    print("Pointing Matrix took {:.3f} s, mode = {}".format(elapsed,mode))
start = stop

# simulate noise

nse = tt.OpSimNoise(stream=0)
nse.exec(data)

comm.comm_world.barrier()
stop = MPI.Wtime()
#elapsed = stop - start
if comm.comm_world.rank == 0:
    print("Noise simulation took {:.3f} s".format(elapsed))
start = stop

# for now, we use noise weights based on the hardware model
# NETs.
hw = lb.Hardware()
detweights = {}
for d in tod.detectors:
    net = hw.NET(d)
    detweights[d] = 1.0 / (net * net)

## Set up MADAM map making.  By setting highmem=False, we will
## purge the pointing matrix after copying it into the madam
## buffers.  This is ok, as long as madam is the last step of 
## the pipeline.
#
#madam = tm.OpMadam(params=pars, detweights=detweights, highmem=False)
#madam.exec(data)
#
#comm.comm_world.barrier()
#stop = MPI.Wtime()
#elapsed = stop - start
#if comm.comm_world.rank == 0:
#    print("TOD simulation and Mapmaking took {:.3f} s".format(elapsed))
#    elapsed = stop - global_start
#    print("Total Time:  {:.2f} seconds".format(elapsed))
#


