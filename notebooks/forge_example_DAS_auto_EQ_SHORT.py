#%%
import numpy as np
from halo import Halo


# to set the path for importing HADES modules
import sys
sys.path.append("../source/")

from LatLongUTMconversion import LLtoUTM, UTMtoLL
from hades_utils import  distance_calculation, sortout, flip_sign
from hades_location import hades_location
from hades_input import hades_input

from tqdm import tqdm
import itertools
import time

import matplotlib.pyplot as plt

#%%
import pandas as pd
from hades_rotation import Cluster
from LatLongUTMconversion import LLtoUTM, UTMtoLL
import rotations_utils as ru

from scipy.spatial.transform import Rotation as R
from rotations_utils import define_axis_vectors
# Plot params
import matplotlib as mpl
import cmasher as cmr
from matplotlib.gridspec import GridSpec

##########################################################
#%%
def get_random_pairs(numbers): 
  # Generate all possible non-repeating pairs 
  pairs = list(itertools.combinations(numbers, 2)) 
  weightrange = np.linspace(0.5, 1, len(numbers))

  keep_pairs = []
  weights = []
  for i in range(len(pairs)):
    tmp = (pairs[i][1] - pairs[i][0])
    if tmp > 500:
      keep_pairs.append(pairs[i])
      weights.append(weightrange[int(pairs[i][1] - pairs[i][0])])


  # do a significant downscale of these
  keep_pairs = keep_pairs[:]

  return keep_pairs, weights

def z_quat(theta, v0):
  """Takes the first angle theta and computes the quaternion around axis v0."""

  v0 = v0

  r_a = R.from_quat([v0[0] * np.sin(theta/2),     # compute quaternions
                              v0[1] * np.sin(theta/2), 
                              v0[2] * np.sin(theta/2), 
                              np.cos(theta/2) ])

  return r_a


def a_quat(theta, v1):
    """Takes the second angle theta and computes the quaternion around axis v0."""

    v1 = v1

    r_b = R.from_quat([v1[0] * np.sin(theta/2),     # compute quaternions
                                v1[1] * np.sin(theta/2), 
                                v1[2] * np.sin(theta/2), 
                                np.cos(theta/2) ])

    return r_b


def b_quat(theta, v2):

    v2 = v2

    r_c = R.from_quat([v2[0] * np.sin(theta/2),     # compute quaternions
                                v2[1] * np.sin(theta/2), 
                                v2[2] * np.sin(theta/2), 
                                np.cos(theta/2) ])

    return r_c


def invert_rotations_spatial(cluster, rotations, station, bary):
    """
    Inverts the rotation
    """

    bary = bary
    v0, v1, v2 = define_axis_vectors(station, bary)

    ca = b_quat(np.radians(rotations[2]), v2).inv().apply(cluster - bary) 
    cb = a_quat(np.radians(rotations[1]), v1).inv().apply(ca ) 
    cF = z_quat(np.radians(rotations[0]), v0).inv().apply(cb ) + bary
    
    

    return cF

#%%

spinner = Halo(text='Optimizing horrible misfit function', spinner='dots')

vp = 5000
vs = vp/1.9

# compute channel pairs
channel_pairs, weights = get_random_pairs(np.arange(344,865))[:100]    
no_of_evts = 13

# input files
general_path = './'
station_file = 'data/inputs/EQ_DAS_stats_lowered_SHORT.txt'
input_file = 'data/inputs/FORGE_picks_SHORT_FORU.dat'

output_path = 'data/outputs/'
output_filename = 'EQ_output_FORGE_rel'

station_names = [f'DAS{str(i).zfill(2)}' for i in range(1049)]
foru = ['FORU' for i in range(100)]

# Declare which sets of two stations you want to use:
# stations = [[foru[i], station_names[channel_pairs[i][1]]] for i in range(100)]     # two selected stations
stations = [[foru[i], station_names[channel_pairs[i][1]]] for i in range(10)] 

# empty matrices to fill for later analysis
huge_locmat = np.zeros([no_of_evts, 3, len(stations)])
rect_correct = []
save_sps = {}

optall = True     # True: use all events for finding rotation, False: use only reference events
master_evts = None    # if optall = False, fill in your reference/master events [list]
station_axis=0
theta_vect=np.zeros([len(stations),3])
bary_UTM = LLtoUTM(23,38.4773,-112.8668)[1:]
bary=np.array([LLtoUTM(23,38.4773,-112.8668)[1], LLtoUTM(23,38.4773,-112.8668)[2],973.1])

wdepths = pd.read_csv('./data/inputs/w78_UTM.csv', delimiter=';')

well_coords_sph = np.array([UTMtoLL(23, wdepths.NS[i], wdepths.EW[i], '12S') for i in range(len(wdepths))])

wlats = np.linspace(well_coords_sph[0][0], well_coords_sph[-1][0], 977)
wlons = np.linspace(well_coords_sph[0][1], well_coords_sph[-1][1], 977)
wdeps = np.linspace(0, 996, 977)

w_ew_ns = np.array([LLtoUTM(23, wlats[i], wlons[i])[1:] for i in range(977)])

w_coords_UTM = np.array([w_ew_ns[:,0], w_ew_ns[:,1],wdeps]).T


f_dist = 0.5
#%%

for i in tqdm(range(len(stations))):
    hobj = hades_input(
    data_path = general_path,          # the general path to the data
    event_file = input_file,        # the input event file path
    station_file = station_file,    # the station file path
    sta_select = stations[i]           # your two stations
    )

    # # # First we compute the relative distances
    hobj.distance_calculation(vp,vs,stations[i])
    # # And then we do absolute locations. First make a location instance:
    hobj.relative_frame(vp,vs,stations[i],y_ref=-1,z_ref=1,fixed_depth=f_dist)   

    # And then we do absolute locations. First make a location instance:

    hloc = hades_location(input_obj=hobj, output_path=output_path, output_frame='latlon')
    hloc.location(output_filename, mode='rel', plot=False)

    # # And then we do absolute locations. First make a location instance:
    # hloc = hades_location(input_obj=hobj, output_path=output_path, output_frame='latlon')
    # hloc.location(output_filename, mode='abs', plot=False)

    refprints = hloc.catalogue[:4, 0]
    reflist = []
    for k in range(len(refprints)):
        tmp = int(refprints[k].split('#R')[1])
        reflist.append(tmp)

    las = sortout(np.array(np.float64(hloc.catalogue[:,1])), reflist)
    los = sortout(np.array(np.float64(hloc.catalogue[:,2])), reflist)

    lat_calc, lon_calc = np.zeros(len(hloc.catalogue)), np.zeros(len(hloc.catalogue))
    for j in range(len(hloc.catalogue)):
        _, lon_calc[j], lat_calc[j] = LLtoUTM(23,  las[j], los[j])

    dep_calc = sortout(np.array(np.float64(hloc.catalogue[:,3])), reflist)

    huge_locmat[:,0,i] = lon_calc.copy()
    huge_locmat[:,1,i] = lat_calc.copy()
    huge_locmat[:,2,i] = dep_calc.copy()*1000

    master_evts = np.array(reflist)-1

    tstp= np.array([hloc.evtsps[stations[i][0]], hloc.evtsps[stations[i][1]]]) #+ np.random.normal(0, 0.01, (3,50))
    dtps_obs = tstp.copy()         # observed ts-tp of n events shape: [n] 
    stavect = np.array([w_coords_UTM[channel_pairs[i][0],:], w_coords_UTM[channel_pairs[i][0],:]])
  
    cl = huge_locmat[:,:,i].copy()

    bary, v0, v1, v2, cluster_opt, dtps_obs, stavect = ru.setup(optall, master_evts, None, dtps_obs, 
                cl, stavect, station_axis, vs, vp, baryp=bary)

    
    instance_cluster = Cluster(cluster=cl, 
                        stations=np.array(stavect), axis=station_axis, bary=bary, subcluster=cl, 
                        vs=vs, vp=vp, dtps_org=dtps_obs, cluster_true=None)


    spinner.start()
    test_thetas = instance_cluster.optimize_rotations(prior=[0.,0.,0.], method='diffev', optall=optall)
    spinner.stop()



    rect=instance_cluster.rect
    thetas_final = test_thetas


    cluster_comp = instance_cluster.invert_rotations_spatial(cl, thetas_final)


    huge_locmat[:,0,i] = cluster_comp[:,0]
    huge_locmat[:,1,i] = cluster_comp[:,1]
    huge_locmat[:,2,i] = cluster_comp[:,2]

    theta_vect[i,:]=thetas_final

    rect = hloc.rect

    rect_correct.append(rect)

    save_sps = hloc.evtsps
    if i>0:
        if rect_correct[i] > np.array(rect_correct[:i]).max():
            print(f'Better rect {i}')
            save_sps.clear()
            save_sps = hloc.evtsps

    time.sleep(0.01)

#%%

ref = master_evts
# cluster = cl
best_run = np.array(rect_correct).argmin()
# cluster_comp = instance_cluster.invert_rotations_spatial(cl, theta_vect[best_run,:])
cluster_comp = huge_locmat[:,:,best_run]

allons = huge_locmat[:,0,:].flatten()
allats = huge_locmat[:,1,:].flatten()
alldeps = huge_locmat[:,2,:].flatten()

# inter-iteration error
iie_x = np.std(huge_locmat[:,0,:], axis=1)
iie_y = np.std(huge_locmat[:,1,:], axis=1)
iie_z = np.std(huge_locmat[:,2,:], axis=1)

iie_total = np.average([iie_x,iie_y,iie_z],axis=0)


np.savetxt('./data/outputs/best_run_EQ_SHORT.csv', huge_locmat[:,:,best_run], delimiter=';')
np.savetxt('./data/outputs/allons_EQ_SHORT.csv', allons, delimiter=';')
np.savetxt('./data/outputs/allats_EQ_SHORT.csv', allats, delimiter=';')
np.savetxt('./data/outputs/aldeps_EQ_SHORT.csv', alldeps, delimiter=';')
np.savetxt('./data/outputs/master_evt_EQ_SHORT.csv', huge_locmat[ref,:,0], delimiter=';')
# np.savetxt('./data/outputs/truth_lon_EQ_SHORT.csv', cluster[:,0], delimiter=';')
# np.savetxt('./data/outputs/truth_lat_EQ_SHORT.csv', cluster[:,1], delimiter=';')
# np.savetxt('./data/outputs/truth_dep_EQ_SHORT.csv', cluster[:,2], delimiter=';')
np.savetxt('./data/outputs/iie_total_EQ_SHORT.csv', iie_total, delimiter=';')

# font
# Edit the font, font size, and axes width
mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2

plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.labelsize'] = 12

# master_evt = np.array([allons[ref], allats[ref], alldeps[ref]])
das_a = [335780.840200, 4262991.991000]

bins = int(np.sqrt(len(allons)))
cmap = cmr.get_sub_cmap(cmr.arctic_r, 0.05,1)

# now put all of the figures in one large one
cm = 1/2.54  # centimeters in inches
fig = plt.figure(figsize=(17*cm*2, 10*cm*2))

gs = GridSpec(2, 2)
ax1 = fig.add_subplot(gs[:,:-1])
ax2 = fig.add_subplot(gs[:-1,-1])
ax3 = fig.add_subplot(gs[-1,-1])

# AX1 -------------------------------------------------------------------------
fig.suptitle(f"FORGE 2019 Earthquake results: DAS + FORU", fontsize=15)
ax1.set_aspect("equal")

h,_,_,im=ax1.hist2d(allons, allats, bins=bins, cmin=1, cmap=cmap, vmin=0, vmax=100, range=([336000,338500], [4.2595e6,4.2620e6]))
ax1.scatter(cluster_comp[ref[0],0], cluster_comp[ref[0],1], s=75, edgecolor='orange', facecolor='yellow', label='Reference event', zorder=20)
sc=ax1.scatter(cluster_comp[:,0], cluster_comp[:,1], s=iie_total, facecolor='None', edgecolor='orange', zorder=15, label='best estimate')

ax1.scatter(das_a[0], das_a[1], marker='s', s=150, c='k', label='DAS cable')
# ax1.scatter(cluster[:,0], cluster[:,1], marker='o', facecolor='None', edgecolor='k')
ax1.grid()
ax1.set_ylim(4.2629e6, 4.2635e6)
plt.colorbar(im,ax=ax1, shrink=.5, label='Location count')
ax1.set_xlabel("Easting [m]", fontsize=12)
ax1.set_ylabel("Northing [m]", fontsize=12)


# legend1 = ax1.legend(
#                     loc="upper right")
# ax1.add_artist(legend1)

for area in [30, 200, 300]:
    ax1.scatter([], [], c='k', alpha=0.3, s=area,
                label='$\sigma$ = ' + str(area) + ' m')
ax1.legend(scatterpoints=1, frameon=True, labelspacing=1, title='City Area', loc='upper right')

# handles, labels = sc.legend_elements(prop="sizes", alpha=0.6)
# legend2 = ax1.legend(handles, labels, loc="lower right", title="Sizes")
# ax1.add_artist(legend2)

ax1.set_xlim(335500,338500)
ax1.set_ylim(4.2635e6, 4.2595e6)
ax1.invert_yaxis()
# -----------------------------------------------------------------------------


# AX2 -------------------------------------------------------------------------
ax2.set_aspect("equal")
bb,_,_,im=ax2.hist2d(allons, alldeps, bins=bins, cmin=1, cmap=cmap, vmin=0, vmax=100, range=([336000,338500], [0,2500]))
# ax2.scatter(cluster[:,0], cluster[:,2], marker='o', facecolor='None', edgecolor='k')
ax2.scatter(cluster_comp[:,0], cluster_comp[:,2], s=iie_total, facecolor='None', edgecolor='orange', zorder=15)
ax2.scatter(cluster_comp[ref[0],0], cluster_comp[ref[0],2], s=75, edgecolor='orange', facecolor='yellow', label='Reference event', zorder=20)
ax2.plot([das_a[0], das_a[0]], [0, 985], c='k', linewidth=4)

ax2.set_xlim(334500,338500)
ax2.set_ylim(0,2250)
ax2.grid()

ax2.invert_yaxis()
ax2.set_xlabel("Easting [m]", fontsize=12)
ax2.set_ylabel("Depth [m]", fontsize=12)
# -----------------------------------------------------------------------------


# AX3 -------------------------------------------------------------------------
ax3.set_aspect('equal')
aa,_,_,im=ax3.hist2d(allats, alldeps, bins=bins, cmin=1,cmap=cmap, vmax=100, range=([4.2595e6,4.2620e6], [0,2500]),
 vmin=0)
# ax3.scatter(cluster[:,1], cluster[:,2], marker='o', facecolor='None', edgecolor='k')
ax3.scatter(cluster_comp[ref[0],1], cluster_comp[ref[0],2], s=75, edgecolor='orange', facecolor='yellow', label='Reference event', zorder=20)
ax3.scatter(cluster_comp[:,1], cluster_comp[:,2],  s=iie_total, facecolor='None', edgecolor='orange', zorder=15, label='best estimate')
ax3.set_xlim(4.2635e6, 4.2595e6)

ax3.plot([das_a[1], das_a[1]], [0, 985], c='k', linewidth=4)
ax3.grid()
ax3.set_ylim(2250,0)
ax3.set_xlabel("Northing [m]", fontsize=12)
ax3.set_ylabel("Depth [m]", fontsize=12)
# -----------------------------------------------------------------------------

fig.tight_layout(pad=0.5)
plt.savefig('./data/images/test_EQ_SHORT.png',dpi=300)

#%%

flipped = flip_sign(huge_locmat[:,:,best_run], 1,1,1, b=ref[0])
reflist = ref

# cl_abs = cluster.copy()
cl = cl

# Fill in original cluster and rotated cluster (or the cluster you intend to optimize)
# cluster =   None    # optional! n events, lat/lon/depth shape: [n, 3]
dtps_obs = tstp

# Fill in your stations
stations = w_coords_UTM[channel_pairs[best_run],:]      # m stations, lat/lon/depth shape [m, 3]
station_axis =  0  # choose index of the station by which you want to define your rotation axis

sta1 = stations[0]
sta2 = stations[1]
bary, v0, v1, v2, cluster_opt, dtps_obs, stavect = ru.setup(optall, master_evts, None, dtps_obs, 
                cl, stations, station_axis, vs, vp, baryp=bary)

instance_cluster_refine = Cluster(cluster=cl, 
                        stations=np.array(stavect), axis=station_axis, bary=bary, subcluster=cl, 
                        vs=vs, vp=vp, dtps_org=dtps_obs, cluster_true=cl)

spinner.start()
test_thetas_refine = instance_cluster_refine.optimize_rotations(prior=[0.,0.,0.], method='diffev', optall=optall)
spinner.stop()
print("Optimizer REFINER outcome in degrees (three rotations)", test_thetas_refine)

cluster_comp_refine = instance_cluster_refine.invert_rotations_spatial(cl, test_thetas_refine)

flipped = flip_sign(cluster_comp_refine, 1,1,1, b=ref[0])


#%%
plt.figure(figsize=(15,10))

ax = plt.subplot(1,3,1)
plt.grid()
ax.scatter(sta1[0], sta1[1], s=100, marker='v', c='r')
ax.scatter(sta2[0], sta2[1], s=100, marker='v', c='r')

ax.scatter(cl[:,0], cl[:,1], facecolor='None', edgecolor='darkgreen', label='cluster_rot') 
# ax.scatter(cl_abs[:,0], cl_abs[:,1], facecolor='None',edgecolor='black', label='cluster_true') 
im=ax.scatter(flipped[:,0], flipped[:,1], s=iie_total, 
           c=tstp[0,:], label='cluster_comp', cmap='bwr')
ax.set_aspect(1)
ax.scatter(bary[0], bary[1], s=100, c='yellow', label='barycentre')

ax = plt.subplot(1,3,2)
plt.grid()
ax.scatter(sta1[0], sta1[2], marker='v', c='r')
ax.scatter(sta2[0], sta2[2], marker='v', c='r')
ax.scatter(cl[:,0], cl[:,2], facecolor='None', edgecolor='darkgreen', label='cluster_rot') 
# ax.scatter(cl_abs[:,0], cl_abs[:,2], facecolor='None',edgecolor='black', label='cluster_true') 
im=ax.scatter(flipped[:,0], flipped[:,2], s=iie_total, 
           c=tstp[0,:], label='cluster_comp', cmap='bwr')
# ax.set_aspect(1)
ax.scatter(bary[0], bary[2], s=100, c='yellow', label='barycentre')
ax.invert_yaxis()

ax = plt.subplot(1,3,3)

ax.scatter(sta1[1], sta1[2], marker='v', c='r')
ax.scatter(sta2[1], sta2[2], marker='v', c='r')

plt.grid()
ax.scatter(cl[:,1], cl[:,2], facecolor='None', edgecolor='darkgreen', label='cluster_rot') 
# ax.scatter(cl_abs[:,1], cl_abs[:,2], facecolor='None',edgecolor='black', label='cluster_true') 
im=ax.scatter(flipped[:,1], flipped[:,2], s=iie_total, 
           c=tstp[0,:], label='cluster_comp', cmap='bwr')
im=ax.scatter(flipped[ref,1], flipped[ref,2], c='yellow', zorder=12)
# ax.set_aspect(1)

ax.scatter(bary[1], bary[2], s=100, c='yellow', label='barycentre')
ax.invert_yaxis()
ax.invert_xaxis()

plt.legend()
plt.colorbar(im)
plt.savefig('./data/images/refined_EQ_SHORT.png', dpi=300)
#plt.show()

np.savetxt('./data/outputs/refined_EQ_SHORT.csv', cluster_comp_refine, delimiter=';')


# %%



AL_sph = pd.read_csv('./data/inputs/Ariel_locs_txt.txt', delimiter=';')

al_UTM = np.array([LLtoUTM(23, AL_sph.lat[i], AL_sph.lon[i])[1:] for i in range(len(AL_sph))])
all_UTM = np.array([al_UTM[:,0], al_UTM[:,1], np.array(AL_sph.dep)*1000]).T

hehe_test = [0,2,5,9,10,12,18,32,33,35,36,37]
# tstsp_htest = np.array([tstp[0][hehe_test], tstp[1][hehe_test]]).T
# ru.pca_theta_calculation(all_UTM, tstsp_htest, stations, plot=False)

plt.figure()
plt.grid()
plt.scatter(huge_locmat[:,0,best_run], huge_locmat[:,1,best_run], s=(tstp[0]*10)**2)
plt.scatter(huge_locmat[ref,0,best_run], huge_locmat[ref,1,best_run], c='yellow',zorder=10)
plt.scatter(huge_locmat[ref[0],0,best_run], huge_locmat[ref[0],1,best_run], c='yellow')
plt.scatter(huge_locmat[ref[-1],0,best_run], huge_locmat[ref[-1],1,best_run], c='r')
plt.scatter(all_UTM[:,0], all_UTM[:,1], c='k', s=100, marker='v')

plt.scatter(huge_locmat[hehe_test,0,best_run], huge_locmat[hehe_test,1,best_run])


tr = [2,3,4,10]

plt.scatter(all_UTM[tr,0], all_UTM[tr,1], s=100, marker='x' )

for i in range(len(all_UTM)):
    plt.annotate(f'{i}', (all_UTM[i,0]+50, all_UTM[i,1]+10))
# plt.scatter(all_UTM[[8],0], all_UTM[[8],1])

plt.show()



# # %%




# %%
