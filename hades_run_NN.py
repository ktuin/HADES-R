#%%
from LatLongUTMconversion import LLtoUTM, UTMtoLL
import matplotlib.pyplot as plt
from hades_input import hades_input
from hades_location import hades_location
import pandas as pd
import numpy as np
from hades_utils import sortout, find_element_in_list


#### Give rotation another look maybe   :)   :(


#%%
n_evs = 100

sta = pd.read_csv('./NNtest/stafile_NNtest_planar.txt', delimiter=' ', header=None)[0] 
stafile = pd.read_csv('./NNtest/stafile_NNtest_planar.txt', delimiter=' ', header=None)

list_a =  list(sta)
csize = 15

#%%

def random_pairs_np(number_list): 
    return np.random.choice(number_list, 2, replace=False) 

comb_a = [random_pairs_np(sta) for i in range(csize)] 
# comb_a = np.array(['S0', 'S26'])

#%% Import some files

data_path='./'
input_file='./NNtest/evfile_NNtest_planar.dat' 
sta_file='./NNtest/stafile_NNtest_planar.txt'
dist_file='./NNtest/distfile_NNtest_planar.txt'

distances = np.loadtxt(dist_file, delimiter=';')


#%%

huge_locmat = np.zeros([n_evs, 3, csize])

Vp=5200
Vs=Vp/1.9

delete_iteration = []

for i in range(csize):
    # try:
    stations = [comb_a[i][0], comb_a[i][1]]#['S0', 'S26']#

    testname = f'NN_{stations[0]}_{stations[1]}_planar'
    out_file=f'./NNtest/outputs/FORGE_{testname}' 

    hobj=hades_input(data_path,input_file,sta_file, stations)
    hobj.relative_frame(Vp,Vs,stations,y_ref=1,z_ref=1,fixed_depth=-0.56)
    hobj.distance_calculation(Vp,Vs,stations, dist=distances)
    hloc=hades_location(hobj,'./')
    hloc.location(out_file,mode='rel', plot=True)
    # hloc.location(out_file,mode='abs', plot=True)


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


        
    # except: 

    #     huge_locmat[:,0,i] = np.zeros(n_evs)
    #     huge_locmat[:,1,i] = np.zeros(n_evs)
    #     huge_locmat[:,2,i] = np.zeros(n_evs)

    #     delete_iteration.append(i)


    #     print(f'SVD did not converge at loop {i} with station combination {stations}')

# if len(delete_iteration) > 0:
#     for o in range(len(delete_iteration)):
#         np.delete(huge_locmat, delete_iteration[o], 2)


allons = huge_locmat[:,0,:].flatten()
allats = huge_locmat[:,1,:].flatten()
alldeps = huge_locmat[:,2,:].flatten()
#%%
huge_newmat = np.array([allons, allats, alldeps])

np.savetxt(f'./NN_RESULTS_UTM_planar.csv', huge_newmat.T, delimiter=';')

# np.savetxt(f'./new_config/outputs/all_locations_test_HOMOG_comb_{csize}.csv', delete_iteration ,delimiter=';')


##########################
# START PLOTTING
##########################
#%%
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# schlum = pd.read_csv('./new_config/data/ms_april-may-2019_loc_error_sp_qc.dat', delimiter=';')
# schlum_evs = np.arange(57,373)   #[75,76,81,83,87,88,98,107,117,120,124,135,147,149,171,174,193,200,211,213,78,246,256,259,260,275,297,310,351,355,363,364]

evs = pd.read_csv('./NNtest/realDATA-NNTEST_planar.txt', delimiter=';')
lat_UTM, lon_UTM, depth_UTM = evs.lat, evs.lon, evs.dep


# deleted = np.loadtxt('./new_config/outputs/DELETED_ITERATIONS_HOMOG_comb_12.txt', delimiter=' ')


# allats = np.loadtxt('./new_config/outputs/all_locations_test_HOMOG_comb_1200_LAT.csv', delimiter=';')
allats = allats[allats.nonzero()]
# allons = np.loadtxt('./new_config/outputs/all_locations_test_HOMOG_comb_1200_LON.csv', delimiter=';')
allons = allons[allons.nonzero()]
# alldeps = np.loadtxt('./new_config/outputs/all_locations_test_HOMOG_comb_1200_DEP.csv', delimiter=';')
alldeps = alldeps[alldeps.nonzero()]


#%%


das_a = [335780.840200, 4262991.991000]
w16A = pd.read_csv('./new_config/data/w16A_UTM.csv', delimiter=';')
das_b = [335865.45, 4262983.529]
das_c = [335456.3361, 4263398.416]
reflist = np.array(reflist)


f, ax = plt.subplots(figsize=(8, 8))
ax.set_title(f"New stimulation synthetics", fontsize=15)
# ax.set_aspect("equal")
ax.grid()

sns.kdeplot(allons, alldeps, levels=30, fill=True)
ax.scatter(allons[:4], alldeps[:4], c='green')
ax.scatter(lon_UTM[:n_evs], depth_UTM[:n_evs], s=10, c='k',zorder=15)
ax.scatter(lon_UTM[reflist], depth_UTM[reflist], s=20, c='yellow',zorder=18, label='Master events')
# plt.scatter(huge_locmat[:,0,2], huge_locmat[:,2,2], s=10, c='orange', zorder=15)
# for i in range(n_evs):
#     plt.plot([ 
#         huge_locmat[:,0,2][i], lon_UTM[i]], 
#         [
#         huge_locmat[:,2,2][i], depth_UTM[i]], c='lightgray'

#     )
# ax.set_xlim(334000,352000)

ax.plot([das_a[0], das_a[0]], [0, 1356], c='k', linewidth=4)

# ax.scatter(np.mean(allons.reshape([n_evs,csize*3]), axis=1)[reflist-1], np.mean(alldeps.reshape([n_evs,csize*3]), axis=1)[reflist-1], c='darkgreen', zorder=17)

ax.invert_yaxis()
ax.set_xlabel("Lon UTM [m]", fontsize=15)
ax.set_ylabel("Depth [m]", fontsize=15)

plt.savefig(f'./new_config/images/test_HOMOG_LonDep_N{csize*3}.png',dpi=300)
plt.show()

#%%

f, ax = plt.subplots(figsize=(8, 8))
ax.set_title(f"New stimulation synthetics", fontsize=15)
# ax.set_aspect("equal")
ax.grid()

sns.kdeplot(allats, alldeps, levels=30, fill=True)
ax.scatter(lat_UTM[:n_evs], depth_UTM[:n_evs], s=10, c='k',zorder=15)
ax.scatter(lat_UTM[reflist], depth_UTM[reflist], s=20, c='yellow',zorder=18)

# plt.scatter(huge_locmat[:,1,2], huge_locmat[:,2,2], s=10, c='orange', zorder=15, label='best estimate')
# ax.scatter(np.mean(allats.reshape([n_evs,csize*3]), axis=1)[reflist-1], np.mean(alldeps.reshape([n_evs,csize*3]), axis=1)[reflist-1], c='darkgreen', zorder=17)

# for i in range(n_evs):
#     plt.plot([ 
#         huge_locmat[:,1,2][i], lat_UTM[i]], 
#         [
#         huge_locmat[:,2,2][i], depth_UTM[i]], c='lightgray'

#     )
# ax.set_xlim(334000,352000)

# ax.plot([stafile[1][stats_a[1]], stafile[3][stats_a[-1]]])


ax.plot([das_a[1], das_a[1]], [0, 1356], c='k', linewidth=4)

ax.invert_yaxis()
# ax.set_xlim(4.2629e6, 4.2635e6)
ax.set_xlabel("Lat UTM [m]", fontsize=15)
ax.set_ylabel("Depth [m]", fontsize=15)
plt.savefig(f'./new_config/images/test_HOMOG_LatDep_N{csize*3}.png',dpi=300)
plt.show()

#%%

f, ax = plt.subplots(figsize=(8, 8))
ax.set_title(f"New stimulation synthetics", fontsize=15)
# ax.set_aspect("equal")
ax.grid()
sns.kdeplot(allons, allats, levels=30, fill=True)
ax.scatter(lon_UTM[:n_evs], lat_UTM[:n_evs], s=10, c='k',zorder=15, label='Ground truth')
ax.scatter(lon_UTM[reflist], lat_UTM[reflist], s=20, c='yellow',zorder=18, label='Master events')

# plt.scatter(huge_locmat[:,0,2], huge_locmat[:,1,2], s=10, c='orange', zorder=15, label='best estimate')
# ax.scatter(np.mean(allons.reshape([n_evs,csize*3]), axis=1)[reflist-1], np.mean(allats.reshape([n_evs,csize*3]), axis=1)[reflist-1], c='darkgreen', zorder=17)


# for i in range(n_evs):
#     plt.plot([ 
#         huge_locmat[:,0,2][i], lon_UTM[i]], 
#         [huge_locmat[:,1,2][i], lat_UTM[i]], c='lightgray'

#     )

# ax.plot([stafile[1][stats_a[0]], stafile[2][stats_a[-1]]])

ax.scatter(das_a[0], das_a[1], marker='s', s=150, c='k', label='DAS cable')

# ax.set_ylim(4.2629e6, 4.2635e6)

ax.set_xlabel("Lon UTM [m]", fontsize=15)
ax.set_ylabel("Lat UTM [m]", fontsize=15)
plt.legend(fontsize=15)
# ax.set_xlim(335200,336200)
# plt.savefig(f'./new_config/images/test_HOMOG_LatLon_N{csize*3}.png',dpi=300)
plt.show()



#%%

dlat, dlon, ddep = [], [], []

for i in range(csize):
    dlat.append(np.abs(huge_locmat[:,1,i]-lat_UTM))
    dlon.append(np.abs(huge_locmat[:,0,i]-lon_UTM))
    ddep.append(np.abs(huge_locmat[:,2,i]-depth_UTM))

# le_lats=
# le_lons=
# le_deps=



# %%

f, ax = plt.subplots(figsize=(8, 8))
plt.grid()
ax.scatter(lon_UTM[:n_evs], depth_UTM[:n_evs], s=30, c='k',zorder=15)
ax.scatter(huge_locmat[:,0,2], huge_locmat[:,2,2], s=30, c='orange', zorder=15)
for i in range(n_evs):
    ax.plot([ 
        huge_locmat[:,0,2][i], lon_UTM[i]], 
        [
        huge_locmat[:,2,2][i], depth_UTM[i]], c='lightgray'

    )
plt.xlabel('LON')
plt.ylabel('DEP')
plt.savefig('hades_NN_bestloc_LONDEP.png', dpi=200)
plt.show()


# %%
import matplotlib as mpl
f, ax = plt.subplots(figsize=(8, 8))
ax.set_title(f"New stimulation synthetics", fontsize=15)
# ax.set_aspect("equal")

# sns.kdeplot(allons, alldeps, levels=30, fill=True)
bb,_,_,im=ax.hist2d(allons, alldeps, bins=np.int(np.sqrt(len(allons))), cmin=1, cmap='Blues',norm=mpl.colors.LogNorm())
ax.scatter(lon_UTM[:n_evs], depth_UTM[:n_evs], s=10, c='k',zorder=15)
ax.scatter(lon_UTM[reflist], depth_UTM[reflist], s=20, c='yellow',zorder=18, label='Master events')
# ax.scatter(huge_locmat[:,0,2], huge_locmat[:,2,2], s=10, c='orange', zorder=15)
ax.scatter(allons[:4], alldeps[:4], s=50, c='green', zorder=22)

# for i in range(n_evs):
#     plt.plot([ 
#         huge_locmat[:,0,2][i], lon_UTM[i]], 
#         [
#         huge_locmat[:,2,2][i], depth_UTM[i]], c='lightgray'

#     )
# ax.set_xlim(334000,352000)

ax.plot([das_a[0], das_a[0]], [0, 1356], c='k', linewidth=4)

ax.set_xlim(335600,337400)
ax.set_ylim(0,1500)

# ax.scatter(np.mean(allons.reshape([n_evs,csize*3]), axis=1)[reflist-1], np.mean(alldeps.reshape([n_evs,csize*3]), axis=1)[reflist-1], c='darkgreen', zorder=17)
ax.grid()

ax.invert_yaxis()
ax.set_aspect(0.8)
ax.set_xlabel("Lon UTM [m]", fontsize=15)
ax.set_ylabel("Depth [m]", fontsize=15)
plt.colorbar(im,ax=ax, shrink=.3, label='Location count')
# plt.savefig(f'./new_config/images/test_HOMOG_LonDep_N{csize*3}_HIST.png',dpi=300)
plt.show()


########################
#%%

f, ax = plt.subplots(figsize=(8, 8))
ax.set_title(f"New stimulation synthetics", fontsize=15)
# ax.set_aspect("equal")


# sns.kdeplot(allats, alldeps, levels=30, fill=True)
aa,_,_,im=ax.hist2d(allats, alldeps, bins=np.int(np.sqrt(len(allons))), cmin=1,cmap='Blues', vmin=0, vmax=35)
ax.scatter(lat_UTM[:n_evs], depth_UTM[:n_evs], s=10, c='k',zorder=15)
ax.scatter(lat_UTM[reflist], depth_UTM[reflist], s=20, c='yellow',zorder=18)

# plt.scatter(huge_locmat[:,1,2], huge_locmat[:,2,2], s=10, c='orange', zorder=15, label='best estimate')
# ax.scatter(np.mean(allats.reshape([n_evs,csize*3]), axis=1)[reflist-1], np.mean(alldeps.reshape([n_evs,csize*3]), axis=1)[reflist-1], c='darkgreen', zorder=17)

# for i in range(n_evs):
#     plt.plot([ 
#         huge_locmat[:,1,2][i], lat_UTM[i]], 
#         [
#         huge_locmat[:,2,2][i], depth_UTM[i]], c='lightgray'

#     )
# ax.set_xlim(334000,352000)

# ax.plot([stafile[1][stats_a[1]], stafile[3][stats_a[-1]]])
ax.plot(w16A.NS, w16A.TVD*-1+1650, c='lightgrey', linewidth=4)

ax.plot([das_a[1], das_a[1]], [0, 1356], c='k', linewidth=4)
ax.plot([das_b[1], das_b[1]], [0, 2896], c='k', linewidth=4)
ax.plot([das_c[1], das_c[1]], [0, 2785], c='k', linewidth=4)
ax.grid()
ax.invert_yaxis()
ax.set_xlim(4.2629e6, 4.2635e6)
ax.set_ylim(3200,0)
plt.colorbar(im,ax=ax, shrink=.3, label='Location count')
ax.set_aspect(0.3)
ax.set_xlabel("Lat UTM [m]", fontsize=15)
ax.set_ylabel("Depth [m]", fontsize=15)
plt.savefig(f'./new_config/images/test_HOMOG_LatDep_N{csize*3}_HIST.png',dpi=300)
plt.show()

################
#%%

f, ax = plt.subplots(figsize=(10, 10))
ax.set_title(f"New stimulation synthetics", fontsize=15)
# ax.set_aspect("equal")

# sns.kdeplot(allons, allats, levels=30, fill=True)
h,_,_,im=ax.hist2d(allons, allats, bins=np.int(np.sqrt(len(allons))), cmin=1, cmap='Blues', vmin=0, vmax=79, range=([335200,336200], [4.26295e6, 4.26346e6]))
ax.scatter(lon_UTM[:n_evs], lat_UTM[:n_evs], s=10, c='k',zorder=15, label='Ground truth')
ax.scatter(lon_UTM[reflist], lat_UTM[reflist], s=20, c='yellow',zorder=18, label='Master events')

# plt.scatter(huge_locmat[:,0,2], huge_locmat[:,1,2], s=10, c='orange', zorder=15, label='best estimate')
# ax.scatter(np.mean(allons.reshape([n_evs,csize*3]), axis=1)[reflist-1], np.mean(allats.reshape([n_evs,csize*3]), axis=1)[reflist-1], c='darkgreen', zorder=17)


# for i in range(n_evs):
#     plt.plot([ 
#         huge_locmat[:,0,2][i], lon_UTM[i]], 
#         [huge_locmat[:,1,2][i], lat_UTM[i]], c='lightgray'

#     )

# ax.plot([stafile[1][stats_a[0]], stafile[2][stats_a[-1]]])
ax.plot(w16A.EW, w16A.NS, c='lightgrey', linewidth=4, label='Injection well')

ax.scatter(das_a[0], das_a[1], marker='s', s=150, c='k', label='DAS cable')
ax.scatter(das_b[0], das_b[1], marker='s', s=150, c='k')
ax.scatter(das_c[0], das_c[1], marker='s', s=150, c='k')
ax.grid()
ax.set_aspect(1)
ax.set_ylim(4.2629e6, 4.2635e6)
plt.colorbar(im,ax=ax, shrink=.3, label='Location count')
ax.set_xlabel("Lon UTM [m]", fontsize=15)
ax.set_ylabel("Lat UTM [m]", fontsize=15)
plt.legend(fontsize=15)
ax.set_xlim(335200,336200)
ax.set_ylim(4.26295e6, 4.26346e6)
plt.tight_layout()
plt.savefig(f'./new_config/images/test_HOMOG_LatLon_N{csize*3}_HIST.png',dpi=300)
plt.show()




# %%

# %%
