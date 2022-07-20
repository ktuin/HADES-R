#%%
from LatLongUTMconversion import LLtoUTM, UTMtoLL
from hades_input import hades_input
from hades_location import hades_location
import hades_utils as hu
import pandas as pd
import numpy as np

#%%

sta = pd.read_csv('./micro/data/FORGE_DAS_micro_new.txt', delimiter=' ', header=None)[0]
stafile = pd.read_csv('./micro/data/FORGE_DAS_micro_new.txt', delimiter=' ', header=None)

list_a =  list(sta)

csize = 15

np.random.seed(csize)
comb_a = [np.random.choice(list_a, csize), np.random.choice(list_a, csize)]
stats_a = []

for i in range(csize):
        stats_a.append(hu.find_element_in_list(comb_a[1][i], list(sta)))
        stats_a.append(hu.find_element_in_list(comb_a[0][i], list(sta)))



plt.figure(figsize=(7,15))
plt.title(f'WELLS AND RANDOM CONNECTIONS, N={csize}')

plt.scatter(stafile[2][stats_a], stafile[3][stats_a],zorder=15, label='78-32')
plt.gca().invert_yaxis()
plt.savefig(f'./micro/images/well_channels_micro_N{csize}.png', dpi=300)
plt.legend()

#%%

data_path='./'
input_file='./micro/data/FORGE_picks_micro_vLOWE.dat' 
sta_file='./micro/data/FORGE_DAS_micro_new.txt'

huge_locmat = np.zeros([32, 3, csize])
rect_correct = []

Vp=6100
Vs=Vp/1.9

delete_iteration = []


for i in range(csize):
    print(f"Loop {i}")
    # try:

    stations = [comb_a[1][i], comb_a[0][i]]


    testname = f'micro_{stations[0]}_{stations[1]}'
    out_file=f'./micro/outputs/FORGE_{testname}' 

    hobj=hades_input(data_path,input_file,sta_file, stations)
    hobj.relative_frame(Vp,Vs,stations,y_ref=-1,z_ref=-1,fixed_depth=-0.22)
    hobj.distance_calculation(Vp,Vs,stations, dist=None)
    hloc=hades_location(hobj,'./')
    hloc.location(out_file, mode='rel', plot=True)
    # hloc.location(out_file,mode='abs', plot=True)


    refprints = hloc.catalogue[:4, 0]
    reflist = []
    for k in range(len(refprints)):
        tmp = int(refprints[k].split('#R')[1])
        reflist.append(tmp+1)

    las = hu.sortout(np.array(np.float64(hloc.catalogue[:,1])), reflist)
    los = hu.sortout(np.array(np.float64(hloc.catalogue[:,2])), reflist)

    lat_calc, lon_calc = np.zeros(len(hloc.catalogue)), np.zeros(len(hloc.catalogue))
    for j in range(len(hloc.catalogue)):
        _, lon_calc[j], lat_calc[j] = LLtoUTM(23,  las[j], los[j])

    dep_calc = hu.sortout(np.array(np.float64(hloc.catalogue[:,3])), reflist)

    huge_locmat[:,0,i] = lon_calc.copy()
    huge_locmat[:,1,i] = lat_calc.copy()
    huge_locmat[:,2,i] = dep_calc.copy()*1000

    rect_correct.append(hloc.rect)


        
    # except: 

    #     huge_locmat[:,0,i] = np.zeros(32)
    #     huge_locmat[:,1,i] = np.zeros(32)
    #     huge_locmat[:,2,i] = np.zeros(32)

    #     delete_iteration.append(i)


    #     print(f'SVD did not converge at loop {i} with station combination {stations}')

if len(delete_iteration) > 0:
    for o in range(len(delete_iteration)):
        np.delete(huge_locmat, delete_iteration[o], 2)


allons = huge_locmat[:,0,:].flatten()
allats = huge_locmat[:,1,:].flatten()
alldeps = huge_locmat[:,2,:].flatten()

best_run = np.array(rect_correct).argmin()

allats = allats[allats.nonzero()]
allons = allons[allons.nonzero()]
alldeps = alldeps[alldeps.nonzero()]


#%%



np.savetxt('./micro/best_run_new2.csv', huge_locmat[:,:,best_run], delimiter=';')
np.savetxt('./micro/allons_new2.csv', allons, delimiter=';')
np.savetxt('./micro/allats_new2.csv', allats, delimiter=';')
np.savetxt('./micro/aldeps_new2.csv', alldeps, delimiter=';')
np.savetxt('./micro/master_evt2.csv', np.array([allons[5], allats[5], alldeps[5]]), delimiter=';')


