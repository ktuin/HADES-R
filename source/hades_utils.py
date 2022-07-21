### Function dump file
### Katinka Tuinstra 

import numpy as np

def sortout(orig, refarray):

    ins1 = np.insert(orig[4:], refarray[0]-1, orig[0], 0)
    ins2 = np.insert(ins1,     refarray[1]-1, orig[1], 0)
    ins3 = np.insert(ins2,     refarray[2]-1, orig[2], 0)
    ins4 = np.insert(ins3,     refarray[3]-1, orig[3], 0)

    resorted = ins4.copy()
    return resorted

def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None

def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None


def plot_tstp():
    pass

def plot_multirun():
    pass

def make_statfile(path, filename, station_names, station_coordinates):
    """ Makes a HADES station input file. With structure:
    station_name lat lon depth[km]

    Parameters
    ----------
    path : `str`
        String with the path to deposit the station.txt file
    filename : `str`
        Filename of the station file
    station_names : `list`
        List of station names. If stations don't have specific names, 
        make a logical naming system such as ['STA001', ... 'STA0015']
    station_coordinates : `numpy.ndarray`
        Numpy array [N, 3] of station [latitude, longitude, depth]
    """
    with open(path+filename+'.txt', 'w') as wrst:
        for i in range(len(station_names)):
            wrst.write(f'{station_names[i]} {station_coordinates[i,0]} {station_coordinates[i,1]} {station_coordinates[i,2]}\n')

def make_datfile(path, filename, references, station_names, p_times, s_times):
    """Work in progress"""
    pass

    # with open(path+filename, 'w') as wrdt:

        

