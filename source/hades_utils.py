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

def make_statfile():
    pass

def make_datfile():
    pass

