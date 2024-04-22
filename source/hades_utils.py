### Function dump file
### Katinka Tuinstra
import numpy as np
from e13tools import raise_error


def sortout(orig, refarray):
    ins1 = np.insert(orig[4:], refarray[0] - 1, orig[0], 0)
    ins2 = np.insert(ins1, refarray[1] - 1, orig[1], 0)
    ins3 = np.insert(ins2, refarray[2] - 1, orig[2], 0)
    ins4 = np.insert(ins3, refarray[3] - 1, orig[3], 0)

    resorted = ins4.copy()
    return resorted


def distance_calculation(cluster, station):
    """
    Calculates the distances between cluster and station.

    Params
    ------
    cluster : `numpy.ndarray`
        Array with cluster coords of N events [N,3]
    station : `numpy.ndarray`
        Array with station coords [1,3]

    Returns
    -------
    distances : `numpy.ndarray`
        Array with distances of N events [1,N]
    """

    distances = np.sqrt(
        (cluster[:, 0] - station[0]) ** 2
        + (cluster[:, 1] - station[1]) ** 2
        + (cluster[:, 2] - station[2]) ** 2
    )

    return distances


def inter_iter_error(cluster1, cluster2):
    """
    Calculates the distance error between two consecutive point clouds
    cluster1 and cluster 2

    Params
    ------
    cluster1 : `numpy.ndarray`
        3D Array with cluster coords of N events [N,3]
    cluster2 : `numpy.ndarray`
        3D Array with cluster coords of N events [N,3]

    Returns
    -------
    distances : `numpy.ndarray`
        3D Array with distances of N events [N,3]
    stdev : `numpy.ndarray`
        1D Array with standard deviations of N events [N,1]
    """

    distances = np.sqrt(
        (cluster1[:, 0] - cluster2[:, 0]) ** 2
        + (cluster1[:, 1] - cluster2[:, 1]) ** 2
        + (cluster1[:, 2] - cluster2[:, 2]) ** 2
    )

    stdev = np.std(distances, axis=1)

    return stdev, distances


def find_element_in_list(element, list_element):
    """Finds element in list."""
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
    """Makes a HADES station input file.

    You can list all recording stations and only use a subset later.
    With structure:
    station_name lat lon depth[km]

    Parameters
    ----------
    path : `string`
        String with the path to deposit the station.txt file
    filename : `str`
        Filename of the station file
    station_names : `list`
        List of station names. If stations don't have specific names,
        make a logical naming system such as ['STA001', ... 'STA0015']
    station_coordinates : `numpy.ndarray`
        Numpy array [N, 3] of station [latitude, longitude, depth]
    """
    with open(path + filename + ".txt", "w") as wrst:
        for i in range(len(station_names)):
            wrst.write(
                f"{station_names[i]} {station_coordinates[i,0]} {station_coordinates[i,1]} {station_coordinates[i,2]}\n"
            )

    print(f"Saved station file {path}{filename}  - success!")


def make_datfile(
    path, filename, ref_idx, ref_coords, origin, station_names, p_times, s_times
):
    """
    Makes HADES event input file.

    You can list all recording stations and only use a subset later.
    The file structure:
    ORIGIN;lat;lon;depth;
    #(R)eventno;eventtime;lat;lon;depth;
    station_name;p_time;s_time;
    ...
    #(R)evtno;....

    Parameters
    ----------
    path : `string`
        String with the path to deposit the event.dat file
    filename : `string`
        File name of  your event input file
    ref_idx : `list` or `numpy.ndarray`
        List of array of indexes of your master events (minimum 4)
    ref_coords : `numpy.ndarray`
        Array [M, 3] of M master events with their lat, lon depth coordinates.
        If you only have one master event, use 0,0,0 as coordinates for the other
        events.
    origin : `numpy.ndarray` or `list`
        Origin coordinates of the cluster. Can be the literal barycentre or the
        coordinates of the first master event (default).
    station_names : `list`
        List of station names. Must be present in the station file as well.
    p_times : `list`
        List of p-pick timestrings in the format %Y/%d/%m %H:%M:%S.%f
    s_times : `list`
        List of s-pick timestrings in the format %Y/%d/%m %H:%M:%S.%f
    """

    if len(ref_idx) < 4:
        raise_error(
            "Minimum reference events are 4. Even if only 1 location is known, select\n more events to form coordinate system."
        )

    with open(path + filename + ".dat", "w") as wrdt:
        wrdt.write(f"ORIGIN;{origin[0]};{origin[1]};{origin[2]};\n")

        count = 0
        for i in range(len(p_times)):
            wrdt.write(f"#")
            if i in ref_idx:
                wrdt.write(f"R")
                coords = ref_coords[:, count]
                print(f"Reference event {i} added.")
                count += 1
            else:
                coords = np.array([0, 0, 0])

            wrdt.write(
                f"{str(i).zfill(2)};{p_times[i][0]};{coords[0]};{coords[1]};{coords[2]};\n"
            )

            for j in range(len(station_names)):
                # we only need the time part of the timestring
                tmp_pt = p_times[i][j].split(" ")[-1]
                tmp_st = s_times[i][j].split(" ")[-1]

                wrdt.write(f"{station_names[j]};{tmp_pt};{tmp_st};\n")

    print(f"Saved event file {path}{filename} - success!")


def flip_sign(cluster, x, y, z, b):
    # first demean the cluster
    cluster_dm = np.array(
        [
            cluster[:, 0] - cluster[b, 0],
            cluster[:, 1] - cluster[b, 1],
            cluster[:, 2] - cluster[b, 2],
        ]
    ).T

    # flip axes
    cluster_flip = np.array(
        [cluster_dm[:, 0] * x, cluster_dm[:, 1] * y, cluster_dm[:, 2] * z]
    ).T

    # now remean
    cluster_rm = np.array(
        [
            cluster_flip[:, 0] + cluster[b, 0],
            cluster_flip[:, 1] + cluster[b, 1],
            cluster_flip[:, 2] + cluster[b, 2],
        ]
    ).T

    return cluster_rm
