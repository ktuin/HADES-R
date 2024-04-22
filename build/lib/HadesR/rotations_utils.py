# HADES-R - A Python library for earthquake location analysis
# Author: Katinka Tuinstra & Francesco Grigoli
# Date: 22 Apr 2024
# File: rotation_utils.py
# Description: This file contains the implementation of the `rotation_utils` functions.

import numpy as np
import latlon2cart
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

"""A set of rotation functions that could be used in the Cluster class, as well as separately."""


def setup(
    optall,
    master_evts,
    cluster,
    dtps_obs,
    cluster_rot,
    stations,
    station_axis,
    vs,
    vp,
    baryp=[0, 0, 0],
):

    print("#------------------------#")
    print("Your choices:")
    print("#------------------------#")

    if type(cluster) == None.__class__:
        bary = np.mean(cluster_rot, axis=0)
        cluster = cluster_rot
    if type(baryp) != [0, 0, 0]:
        bary = baryp
    else:
        bary = np.mean(cluster, axis=0)  # get the barycenter of the cluster

    # ------------------------#
    # set up axis unit vectors and rotations
    v0, v1, v2 = define_axis_vectors(
        sta=stations[station_axis], bary=bary
    )  # axis unit vectors
    # ------------------------#
    print(f"Station {stations[station_axis]} chosen as axis station")

    if optall == False:  # create a subcluster of master events
        subcluster_rot = np.zeros([len(master_evts), 3])
        for i in range(len(master_evts)):
            u = master_evts[i]
            subcluster_rot[i, :] = cluster_rot[u, :]
            subcluster_org[i, :] = cluster[u, :]

        stavect = stations.copy()
        dtps_obs = np.zeros([len(stavect), len(cluster_rot)])

        for o in range(len(stavect)):
            _, _, ttp_org, tts_org = compute_traveltime(
                cluster=subcluster_org,
                stat=stavect[o],
                vp=vp,
                vs=vs,
                frame=False,
                noise=0.0,
            )
            dtps_obs[o, :] = np.array(
                [tts_org - ttp_org]
            )  # tt differences between s and p

        cluster_opt = subcluster_rot
        print(f"Optall = {optall}, use only master events for rotation finder")

    elif optall == True:
        stavect = stations
        dtps_obs = dtps_obs.copy()

        cluster_opt = cluster_rot
        print(f"Optall = {optall}, use all cluster events for rotation finder")

    return bary, v0, v1, v2, cluster_opt, dtps_obs, stavect


def compute_distance(x_c, x_r, frame):
    """Computes the 3D Euclidean distance between points
    x_c: location of the cluster
            if spherical coords, make sure to frame it as lat, lon, depth
    x_r: location of the receiver
    frame: Bool, for Cartesian set False, for Spherical (WGS84) set True"""

    if frame == True:  # if in latlon convert to a UTM domain for cartesian distance
        x_cC, x_rC = np.zeros(np.shape(x_c)), np.zeros(np.shape(x_r))
        for i in range(len(cluster)):
            orig_frame = latlon2cart.Coordinates(x_c[:, 1], x_c[:, 0], 0)

            x_cutm = orig_frame.geo2cart(x_c[:, 1], x_c[:, 0], 0)
            x_rutm = orig_frame.geo2cart(x_r[1], x_r[0], 0)

            x_cC[i, :] = np.array([x_cutm[i, 2], x_cutm[i, 1], x_c[i, 2]])
            x_rC[i, :] = np.array([x_rutm[2], x_rutm[1], x_r[2]])

    if frame == False:
        x_cC = x_c.copy()
        x_rC = x_r.copy()

    dist = np.zeros(len(x_c[:, 0]))
    for j in range(len(dist)):
        dist[j] = np.sqrt(
            (x_cC[j, 0] - x_rC[0]) ** 2
            + (x_cC[j, 1] - x_rC[1]) ** 2
            + (x_cC[j, 2] - x_rC[2]) ** 2
        )

    return dist


def demean_traveltimes(tt):
    """Removes the mean from the computed traveltimes. Serves to remove part of the error that
    is contained in the difference between observed wavefields (traveled through a complex medium)
    and the simple homogeneous model with straight euclidean distance in the trial data.
    tt: traveltime array"""

    tt_d = tt - np.mean(tt)
    tt_d = tt_d / np.abs(tt_d).max()

    return tt_d


def compute_traveltime(cluster, stat, vp, vs, frame, noise=0):
    """computes the traveltimes between cluster points and station coordinates
    cluster : your earthquake cluster (or any point cluster)
    station : the station you would like to compute the traveltimes for
    vp, vs : p- and s-wave velocities in [m/s]
    frame : Bool, set to either spherical (latlon) -- True, or False for UTM
    noise : ONLY USE FOR TESTING. Percentage of noise you'd like to add (default is zero)
    """

    dist = compute_distance(cluster, stat, frame)

    tts = dist / vs
    ttp = dist / vp

    if noise > 0:
        np.random.seed(2)
        noise = np.random.normal(loc=0, scale=noise, size=np.shape(dist))

        tts_n = tts + noise
        ttp_n = ttp + noise

    elif noise == 0:
        tts_n, ttp_n = tts.copy(), ttp.copy()

    ttp_demean = demean_traveltimes(ttp_n)
    tts_demean = demean_traveltimes(tts_n)

    return ttp_demean, tts_demean, ttp_n, tts_n


def distance_error(cl1, cl2):
    """compute the distance error if there is a 'correct' cluster present to compare with."""

    rms = np.sqrt(
        (  # compute the RMS
            ((cl1[:, 0] - cl2[:, 0]) ** 2)
            + ((cl1[:, 1] - cl2[:, 1]) ** 2)
            + ((cl1[:, 2] - cl2[:, 2]) ** 2)
        )
    )

    return rms


def test_cluster_rotation(cluster, thetas, deg, v0, v1, v2, bary):
    if deg == True:
        thetas = np.array(
            [np.radians(thetas[0]), np.radians(thetas[1]), np.radians(thetas[2])]
        )

    # Compute the quaternions
    r_1 = R.from_quat(
        [
            v0[0] * np.sin(thetas[0] / 2),
            v0[1] * np.sin(thetas[0] / 2),
            v0[2] * np.sin(thetas[0] / 2),
            np.cos(thetas[0] / 2),
        ]
    )
    r_2 = R.from_quat(
        [
            v1[0] * np.sin(thetas[1] / 2),
            v1[1] * np.sin(thetas[1] / 2),
            v1[2] * np.sin(thetas[1] / 2),
            np.cos(thetas[1] / 2),
        ]
    )
    r_3 = R.from_quat(
        [
            v2[0] * np.sin(thetas[2] / 2),
            v2[1] * np.sin(thetas[2] / 2),
            v2[2] * np.sin(thetas[2] / 2),
            np.cos(thetas[2] / 2),
        ]
    )

    # Apply the rotations (and re-place them at the barycenter coordinates)
    c4 = r_3.inv().apply(cluster - bary)
    c5 = r_2.inv().apply(c4)
    c6 = r_1.inv().apply(c5) + bary

    return c6


def define_axis_vectors(sta, bary):
    """Method that defines the three axes of rotation v0, v1, v2 that are comprised of:
    - the vertical axis from the barycentre upwards
    - the projected axis on the horizontal between a station and the barycentre
    - the axis normal to the plane spanned between the two previous axes"""

    vec0 = np.array([0, 0, 1])  # Z-axis vector
    vec1 = np.array([sta[0] - bary[0], sta[1] - bary[1], 0])  # sta1-barycenter vector
    vec2 = np.cross(vec0, vec1)
    # vec2 = np.array([sta2[0]-bary[0], sta2[1]-bary[1], sta2[2]])   # sta2-barycenter vector

    n0, n1, n2 = (
        np.linalg.norm(vec0),
        np.linalg.norm(vec1),
        np.linalg.norm(vec2),
    )  # norms of the vectors
    v0 = np.array(
        [vec0[0] / n0, vec0[1] / n0, vec0[2] / n0]
    )  # convert into unit vectors
    v1 = np.array([vec1[0] / n1, vec1[1] / n1, vec1[2] / n1])
    v2 = np.array([vec2[0] / n2, vec2[1] / n2, vec2[2] / n2])

    return v0, v1, v2


def plot_clusters_3D(cluster, sta1, sta2, bary):

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")

    # ax.set_title('Cluster of points with axes from the barycenter \n to the Z-axis and two stations as well as \n the projected axes')

    ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], label="Cluster points")
    ax.scatter(
        sta1[0], sta1[1], sta1[2], marker="v", color="red", s=50, label="Station 1"
    )
    ax.scatter(
        sta2[0], sta2[1], sta2[2], marker="v", color="darkred", s=50, label="Station 2"
    )

    ax.scatter(bary[0], bary[1], bary[2], color="cyan", label="barycenter")

    ax.plot(
        xs=[bary[0], sta1[0]],
        ys=[bary[1], sta1[1]],
        zs=[bary[2], sta1[2]],
        color="grey",
        label="Axis",
    )
    ax.plot(
        xs=[bary[0], sta2[0]],
        ys=[bary[1], sta2[1]],
        zs=[bary[2], sta2[2]],
        color="grey",
    )
    ax.plot(
        xs=[bary[0], bary[0]], ys=[bary[1], bary[1]], zs=[bary[2], 0.0], color="grey"
    )

    ax.plot(
        xs=[bary[0], sta1[0]],
        ys=[bary[1], sta1[1]],
        zs=bary[2],
        color="grey",
        linestyle="--",
        label="projected axis",
    )
    ax.plot(
        xs=[bary[0], sta2[0]],
        ys=[bary[1], sta2[1]],
        zs=bary[2],
        color="grey",
        linestyle="--",
    )

    ax.set_zlim(-5, 5.0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # plt.legend()

    # #plt.show()

    return fig, ax


def plot_cluster_misfit_3D(cluster, misfit, **kwargs):
    """To write still"""
    fig = plt.figure(figsize=(10, 12))

    ax = fig.add_subplot(projection="3d")
    # ax.set_title(f'Error w.r.t. original cluster')
    if "vmin" in kwargs or "vmax" in kwargs:
        im = ax.scatter(
            cluster[:, 0],
            cluster[:, 1],
            cluster[:, 2],
            c=misfit,
            s=75,
            cmap="bwr",
            label="cluster",
            vmin=kwargs["vmin"],
            vmax=kwargs["vmax"],
        )
    else:
        minmax = np.max([-misfit.min(), misfit.max()])
        im = ax.scatter(
            cluster[:, 0],
            cluster[:, 1],
            cluster[:, 2],
            c=misfit,
            s=75,
            cmap="bwr",
            label="cluster",
            vmin=-minmax,
            vmax=minmax,
        )

    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    plt.colorbar(im, label="Error", orientation="horizontal", shrink=0.7)

    return fig, ax


def plot_cluster_tt_3D(cluster, stations, sta, dtps, bary):
    """To write still"""

    fig = plt.figure(figsize=(10, 12))

    minmax = max([dtps.min(), dtps.max()])

    ax = fig.add_subplot(projection="3d")

    ax.set_title(f"Ts - Tp with respect to station {sta}")

    im = ax.scatter(
        cluster[:, 0],
        cluster[:, 1],
        cluster[:, 2],
        c=dtps[sta, :],
        s=75,
        cmap="plasma",
        label="cluster",
    )
    ax.scatter(
        stations[sta, 0],
        stations[sta, 1],
        stations[sta, 2],
        marker="v",
        color="red",
        s=200,
    )
    ax.scatter(
        stations[:, 0],
        stations[:, 1],
        stations[:, 2],
        marker="v",
        color="darkred",
        s=50,
    )
    ax.plot(
        xs=[bary[0], stations[sta, 0]],
        ys=[bary[1], stations[sta, 1]],
        zs=[bary[2], stations[sta, 2]],
        color="red",
    )
    # ax.plot(xs=[bary[0], stations[:,0]], ys=[bary[1],stations[:,1]], zs=[bary[2],stations[:,2]], color='grey')
    ax.scatter(bary[0], bary[1], bary[2], color="cyan", label="barycentre")

    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    ax.set_zlim(cluster[:, 2].min() - 0.2 * cluster[:, 2].min(), 1)
    plt.colorbar(im, orientation="horizontal", shrink=0.7)

    return fig, ax


def plot_compare_beforeafter(cluster, cluster_rot, cluster_comp):
    """To write still"""

    fig = plt.figure(figsize=(15, 20))
    ax1 = fig.add_subplot(121, projection="3d")
    plt.title("Before rotation", fontsize=15)
    ax1.scatter(
        cluster[:, 0],
        cluster[:, 1],
        cluster[:, 2],
        s=35,
        color="black",
        label="original cluster",
    )
    ax1.scatter(
        cluster_rot[:, 0],
        cluster_rot[:, 1],
        cluster_rot[:, 2],
        s=35,
        color="green",
        marker="p",
        label="rotated cluster",
    )
    for i in range(len(cluster_rot)):
        ax1.plot(
            [cluster[i, 0], cluster_rot[i, 0]],
            [cluster[i, 1], cluster_rot[i, 1]],
            [cluster[i, 2], cluster_rot[i, 2]],
            "cyan",
            alpha=0.5,
        )
    plt.legend()
    ax1.set_xlabel("x"), ax1.set_ylabel("y"), ax1.set_zlabel("z")

    ax2 = fig.add_subplot(122, projection="3d")
    plt.title("After rotation", fontsize=15)
    ax2.scatter(
        cluster[:, 0],
        cluster[:, 1],
        cluster[:, 2],
        s=35,
        color="black",
        label="original cluster",
    )
    ax2.scatter(
        cluster_comp[:, 0],
        cluster_comp[:, 1],
        cluster_comp[:, 2],
        s=35,
        color="green",
        marker="p",
        label="computed cluster",
    )
    for i in range(len(cluster_comp)):
        ax2.plot(
            [cluster[i, 0], cluster_comp[i, 0]],
            [cluster[i, 1], cluster_comp[i, 1]],
            [cluster[i, 2], cluster_comp[i, 2]],
            "cyan",
        )

    plt.legend()
    ax2.set_xlabel("x"), ax2.set_ylabel("y"), ax2.set_zlabel("z")

    return fig, ax1, ax2


def pca_theta_calculation(cluster_comp, evtsps, stations, plot):

    xobs, yobs, zobs = cluster_comp[:, 0], cluster_comp[:, 1], cluster_comp[:, 2]
    stations = stations

    rect = 1
    signs = []
    count = 1
    if plot == True:
        fig = plt.figure(figsize=(12, 4))

    for sta in range(len(stations[:, 0])):
        X = np.zeros([np.size(xobs), 2])
        dx = xobs - stations[sta, 0]
        dy = yobs - stations[sta, 1]
        dz = zobs - stations[sta, 2]
        tsp = np.array(evtsps[:, sta])
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        ir_dist = np.argsort(dist)
        X[:, 0] = dist[ir_dist]
        X[:, 1] = tsp[ir_dist]
        M = np.mean(X.T, axis=1)
        C = X - M
        V = np.cov(C.T)
        values, vectors = np.linalg.eigh(V)
        sign = np.sign(vectors[0, 1] * vectors[1, 1])
        signs.append(1 * sign)

        print(np.max(values), np.min(values))

        rect = 1 / (rect * (np.max(values) / np.min(values)))

        if plot == True:
            ax1 = plt.subplot(1, 4, count)
            plt.title(f"Station {count}")
            plt.xlabel("Distance [m]"), plt.ylabel("Ts-Tp [s]")
            ax1.scatter(dist[ir_dist], tsp[ir_dist])

        count += 1

    if signs[0] > 0 and signs[1] > 0:
        rect = rect
    else:
        rect = 1e10  # -1*rect
    if plot == True:
        fig.suptitle(f"Rectilinearity = {rect}")
        fig.tight_layout()
        # plt.show()

    print("----------------------------")
    print("Rectilinearity = ", rect)
    print("----------------------------")

    return rect


def apply_rotations_spatial(cluster, rotations):

    ca = a_quat(rotations[0]).apply(cluster - self.bary) + self.bary
    cb = b_quat(rotations[1]).apply(ca - self.bary) + self.bary
    cF = c_quat(rotations[2]).apply(cb - self.bary) + self.bary

    return cF
