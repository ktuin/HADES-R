from e13tools import raise_error
import numpy as num
import datetime
import LatLongUTMconversion
import os
import sys
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R


km=1000.

class hades_location(object):
    """This class takes input objects from hades_input.py to locate events with
    respect to other events (relative location), after which an absolute location is 
    found using 1 (relative mode) or more (< 4, absolute) master events. 

    Parameters
    ----------
    input_obj : `hades.input.Struct`
        Input object from hades_input.py. 
    output_path : `string`
        Output filename or path. 

    References
    ----------
    See [Grigoli et al. 2021] for further details."""


    def __init__(self, input_obj, output_path):
        self.input=input_obj
        self.output_path=output_path


    def location(self, filename, mode, plot):
        """Decision module that dictates what hades_location will do according to
        the mode of operation ('rel' or 'abs). Relative relocation means relative
        to one master event, absolute means within the reference coordinate system
        of the minimum of four master events. 

        Parameters
        ----------
        filename : `string`
            Output file filename or path.
        mode : `string`
            Absolute ('abs') or relative mode ('rel').
        plot : `bool`
            Plot the rectilinearity and location results. 
        
        Returns
        -------
        None
        """

        mode_opts = ['abs', 'rel']
        if mode not in mode_opts:
            raise_error(err_msg='Mode incorrect, options are abs and rel')

        distances=(self.input).distances
        if mode=='rel':
            references=(self.input).rel_references
        elif mode=='abs':
            references=(self.input).references
        nref,_=num.shape(references)
        nevs,_=num.shape(distances)
        for i_ev in range(nref,nevs):
            sys.stdout.write(' Locating events %3d %% \r' %((i_ev/nevs)*100))
            references=hades_location.__dgslocator(i_ev, references, distances)
            sys.stdout.flush()
        self.locations=references
        if mode=='rel':
            self.__absolute_cluster_location(filename, plot)
            references1=(self.input).references
            references2=(self.input).rel_references
            print(references1,references2)
        elif mode=='abs':
            rect = self.__pca_theta_calculation(xobs=self.locations[:,0],
                yobs=self.locations[:,1], zobs=self.locations[:,2], 
                evtsps=self.__initialize_tsp_db((self.input).stations), 
                stations=(self.input).stations, plot=True)
            self.rect = rect
            self.__catalogue_creation(filename)
            self.__plot_results(filename)
        sys.stdout.write('\n')


    def __dgslocator(event, references, distances):
        '''Relative event locator using inter-event distances with respect to
        the reference events. Locates all events in a relative fram with respect
        to to reference or master events. 
              
        Parameters
        ----------
        event : string 
            Is the id of the event you want locate (# number of event)
        references : numpy.ndarray
            An object array of the form ['eventid',x,y,z] containing the master events
        
        Returns
        -------
        references : numpy.ndarray
            Returns an array-like instance with all relative locations of the events.'''

        n_ref, _ =num.shape(references)

        X=num.array([references[:,0],references[:,1],references[:,2]]).T
        XC=num.mean(X,axis=0)
        D=num.zeros([n_ref,n_ref])
        for i in range(n_ref):
            for j in range(n_ref):
                Xi=references[i,0]; Xj=references[j,0]; #need to be optimized
                Yi=references[i,1]; Yj=references[j,1];
                Zi=references[i,2]; Zj=references[j,2];
                dio=distances[i,event]; djo=distances[j,event];
                dij=num.sqrt((Xi-Xj)**2+(Yi-Yj)**2+(Zi-Zj)**2)
                # dij=distances[i,j]
                D[i,j]=(dio**2+djo**2-dij**2)/2.
                D[j,i]=D[i,j]

        U,S, _=num.linalg.svd(D, full_matrices=True)
        S=num.diag(S)
        Y=num.dot(U[:,0:3],num.sqrt(S[0:3,0:3]))
        XC=num.mean(X,axis=0)
        YC=num.mean(Y,axis=0)

        XTY=num.dot((X-XC).T,(Y-YC))
        U1,_,V1=num.linalg.svd(XTY, full_matrices=True)
        Q=num.dot(U1,V1)
        Y=num.dot(Q,(Y-YC).T)
        Xfin=num.dot(-Q,YC)+XC
        evloc=num.array([Xfin[0], Xfin[1], Xfin[2]])
        references=num.vstack((references,evloc))
        return references


    def __absolute_cluster_location(self,filename, plot, acc=5):
        """Finds the absolute location of events with respect to a single master event, via rotation
        optimisation. The optimisation runs by maximising the rectilinearity by minimising the (normalised) 
        error. Uses brute-force grid search of 350 degrees rotation around the Z, A, B axes, with a 5 degree
        standard accuracy. The barycentre is the master event, which is the origin of the rotations.
        
        Parameters
        ----------
        filename : `string`
            Filename or path of the output file. 
        plot : `bool`
            Plots the distance-(ts-tp) and rectilinearity.
        acc : `int`, optional
            Accuracy in degrees for the brute force angle search. Default is 5.  
        
        Returns
        -------
        None  
        """

        ### Rotation functions to optimize #########################################

        def define_axis_vectors(sta, bary):
            """Method that defines the three axes of rotation v0, v1, v2 that are comprised of:
                - the vertical axis from the barycentre upwards (Z)
                - the projected axis on the horizontal between a station and the barycentre (A)
                - the axis normal to the plane spanned between the two previous axes (B)
                
                Parameters
                ----------
                sta : `numpy.ndarray`, [1, 3]
                    Station coordinates of the station from which the rotation coordinate system is
                    constituted. There is always a Z-axis, the sta forms the A axis and the 
                    orthogonal axis to that is the B axis. 
                bary : `numpy.ndarray`, [1, 3]
                    Barycentre of the cluster, origin of the rotation axes. This does not neccessarily
                    need to be the centre of the cluster. It can also be the first master event.

                Returns
                -------
                None.
                    """

            vec0 = num.array([0,0, 1])                                      # Z-axis vector
            vec1 = num.array([sta[0]-bary[0], sta[1]-bary[1], 0])         # sta1-barycenter vector
            vec2 = num.cross(vec0, vec1)

            n0, n1, n2 = num.linalg.norm(vec0), num.linalg.norm(vec1), num.linalg.norm(vec2)   # norms of the vectors
            v0 = num.array([vec0[0]/n0, vec0[1]/n0, vec0[2]/n0])            # convert into unit vectors
            v1 = num.array([vec1[0]/n1, vec1[1]/n1, vec1[2]/n1])
            v2 = num.array([vec2[0]/n2, vec2[1]/n2, vec2[2]/n2])

            self.v0 = v0
            self.v1 = v1
            self.v2 = v2

            return v0, v1, v2

        def z_quat(theta):
            """Takes the first angle theta and computes the quaternion around axis v0."""

            v0 = self.v0
            
            r_a = R.from_quat([v0[0] * num.sin(theta/2),     # compute quaternions
                                        v0[1] * num.sin(theta/2), 
                                        v0[2] * num.sin(theta/2), 
                                        num.cos(theta/2) ])

            return r_a


        def a_quat(theta):
            """Takes the second angle theta and computes the quaternion around axis v0."""

            v1 = self.v1

            r_b = R.from_quat([v1[0] * num.sin(theta/2),     # compute quaternions
                                        v1[1] * num.sin(theta/2), 
                                        v1[2] * num.sin(theta/2), 
                                        num.cos(theta/2) ])

            return r_b


        def b_quat(theta):

            v2 = self.v2

            r_c = R.from_quat([v2[0] * num.sin(theta/2),     # compute quaternions
                                        v2[1] * num.sin(theta/2), 
                                        v2[2] * num.sin(theta/2), 
                                        num.cos(theta/2) ])

            return r_c


        def apply_rotations_spatial(cluster, rotations, station):
            """Applies a rotation to a cluster of N points, around a pre-defined barycentre.
            The barycentre can be the actual barycentre of the cloud, or a master event.
            It applies three rotations (a, b, Final (F)) around three axes using the quat
            modules. 
            
            Parameters
            ----------
            cluster: array [N x 3]
                The 3D array of points
            rotations: list [1, 3] 
                List of rotations around 3 axes Z, A, B
            station: array [1, 3]
                Array with station location XYZ
            
            Returns
            -------
                array [N x 3]
                    Array with the new rotated cluster
            """

            bary = cluster[0,:]
            # print('Barycenter =', cluster[0,:])
            # bary = num.array([num.mean(cluster[:,0]), num.mean(cluster[:,1]), num.mean(cluster[:,2])])
            v0, v1, v2 = define_axis_vectors(station, bary)

            ca = z_quat(num.radians(rotations[0])).apply(cluster - bary) + bary
            cb = a_quat(num.radians(rotations[1])).apply(ca - bary) + bary
            cF = b_quat(num.radians(rotations[2])).apply(cb - bary) + bary

            return cF



        #currently only search along strike is implemented

        Vp=(self.input).vp
        Vs=(self.input).vs
        kv=(Vp*Vs)/(Vp-Vs)
        stations=(self.input).stations
        # print("Using stations", stations)
        depth=(self.input).origin[-1]
        thetas=num.arange(0,41)*0.025*num.pi*2
        evtsps=self.__initialize_tsp_db(stations)
        rms_min=1E10
        zrot=self.locations[:,2]+depth
        for ysign in [-1,1]:
            for theta in thetas:
                crot=(self.locations[:,0]+1j*ysign*self.locations[:,1])*num.exp(-1j*theta)
                rms=self.__rms_theta_calculation(crot.real,crot.imag,zrot,evtsps,kv,stations)
                if rms < rms_min:
                    rms_min=rms
                    theta_best=theta
                    ysign_best=ysign
        crot=(self.locations[:,0]+1j*ysign_best*self.locations[:,1])*num.exp(-1j*theta_best)
        self.locations[:,0]=crot.real
        self.locations[:,1]=crot.imag
        self.locations[:,2]=zrot
        pca_max_z, pca_max_a, pca_max_b = 1,1,1
        sts = list(stations.keys())
        self.evtsps = evtsps

        thetas=num.arange(0,360, acc)
        theta_best_z, theta_best_a, theta_best_b = 0,0,0
        pca_max_a, pca_max_b, pca_max_z = 1,1,1


        print('------------------ trying Z rotation ---------------')
        for theta in thetas:
            qrot_z = apply_rotations_spatial(self.locations, 
                    rotations=[theta, 0 , 0], station=(self.input).stations[sts[0]])
            pca=self.__pca_theta_calculation(qrot_z[:,0], qrot_z[:,1], qrot_z[:,2], evtsps,stations, plot=False)
            if pca < pca_max_z:
                print('Update Z, theta', theta)
                pca_max_z=pca
                theta_best_z=theta
            else:
                theta_best_z=theta_best_z

        print('------------------ trying A rotation ---------------')
        for theta in thetas:
            qrot_a = apply_rotations_spatial(self.locations, 
                    rotations=[theta_best_z, theta , 0], station=(self.input).stations[sts[0]])
            pca=self.__pca_theta_calculation(qrot_a[:,0], qrot_a[:,1], qrot_a[:,2], evtsps,stations, plot=False)

            if pca < pca_max_a:
                print('Update A, theta', theta)
                pca_max_a=pca
                theta_best_a=theta
            else: 
                theta_best_a=theta_best_a

        print('------------------ trying B rotation ---------------')
        for theta in thetas:
            qrot_b = apply_rotations_spatial(self.locations, 
                    rotations=[theta_best_z, theta_best_a , theta], station=(self.input).stations[sts[0]])
            pca=self.__pca_theta_calculation(qrot_b[:,0], qrot_b[:,1], qrot_b[:,2], evtsps,stations, plot=False)

            if pca < pca_max_b:
                print('Update B, theta', theta)
                pca_max_b=pca
                theta_best_b=theta
                self.pca = pca
            else:
                theta_best_b=theta_best_b


        qrot_best = apply_rotations_spatial(self.locations, 
                    rotations=[theta_best_z, theta_best_a , theta_best_b], station=(self.input).stations[sts[0]])
        self.__pca_theta_calculation(qrot_best[:,0], qrot_best[:,1], qrot_best[:,2], evtsps,stations, plot=plot)
        
        print('theta best', [theta_best_z, theta_best_a, theta_best_b])
        theta_best=(360*num.pi)/180

        self.locations[:,0]=qrot_best[:,0]#crot.real
        self.locations[:,1]=qrot_best[:,1]#crot.imag
        self.locations[:,2]=qrot_best[:,2]#zrot
        self.__catalogue_creation(filename)

        if plot == True:
            self.__plot_results(filename)


    def __rms_theta_calculation(self,xobs,yobs,zobs,evtsps,kv,stations):
        """Calculates the best initial lat-lon orientation of the cluster based on the RMS
        between the observed and calculated traveltimes from a homogeneous model.
        
        Parameters
        ----------
        xobs : `numpy.ndarray`
            x-locations of current relative cluster
        yobs : `numpy.ndarray`
            y-locations of current relative cluster
        zobs : `numpy.ndarray`
            z-locations of the current relative clsuter
        evtsps : `numpy.ndarray`
            Array containing all (ts-tp) per event per station. 
        kv : `float`
            Velocity ratio 
        stations : `input.stations`
            Station object with the station names and coordinates.

        Returns
        -------
        rms : `float`
            The calculated RMS error between observed and calculated traveltimes
        """

        rms=0
        for sta in stations.keys():
            dx=(xobs-(self.input).stations[sta][0])
            dy=(yobs-(self.input).stations[sta][1])
            dz=(zobs-(self.input).stations[sta][2])
            tsp_obs=num.array(evtsps[sta])
            tsp_obs=tsp_obs-num.mean(tsp_obs)
            tsp_calc=num.sqrt(dx**2+dy**2+dz**2)/kv
            tsp_calc=tsp_calc-num.mean(tsp_calc)
            rms+=num.sqrt(num.sum((tsp_calc-tsp_obs)**2)/num.size(tsp_obs))
        rms=rms/len(stations.keys())
        return rms


    def __pca_theta_calculation(self,xobs,yobs,zobs,evtsps,stations,plot):
        """Calculates the rectilinearity between calculated distances (event-station)
        with respect to observed traveltime difference (ts-tp) per event. 
        
        Ideal rectilinearity is infite, but this scaled so that the maximum 
        rectilinearity occurs at 0. This method is independent of re-calculated
        traveltimes and therefore velocity model independent.
        
        Parameters
        ----------
        xobs : `numpy.ndarray`
            x-locations of current relative cluster
        yobs : `numpy.ndarray`
            y-locations of current relative cluster
        zobs : `numpy.ndarray`
            z-locations of the current relative clsuter
        evtsps : `numpy.ndarray`
            Array containing all (ts-tp) per event per station. 
        stations : `input.stations`
            Station object with the station names and coordinates.
        plot : `bool`
            Plots the rectilinearity. 

        Returns
        -------
        rect : `float`
            Rectilinearity (ideal = 0).
            """


        rect=1
        if plot == True:
            plot_data=[]
        for sta in stations.keys():
            X=num.zeros([num.size(xobs),2])
            dx=(xobs-(self.input).stations[sta][0])
            dy=(yobs-(self.input).stations[sta][1])
            dz=(zobs-(self.input).stations[sta][2])
            tsp=num.array(evtsps[sta])
            #tsp=tsp_obs-num.mean(tsp_obs)
            dist=num.sqrt(dx**2+dy**2+dz**2)
            ir_dist=num.argsort(dist)
            X[:,0]=dist[ir_dist]
            X[:,1]=tsp[ir_dist]
            M=num.mean(X.T, axis=1)
            C=X-M
            V=num.cov(C.T)
            values, vectors = num.linalg.eig(V)
            values=values/num.max(values)
            vector=vectors[:,num.argmax(values)]
            if num.sign(vector[0])>0 and num.sign(vector[1])>0:
                rect= ( 1 / rect*(num.max(values)/num.min(values)) ) 
            else:
                rect = 1
            if plot == True:
                plot_data.append([dist[ir_dist],tsp[ir_dist]])
        if plot == True:
            plt.figure()
            plt.title(' rect value : '+str(rect))
            plt.xlabel('distance')
            plt.ylabel('ts-tp [s]')
            plt.plot(plot_data[0][0],plot_data[0][1],'or', label='sta1')
            plt.plot(plot_data[1][0],plot_data[1][1],'ob', label='sta2')
            plt.legend()
            plt.show()
        
        self.rect = rect

        return rect


    def __initialize_tsp_db(self,stations):
        """Starts a database of all (ts-tp) at each station (`input.station`)."""
        evtsps={}
        for sta in stations.keys():
            evtsps[sta]=[]
            for event in (self.input).events:
                evtsps[sta].append((self.input).data[event][sta][-1])
        return evtsps

    
    def __catalogue_creation(self, filename):
        """Makes a .txt file (`string`) with the catalogue containing all (ts-tp) and event locations.
        !!! look out, the order is [master events : rest of the events], so an argsort
        needs to be done to regain the original order of events for comparison."""
        
        nev,_=num.shape(self.locations)
        evids=(self.input).events
        print('Location process completed, number of located events: %d '%(nev))
        catalogue=[]
        with open(filename+'.txt','w') as f:
            f.write('Id;evtno;Lat;Lon;Depth;sta1;tstp1;tP1;sta2;tstp2;tP2;\n')
            for i in range(nev):
                lat,lon=LatLongUTMconversion.UTMtoLL(23, self.locations[i,1]+(self.input).origin[1], self.locations[i,0]+(self.input).origin[0],(self.input).origin[2])
                depth=(self.locations[i,2])/1000
                event=evids[i]
                evtno=int(evids[i].split('#')[-1].split('R')[-1])
                t_string=''
                for sta in (self.input).sel_sta:
                    if sta in (self.input).data[event].keys():
                        tsp=(self.input).data[event][sta][-1]
                        tid=(self.input).data[event][sta][0]
                        t_string=t_string+f'{sta};{tsp};{str(tid)};'
                f.write(f'{event};{evtno};{lat};{lon};{depth};{t_string}\n')
                # f.write(event+' '+'%6.4f '%(lat)+'%6.4f '%(lon)+'%6.4f'%(depth)+t_string+'\n')
                catalogue.append([event,lat,lon,depth])
        self.catalogue=num.array(catalogue)


    def __plot_results(self, filename):
        """Plots the results out of the catalogue from path with filename (`string`)."""

        # Custom colours
        c1='#4285F4'
        c2='#EA4335'
        c4='#34A853'
        c3='#FBBC05'
        nref=len((self.input).refevid)
        plt.figure(figsize=(10.0,15.0))
        ax1=plt.subplot(111)
        ax1.scatter(self.locations[nref:,0],self.locations[nref:,1], s=50, c=c1)
        ax1.scatter(self.locations[0:nref,0],self.locations[0:nref:,1],s=50, c=c3)
        for sta in (self.input).stations.keys():
            station=(self.input).stations[sta]
            ax1.scatter(station[0], station[1], c=c2, marker='v', s=200, zorder=3, linewidth=0.5)
        ax1.grid('on')
        ax1.set_aspect('equal')
        fout=os.path.join(self.output_path,filename)
        plt.savefig(fout+'.eps')
        plt.show()
