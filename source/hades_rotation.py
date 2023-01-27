import numpy as np
from math import radians, degrees

from scipy.spatial.transform import Rotation as R
from scipy.optimize import fmin_powell, fmin_cg, fmin_bfgs, fmin_l_bfgs_b, basinhopping,brute, dual_annealing
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

from rotations_utils import compute_traveltime, define_axis_vectors, demean_traveltimes

class Cluster:
    """ This class can do 3D rotation optimization on a cluster of points (or earthquakes).
    What's different from rotations_class? That is that here I try to include a list of stations
    instead of just two."""


    def __init__(self, cluster, axis, stations, bary, subcluster, vs, vp, dtps_org, cluster_true):

        self.cluster = cluster
        self.dtps_org = dtps_org
        self.stations = stations
        self.bary = bary
        self.subcluster = subcluster
        self.vs = vs
        self.vp = vp

        self.stachoice = self.stations[axis,:]
        self.rectvect = []
        self.tried_rots = []
        self.rmsvect = []
        self.costvect = []
        self.cluster_true = cluster_true
        
        

        ### WRITE AN INIT 


    # --------------------- METHODS ----------------------- #


    ### Rotation functions to optimize #########################################
    def a_quat(self, theta):
        """ Quaternions for rotation of `theta` radians around ax0 axis

        Parameters
        ----------
        theta : scalar
            Rotation angle in radians

        Returns
        -------
        r_a: quaternion for the rotation
        """
        v0, _, _ = define_axis_vectors(self.stachoice, self.bary)
        
        r_a = R.from_quat([v0[0] * np.sin(theta/2),     # compute quaternions
                                    v0[1] * np.sin(theta/2), 
                                    v0[2] * np.sin(theta/2), 
                                    np.cos(theta/2) ])

        return r_a


    def b_quat(self, theta):
        """ Quaternions for rotation of `theta` radians around ax1 axis

        Parameters
        ----------
        theta : scalar
            Rotation angle in radians

        Returns
        -------
        r_a: quaternion for the rotation
        """

        _, v1, _ = define_axis_vectors(self.stachoice, self.bary)  

        r_b = R.from_quat([v1[0] * np.sin(theta/2),     # compute quaternions
                                    v1[1] * np.sin(theta/2), 
                                    v1[2] * np.sin(theta/2), 
                                    np.cos(theta/2) ])

        return r_b


    def c_quat(self, theta):
        """ Quaternions for rotation of `theta` radians around ax2 axis
        Parameters
        ----------
        theta : scalar
            Rotation angle in radians

        Returns
        -------
        r_a: quaternion for the rotation
        """
        _, _, v2 = define_axis_vectors(self.stachoice, self.bary)

        r_c = R.from_quat([v2[0] * np.sin(theta/2),     # compute quaternions
                                    v2[1] * np.sin(theta/2), 
                                    v2[2] * np.sin(theta/2), 
                                    np.cos(theta/2) ])

        return r_c

    ######################################################################## 


    ###### Mismatch functions ######
    def correl_mismatch(self, dtps_org, dtps_rot):  # seems to work better
        """ Negative correlation between the two clusters, flattened to 1D
        """
            
        correl = np.corrcoef(dtps_rot.ravel(), dtps_org.copy().ravel())[0, 1]
        print("Correll", -correl)
        return -correl

    def rms_mismatch(self, dtps_org, dtps_rot):
        """ RMS mismatch value between the two clusters in traveltime"""

        # plt.figure()
        # plt.plot(dtps_rot[0,:], label='rot')
        # plt.plot(dtps_org[0,:], label='org')
        # plt.legend()
        # plt.show()
        
        dtps_dm = demean_traveltimes(dtps_org.copy())
            
        rms = np.sqrt((                                      # compute the RMS
                        np.sum((dtps_rot[0,:]- dtps_dm[0,:])**2)  +
                        np.sum((dtps_rot[1,:]- dtps_dm[1,:])**2)  +
                        np.sum((dtps_rot[2,:]- dtps_dm[2,:])**2))) 

        #print("RMS", rms)
        self.rms = rms
        return rms
    
    def rms_mismatch_abs(self, cluster_rot):
        """ RMS mismatch value between the two clusters in absolute sense"""

        # plt.figure()
        # plt.plot(dtps_rot[0,:], label='rot')
        # plt.plot(dtps_org[0,:], label='org')
        # plt.legend()
        # plt.show()
            
        rms = np.sqrt((                                      # compute the RMS
                        np.sum((self.cluster_true[:,0]- cluster_rot[:,0])**2)  +
                        np.sum((self.cluster_true[:,1]- cluster_rot[:,1])**2) +
                        np.sum((self.cluster_true[:,2]- cluster_rot[:,2])**2))) 

#         print("RMS", rms)
        #self.rms = rms
#         self.rmsvect.append(rms)
        return rms
    
    
    def l1_mismatch(self, dtps_org, dtps_rot):
        """L1 norm mismatch between the two clusters in traveltime"""

        l1_norm = np.sqrt((                                      # compute the L1 norm
               np.abs( np.sum(np.abs((dtps_rot[0,:]- dtps_org.copy()[0,:]))  +
                np.sum((dtps_rot[1,:]- dtps_org.copy()[1,:])) ))))
        
        # print("L1", l1_norm)
        return l1_norm

    def pca_theta_calculation(self, evtsps, c_try, plot=False):

        xobs, yobs, zobs = c_try[:,0], c_try[:,1], c_try[:,2]
        stations = self.stations

        rect=1
        signs=[]
        rectall=[]
        count = 1 
        if plot==True:
            fig=plt.figure(figsize=(12,4))

        for sta in range(len(stations[:,0])):
            X=np.zeros([np.size(xobs),2])
            dx=(xobs-stations[sta,0])
            dy=(yobs-stations[sta,1])
            dz=(zobs-stations[sta,2])
            tsp=np.array(evtsps[:,sta])
            dist=np.sqrt(dx**2+dy**2+dz**2)
            ir_dist=np.argsort(dist)
            X[:,0]=dist[ir_dist]
            X[:,1]=tsp[ir_dist]
            M=np.mean(X.T, axis=1)
            C=X-M
            V=np.cov(C.T)
            values, vectors = np.linalg.eigh(V)
            sign=np.sign(vectors[0,1]*vectors[1,1])
            signs.append(1*sign)

            rect= 1e18/(rect*(np.max(values)/(np.min(values)+1e-30)))


            if plot==True:          
                ax1 = plt.subplot(1,4,count)
                plt.title(f'Station {count}')
                plt.xlabel('Distance [m]'), plt.ylabel('Ts-Tp [s]')
                ax1.scatter(dist[ir_dist], tsp[ir_dist], c='k')
            
            count += 1

            if signs[sta]>=0:
                rect=rect
            else:
                rect=1e24#-1*rect

            rectall.append(rect)

        if plot==True:
            fig.suptitle(f'Rectilinearity = {np.mean(np.array(rectall))}')
            fig.tight_layout()
            plt.show()

        return np.mean(np.array(rectall))


    ######################### optimization part #############################                                                              
    def apply_rotations(self, cluster, rotations):
        """ Applies rotations (quaternion-wise) to the cluster
        cluster: your cluster of points
        rotations: given in quaternions around the 3 axes"""

        ca = self.c_quat(np.radians(rotations[2])).apply(cluster - self.bary) 
        cb = self.b_quat(np.radians(rotations[1])).apply(ca ) 
        cF = self.a_quat(np.radians(rotations[0])).apply(cb ) + self.bary

        self.ca = ca
        self.cb = cb
        self.cF = cF

        temp = np.zeros([len(self.stations), len(cF)])

        for i in range(len(self.stations)):
            _, _, ttp_cF, tts_cF = compute_traveltime(cluster=cF, stat=self.stations[i,:], 
                                        vp=self.vp, vs=self.vs, frame=False, noise=0.0)

            temp[i, :] = demean_traveltimes( tts_cF - ttp_cF )

        self.dtps_rot = temp

        return self.dtps_rot

    def apply_rotations_spatial(self, cluster, rotations):
    
        ca = self.a_quat(np.radians(rotations[0])).apply(cluster - self.bary) + self.bary
        cb = self.b_quat(np.radians(rotations[1])).apply(ca - self.bary) + self.bary
        cF = self.c_quat(np.radians(rotations[2])).apply(cb - self.bary) + self.bary

        self.cF = cF

        return self.cF
    
    def invert_rotations_spatial(self, cluster, rotations):
        

        try:

            ca = self.c_quat(np.radians(rotations[2])).inv().apply(cluster - self.bary) 
            cb = self.b_quat(np.radians(rotations[1])).inv().apply(ca ) 
            cF = self.a_quat(np.radians(rotations[0])).inv().apply(cb ) + self.bary

            self.cF = cF
            
        except:
            print('rotations', rotations)
            
            raise AttributeError
        
        return self.cF



    def cost_function(self, rotations):
        """ Computes cost function of the rotation problem. Options are different 
        mismatch metrics, e.g., rms_mismach, l1_norm or correl_mismatch."""
        
        self.tried_rots.append(rotations)

        # Select for other mismath criteria (fill in return)
        Y_t = self.apply_rotations(self.subcluster,  rotations)
        # select for PCA mismatch (fill in return)
        O_t = self.invert_rotations_spatial(self.subcluster, rotations)
        self.O_t = O_t 
        # self.rms = self.rms_mismatch_abs(self.O_t)        
        #rect = self.pca_theta_calculation(self.dtps_org.T, self.O_t)
#         self.costvect.append(self.rms_mismatch(self.dtps_org, Y_t))
        self.costvect.append(self.pca_theta_calculation(self.dtps_org.copy().T, self.O_t))
        # self.rmsvect.append(self.rms)
        return np.log10(self.pca_theta_calculation(self.dtps_org.T, self.O_t) )
#         return self.rms_mismatch(self.dtps_org, Y_t)
        # return self.rms_mismatch_abs(self.O_t)



 
    def optimize_rotations(self, prior, method, optall):
        """the optimizer using different scipy optimizers. 
        prior: an initial guess of the degree amount (standard: [0, 0, 0])
        method: choose between powell, cg (conjugate gradient), bfgs (mostly the most efficient)
        optall: Bool, use all events or only master events"""

        self.optall = optall
        #bounds = [(0, 2*np.pi), (0, 2*np.pi), (0,2*np.pi)]
        bounds = [(0, 359), (0, 359), (0, 359)]
        
        if method == 'powell':
            best_params = fmin_powell(self.cost_function, prior)
            thetas = degrees(best_params[0]), degrees(best_params[1]), degrees(best_params[2])
            
        elif method == 'brute':
            best_params_brute = brute(func=self.cost_function, ranges=bounds, Ns=72, workers=4)
            print(best_params_brute)
            thetas = np.array([degrees(best_params_brute[0]), degrees(best_params_brute[1]), degrees(best_params_brute[2])])
            

        elif method == 'cg':
            best_params_cg = fmin_cg(self.cost_function, prior, gtol=0,full_output=True)
            thetas = np.array([degrees(best_params_cg[0][0]), degrees(best_params_cg[0][1]), degrees(best_params_cg[0][2])])
            print(best_params_cg)
        elif method == 'bfgs' or method == None:

            best_params_bfgs = fmin_bfgs(self.cost_function, prior, norm=2, gtol=1e-50, epsilon=1)
            thetas = np.array([degrees(best_params_bfgs[0]), degrees(best_params_bfgs[1]), degrees(best_params_bfgs[2])])


        elif method == 'l-bfgs':
            best_params_lbfgs = fmin_l_bfgs_b(self.cost_function, prior, bounds=bounds, approx_grad=True)
            print(best_params_lbfgs)
            thetas = np.array([degrees(best_params_lbfgs[0][0]), degrees(best_params_lbfgs[0][1]), degrees(best_params_lbfgs[0][2])])

            
        elif method == 'basinhopping':
            best_params_bh = basinhopping(self.cost_function, prior, niter=350, T=120, 
                                        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds,
                                                            "options":{"ftol":1e-16, "maxiter":10}
                                                           }, 
                                        target_accept_rate=0.5,
                                        stepsize=359,
                                        stepwise_factor=0.4,  
                                        )
            
                        
            
            thetas = np.array([best_params_bh.x[0], best_params_bh.x[1], best_params_bh.x[2]])
            self.best_params=best_params_bh
            print('BEST PARAMETERS\n',best_params_bh)
            
        elif method == 'diffev':
            best_params_de = differential_evolution(func=self.cost_function, bounds=bounds,
                                                   strategy='best2exp', maxiter=1000)
            
            thetas = np.array([best_params_de.x[0], best_params_de.x[1], best_params_de.x[2]])
            self.best_params=best_params_de
            print('BEST PARAMETERS\n',best_params_de)
            
            
                      
        elif method == 'dual_annealing':
            best_params_da = dual_annealing(self.cost_function, bounds=bounds,  visit=2.9,
                                            initial_temp=5e4, restart_temp_ratio=2e-6, maxiter=10000,
                                            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds,
                                            "options":{"ftol":1e-18, "maxiter":7}}
                                            )
            
            thetas = np.array([best_params_da.x[0], best_params_da.x[1], best_params_da.x[2]])
            self.best_params=best_params_da
            print('BEST PARAMETERS\n',best_params_da)
            
        else:
            AssertionError(f"Your method {method} is not in the method list. Misspelled?")
            
            
        print('thetas', thetas)
            
        self.tC = self.invert_rotations_spatial(self.cluster, thetas)   
        self.rect = self.pca_theta_calculation(self.dtps_org.T, self.tC,   plot=True)

        # self.rms_abs = self.rms_mismatch_abs(self.tC)
            
        self.thetas = thetas
        
        print(self.thetas)
        
        print('\n')
        print('---------------------------------------')
        print(f'Final rectilinearity: {self.rect}')
        print('---------------------------------------')
        print('\n')
        
        # print('absolute RMS [m]', self.rms_abs)
        
        plt.figure()
        plt.grid()
        plt.title("Cost curve")
        plt.plot(np.array(self.costvect))
        plt.ylabel(f'Misfit {method} value')
        plt.xlabel('Iteration')
        plt.show()

        return  self.thetas

