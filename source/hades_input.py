import numpy as num
import datetime
import LatLongUTMconversion
import LatLon2Cart
import os
import sys

km=1000.

class hades_input():
    """This class prepares earthquake data (formatted correctly) for the hades_location
    class. 
    
    Parameters
    ----------
    data_path : `string`
        Path to the data (event, stations). 
    event_file : `string`
        Filename of the event file (.dat).
    station_file : `string`
        Filename of the station file (.txt)
    sta_select : `list`
        A selection of two stations if multiple stations are available in the event file. 
    """

    def __init__(self,data_path,event_file,station_file, sta_select):
        event_file=os.path.join(data_path,event_file)
 
        references,refevid,evtsp,reforig,events=hades_input.__read_evfile(event_file)
        self.references=references
        self.refevid=refevid
        self.origin=reforig
        self.data=evtsp
        self.events=events
        self.sta_select = sta_select
        
        # station file
        station_file=os.path.join(data_path,station_file)
        stations=hades_input.__read_stafile(station_file,reforig)
        print(sta_select)
        self.stations= stations
      
        self.stations = {f'{sta_select[0]}' : stations[sta_select[0]], f'{sta_select[1]}' : stations[sta_select[1]]}
        print(self.stations)


    def __read_evfile(input_file):
        """Reads the input file, outputs all data of interest.
        
        There are always >4 master events in the input file, even in relative mode. This is
        because for the relative mode, the one pre-located master event must be accompanied by 
        three events to construct a relative coordinate system. 

        Parameters
        ----------
        input_file : `string`
            Path to the input file to be read
        
        Returns
        -------
        references : `numpy.ndarray`
            Array of the used reference events index in the events file. 
        refevid : `list`
            ID of the reference events (their name in the input file).
        evtsp : `numpy.ndarray`
            Array containing the (ts-tp) of each event measured at each station.
        refor : `tuple`
            Origin of the reference coordinate system in original coordinate system.
        events : `list`
            List of event id's and their order.
        
        """
        with open(input_file, 'r') as f:
            toks=f.readline().split(';')
            latref,lonref=eval(toks[1]),eval(toks[2])
            orig=LatLon2Cart.Coordinates(latref,lonref,0)
            #z0,e0,n0=LatLongUTMconversion.LLtoUTM(23, eval(toks[1]), eval(toks[2])) #order lat lon
            #e0,n0,z0 = orig.geo2cart(eval(toks[1]), eval(toks[2]),0)
            try:
                depthref=eval(toks[3])*km
            except:
                depthref=0.
            refor=(latref,lonref,depthref)
            refevid=[]
            events=[]
            evtsp={}
            references=[]
            for line in f:
                toks=line.split(';')
                if toks[0][0]=='#':
                    evid=toks[0]
                    evtsp[evid]={}
                    evdate=toks[1][0:10]
                    if toks[0][1]=='R':
                        #_,e,n=LatLongUTMconversion.LLtoUTM(23, eval(toks[2]), eval(toks[3])) # order lat lon
                        e,n,z = orig.geo2cart(eval(toks[2]), eval(toks[3]),0)
                        depth=eval(toks[4])*km
                        refevid.append(evid)
                        references.append([e,n,depth])
                    else:
                        events.append(evid)
                else:
                    sta=toks[0]
                    if toks[1]=='na' or toks[2]=='na':
                        continue
                    tp=datetime.datetime.strptime(evdate+'T'+toks[1], '%Y/%m/%dT%H:%M:%S.%f')
                    ts=datetime.datetime.strptime(evdate+'T'+toks[2], '%Y/%m/%dT%H:%M:%S.%f')
                    tsp=(ts-tp).total_seconds()
                    evtsp[evid][sta]=[tp,ts,tsp]

        references=num.array(references)
        return references,refevid,evtsp,refor,events


    def __read_stafile(input_file,refor):
        """Reads the station file."""

        (latref,lonref,depthref)=refor
        stations={}
        orig=LatLon2Cart.Coordinates(latref,lonref,0)
        with open(input_file, 'r') as f:
            for line in f:
                toks=line.split()
                sta=toks[0]
                #z,e,n=LatLongUTMconversion.LLtoUTM(23, eval(toks[1]), eval(toks[2])) #order lat lon
                e,n,z = orig.geo2cart(eval(toks[1]), eval(toks[2]),0)
                elev=eval(toks[3])*km
                stations[sta]=[e,n,elev]
                #if z==z0:
                #    stations[sta]=[e,n,elev]
                #else:
                #    print('cluster and stations are in different UTM zones')
                #    sys.exit()
        return stations


    def distance_calculation(self, Vp, Vs, sta, dist=None):
        ''' This method calculates the inter-event distance matrix
        for the entire dataset. Input: It requires the P and S wave velocities and
        the station name. For single station mode sta needs to be  string with the name of the
        station "STANAME", for multistation sta needs to be a list with the stations
        ["STANAME_1", "STANAME_2", ... ,"STANAME_N"].
        '''
        
        if dist is not None:
            predist = True
            print("Predefined distances used")
        else:
            predist = False


        print('stations used:',sta)
        evrefid=self.refevid
        evrefs=self.references
        self.sel_sta=sta
        evids=list((self.data).keys())
        events=self.refevid+self.events
        nevs=len(events)
        nref=len(evrefid)

        print(f'{nref} master events used')

        if predist == False:
            distances=num.zeros([nevs,nevs])
            kv=(Vp*Vs)/(Vp-Vs)
            for i in range(nevs-1):
                for j in range(i+1,nevs):
                    if (i<nref) and (j<nref):
                        distances[i,j]=num.sqrt((evrefs[i][0]-evrefs[j][0])**2+(evrefs[i][1]-evrefs[j][1])**2+(evrefs[i][2]-evrefs[j][2])**2)
                    else:
                        tsp_ev1=self.data[events[i]]
                        tsp_ev2=self.data[events[j]]
                        distances[i,j]=hades_input.__interev_distance(tsp_ev1,tsp_ev2,kv,sta,self.stations)
                        distances[j,i]=distances[i,j]
            self.distances=distances
            print('DISTANCE SHAPE',num.shape(distances))
            self.events=events
        else:
            self.distances=dist
            self.events=events

        self.vp=Vp
        self.vs=Vs


    def __interev_distance(tsp_ev1,tsp_ev2,kv,sta, stations):
        if (type(sta)==str) and sta!='ALL':
            ie_dist=hades_input.__onesta_interev_distance(tsp_ev1,tsp_ev2,kv,sta)
        elif type(sta)==list and len(sta)==2:
            ie_dist=hades_input.__twosta_interev_distance(tsp_ev1,tsp_ev2,kv,sta)
        else:
            print('Error in reading the station list for interevent distance')
            sys.exit()
        return ie_dist


    def __onesta_interev_distance(tsp_ev1,tsp_ev2,kv,sta):
        stalist=list(set(tsp_ev1.keys()) & set(tsp_ev2.keys()))
        if sta in stalist:
            ie_dist=num.abs(tsp_ev1[sta][-1]-tsp_ev2[sta][-1])*kv
            return ie_dist
        else:
            return num.NaN


    def __twosta_interev_distance(tsp_ev1,tsp_ev2,kv,sta):
        sta1=sta[0]
        sta2=sta[1]
        stalist=list(set(tsp_ev1.keys()) & set(tsp_ev2.keys()))
        if sta1 in stalist and sta2 in stalist:
            iedist_sta1=num.abs(tsp_ev1[sta1][-1]-tsp_ev2[sta1][-1])*kv
            iedist_sta2=num.abs(tsp_ev1[sta2][-1]-tsp_ev2[sta2][-1])*kv
            ie_dist=num.sqrt(iedist_sta1**2+iedist_sta2**2)
            return ie_dist
        else:
            return num.NaN


    def relative_frame(self,Vp,Vs,sta,y_ref=-1,z_ref=-1,fixed_depth=0):

        kv=(Vp*Vs)/(Vp-Vs)
        if len(self.refevid)>4:
            print('For relative frame construction you need only 4 events, you selected : ', len(self.refevid))
            sys.exit()
        else:
            events=self.refevid

        d=num.zeros([4,4])
        for i in range(3):
            tsp_ev1=self.data[events[i]]
            for j in range(i+1,4):
                tsp_ev2=self.data[events[j]]
                d[i,j]=hades_input.__interev_distance(tsp_ev1,tsp_ev2,kv,sta,self.stations)
                d[j,i]=d[i,j]

        references=num.zeros([4,3])

        references[0,0]=0.
        references[0,1]=0.
        references[0,2]=0.

        references[1,0]=d[0,1]
        references[1,1]=0.
        references[1,2]=0.

        references[2,0]=(d[0,2]**2-d[1,2]**2)/(2*references[1,0])+(references[1,0]/2)
        references[2,1]=y_ref*num.sqrt(d[0,2]**2-references[2,0]**2)
        references[2,2]=0.

        references[3,0]=(d[0,3]**2-d[1,3]**2)/(2*references[1,0])+(references[1,0]/2)
        references[3,1]=(d[1,3]**2-d[2,3]**2-(references[3,0]-references[1,0])**2+(references[3,0]-references[2,0])**2)/(2*references[2,1])+(references[2,1]/2)

        if fixed_depth:
            references[3,2]=fixed_depth*1000.
        else:
            references[3,2]=z_ref*num.sqrt(d[0,3]**2-references[3,0]**2-references[3,1]**2)

        self.rel_references=references
        self.refevid=events
