# -*- coding: utf-8 -*-
"""
Created on Wed 30 Sep 2020

@author: martindominik 
"""

from __future__ import division

import sys
import copy
import numpy as np

class _Deviation(object):
    """
    Groups evaluation of deviation of a point
    
    FLOAT sig;
    FLOAT sigscaled;
    FLOAT crit_dev;
    """
    
    def __init__(self,sig,sigscaled,crit_dev):
        self.sig = sig
        self.sigscaled = sigscaled
        self.crit_dev = crit_dev
        

class _DataPoint(object):
    """
    Holds information on a specific data point, corresponds to struct ANOMALY_LIST in signalmen.c
    
    Grouped in data_flux:
        FLOAT time;
        FLOAT mag;
        FLOAT err;
        FLOAT seeing;  (not implemented in pyLIMA)
        FLOAT backgnd; (not implemented in pyLIMA)
        FLOAT airmass; (not implemented in pyLIMA)
        FLOAT exptime; (not implemented in pyLIMA)
        int type;      (not implemented in pyLIMA)
        
    tel_idx:
        int arch_index;
        
    tel_name:
        char extsite[256];
        
    char filter;
 
    Grouped in deviation
        FLOAT sig;
        FLOAT sigscaled;
        FLOAT crit_dev;
    """
    
    def __init__(self,anom_status,data_idx,deviation=[]):
        self.data_idx = data_idx
        self.data_flux = anom_status._alldata[data_idx,0:3]
        self.tel_idx = anom_status._alldata[data_idx,5].astype(int)
        self.tel_name = anom_status.event.telescopes[self.tel_idx].name
        self.filter = anom_status.event.telescopes[self.tel_idx].filter
        self.deviation = deviation
        # Question: Should we store fluxes or magnitudes here -> see "signalmen.c" for details


class AnomalyStatus(object):
    """
    ######## AnomalyStatus class ########

    This class defines an AnomalyStatus object holding the assessment of an event for an ongoing anomaly 
    and providing methods to carry out the assessment
    
    It contains the complete SIGNALMEN management
    and builds on the Event class for fitting models to event data.

    Attributes :

         filename : specifies master file name for various input/output files to control SIGNALMEN
                      and log assessment

         alldata : numpy array of full photometric data sorted in time sequence

         event : instance of Event class with all data (old and new)

         old_event : instance of Event class with previous data and model

         eventNew : instance of Event class with data and model corresponding to current assessment step

         status : SIGNALMEN status (0 = ordinary, 1 = check, 2 = anomaly) after assessment

         prev_status : SIGNALMEN status (0 = ordinary, 1 = check, 2 = anomaly) before assessment 
 
	 time_prev : epoch of previous assessment (new assessment starts after this epoch)

	 [Some other attributes for management required, maybe bundle in one or several classes, e.g.
             list of anomalous points
             list of specific points to be excluded for modelling]

	 [For specific fits, data has to be copied from "alldata" to the telescopes attribute of "event_curr"
             taking into account the time range and further exclusion criteria]
  
         ["alldata" initially has to be populated with data stored in "event_all" and sorted in time sequence]
         
     unsigned long ndata; /* data to be considered for fit (0..ndata-1) */
     unsigned long totdata; /* total # data */
     unsigned long olddata; /* total # data previous run */
     
     ANOMALY_LIST_PTR exclude_list;
     boolean include_modelling;
     unsigned long include_Idx;
     

         
    """

    def __init__(self,event):
        """ Initialization of the attributes described above. """

        self.event = event
        # build array of all data, containing time, flux, flux_err, mag, mag_err, telescope_index, [deltas_north, deltas_east]
        # include [deltas_north, deltas_east] only if these are provided in the deltas_position array of the telescope object
        alldata_list = []
        tel_idx = 0
        if isinstance(self.event.telescopes[0].deltas_positions,list):
            par_deltas_exist = self.event.telescopes[0].deltas_positions
        else:
            par_deltas_exist = self.event.telescopes[0].deltas_positions.size != 0
             # check whether parallax offsets have been calculated according to the model type
        for telescope in self.event.telescopes:
            dlen = len(telescope.lightcurve_flux)
            if par_deltas_exist:
                lightcurve = np.c_[telescope.lightcurve_flux,telescope.lightcurve_magnitude[:,1:3],np.full(dlen,tel_idx),
                    telescope.deltas_positions[0,:],telescope.deltas_positions[1,:]]
            else:
                lightcurve = np.c_[telescope.lightcurve_flux,telescope.lightcurve_magnitude[:,1:3],np.full(dlen,tel_idx)]
            alldata_list.append(lightcurve)
            tel_idx += 1
        self._alldata = np.concatenate(alldata_list)
        # sort array with all data in time sequence
        # do not use quicksort given that array is partially presorted
        self._alldata = self._alldata[self._alldata[:,0].argsort(kind='heapsort')]
        self._totdata = len(self._alldata)  # Total number of data points
        
        
    def _filter_data(self, this_event, ndata, exclude_list):
        """ Filter event data from self._alldata to this_event.telescopes
        
        (please explain parameters)
        
        """
        
        # extract data indices from exclude_list
        if exclude_list:
            exclude_indices = [datapoint.data_idx for datapoint in exclude_list]
            mask = np.ones(len(self._alldata),dtype=bool)
            mask[exclude_indices] = False
            selected_data = self._alldata[mask][0:ndata,:]
        else:
            selected_data = self._alldata[0:ndata,:]
            
        if isinstance(self.event.telescopes[0].deltas_positions,list):
            par_deltas_exist = self.event.telescopes[0].deltas_positions
        else:
            par_deltas_exist = self.event.telescopes[0].deltas_positions.size != 0
             # check whether parallax offsets have been calculated according to the model type
        tel_idx = 0
        for telescope in this_event.telescopes:
            selection = selected_data[selected_data[:,5].astype(int) == tel_idx]
            telescope.lightcurve_flux = selection[:,0:3]
            if par_deltas_exist:
                telescope.lightcurve_magnitude = selection[:,np.array([True,False,False,True,True,False,False,False])]
                telescope.deltas_positions = np.array([selection[:,6],selection[:,7]])
            else:
                telescope.lightcurve_magnitude = selection[:,np.array([True,False,False,True,True,False])]
            tel_idx += 1


    def assess(self, model, method, start_time, DE_population_size=10, flux_estimation_MCMC='MCMC', fix_parameters_dictionnary=None,
            grid_resolution=10, computational_pool=None, binary_regime=None,
            robust=False):
        """ Method to assess an event for an ongoing anomaly
        
        start_time  Start assessment after start_time

        [add details...]

        """
        
        # This is some sample test code
        self.old_event = copy.deepcopy(self.event)  # copy from Event object provided as argument
        self._filter_data(self.old_event,self._totdata,[])
        self.old_event.fit(model,method)
           # need to use robust fitting for SIGNALMEN assessment !!!!
        
        # get index of first data point AFTER start_time, self._alldata is sorted in time order
        next_idx = self._alldata[:,0].searchsorted(start_time, side='right')
        
        while next_idx < self._totdata:   # main loop to step through data points
            next_idx += 1
            
        # Note: pyLIMA needs to be enabled to support fits with "insufficient" data -> provide defined result
        #           [see SIGNALMEN recipe]
        #       check both the inclusion of data in fit, and providing well-defined fit result


