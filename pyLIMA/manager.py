# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:17:30 2015

@author: ebachelet
"""

###############################################################################

# General code manager

###############################################################################

import os
import time
import glob
import matplotlib.pyplot as plt

import numpy as np

import event
import telescopes
import microlmodels



def main(command_line):
   
    events_names=[event_name for event_name in os.listdir('../../Dun_lightcurves/') if ('OGLE2016BLG0676.dat'  in event_name) and ('~' not in event_name)]
    import pdb; pdb.set_trace()

    events = []
    start = time.time()
    results = []
    errors = []
   
    
    for event_name in events_names[0:20]:

        #name='Lightcurve_'+str(9994)+'_'
        name = event_name[:-4]
        current_event = event.Event()
        current_event.name = name
        current_event.ra = 269.39166666666665 
        current_event.dec = -29.22083333333333
        event_telescopes = [i for i in events_names if name  in i]
        #event_telescopes = ['OGLE-2016-BLG-0676.dat','MOA-2016-BLG-215_MOA_transformed.dat','MOA-2016-BLG-215_transformed.dat']
        #event_telescopes = ['MOA-2016-BLG-215_transformed.dat']
        #Names = ['OGLE','MOA','K2']
        event_telescopes = [event_name]
        count=0

        start=time.time()
        for event_telescope in event_telescopes:
            try :
               raw_light_curve = np.genfromtxt(command_line.input_directory + event_telescope, usecols=(0, 1, 2))
               good = np.where(raw_light_curve[:,1]<20)[0]
               raw_light_curve=raw_light_curve[good]
               lightcurve=np.array([raw_light_curve[:,0],raw_light_curve[:,1],raw_light_curve[:,2]]).T
               if lightcurve[0,0]>2450000 :
                   lightcurve[:,0] = lightcurve[:,0]-2450000
            except :
                pass
            
            telescope = telescopes.Telescope(name='K2', camera_filter='I', light_curve=lightcurve)
            telescope.gamma=0.5
            
            current_event.telescopes.append(telescope)
            count+=1
           
        print 'Start;', current_event.name
       
        current_event.find_survey('K2')
   
        current_event.check_event()
       
        Model = microlmodels.MLModels(current_event, command_line.model)
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()

        current_event.fit(Model,'DE')
        
        import pdb; pdb.set_trace()
        current_event.fits[0].produce_outputs()
        
        plt.show()
        
    end = time.time()
   
    print end - start

    all_results = [('Fits.txt', results),
                   ('Fits_Error.txt', errors)]

    for file_name, values in all_results:
        np.savetxt(os.path.join(command_line.output_directory, file_name), np.array(values), fmt="%s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='PSPL')
    parser.add_argument('-i', '--input_directory', default='../../Dun_lightcurves/') 
    parser.add_argument('-o', '--output_directory', default='/nethome/Desktop/Microlensing/K2_C9/Dun_lightcurves/Results/')
    parser.add_argument('-c', '--claret', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Claret2011/J_A+A_529_A75/')
    arguments = parser.parse_args()

    model = arguments.model

    main(arguments)
