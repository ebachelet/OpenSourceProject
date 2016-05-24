# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:25:12 2016

@author: ebachelet
"""
import numpy as np
import os.path 


from pyLIMA import telescopes


def test_init():
    
    telescope = telescopes.Telescope()
    
    assert telescope.name == 'NDG'
    assert telescope.filter == 'I'
    assert len(telescope.lightcurve) == 0
    assert telescope.location == 'Earth'
    
    assert telescope.location == 'Earth'
    assert len(telescope.lightcurve_flux) == 0
    assert telescope.altitude == 0.0
    assert telescope.longitude == 0.57
    assert telescope.latitude == 49.49
    assert telescope.gamma == 0.0
    assert len(telescope.deltas_positions) == 0
    
    telescope2 = telescopes.Telescope('Goleta','sdss_i',np.array([[0,1,0.1],[3,4,0.1]]))
    
    assert telescope2.name == 'Goleta'
    assert telescope2.filter == 'sdss_i'
    assert telescope2.lightcurve.shape == (2,3)
    
    telescope2.location = 'Space'
    telescope2.altitude = 1.0
    telescope2.longitude = -150.0
    telescope2.latitude = 35.0
    telescope2.gamma = 0.6
    
    assert telescope2.location == 'Space'      
    assert telescope2.altitude == 1.0
    assert telescope2.longitude == -150.0
    assert telescope2.latitude == 35.0
    assert telescope2.gamma == 0.6
    
def test_clean_data_already_clean():
    
     telescope = telescopes.Telescope(light_curve=np.array([[0,1,0.1],[3,4,0.1],[5,6,0.1]]))
     
     
     clean_lightcurve = telescope.clean_data()
     assert np.allclose(clean_lightcurve,np.array([[0,1,0.1],[3,4,0.1],[5,6,0.1]]))

def test_clean_data_not_clean():
     
     telescope = telescopes.Telescope(light_curve=np.array([[0,1,0.1],[3,np.nan,0.1],[5,6,0.1],[7,np.nan,np.nan],[8,1,27.0],[9,2,0.03]]))  
     clean_lightcurve = telescope.clean_data()
     assert np.allclose(clean_lightcurve,np.array([[0,1,0.1],[5,6,0.1],[9,2,0.03]]))

def test_clean_data_not_clean_but_2_points():   
    
     telescope = telescopes.Telescope(light_curve=np.array([[0,1,0.1],[2,3,-28.0]]))
     clean_lightcurve = telescope.clean_data()
     assert np.alltrue(clean_lightcurve==np.array([[0,1,0.1],[2,3,-28.0]]))
     
def test_lightcurve_in_flux():
    
     telescope = telescopes.Telescope(light_curve=np.array([[0,1,0.1],[3,4,0.1],[5,6,0.1]]))
             
     telescope.lightcurve_flux = telescope.lightcurve_in_flux()
            
     assert np.allclose(telescope.lightcurve_flux,np.array([[  0.00000000e+00,   3.63078055e+10,  -3.34407247e+09],
                                                            [  3.00000000e+00,   2.29086765e+09,  -2.10996708e+08],
                                                            [  5.00000000e+00,   3.63078055e+08,  -3.34407247e+07]]))

def test_find_gamma():
    
    telescope = telescopes.Telescope(camera_filter="z'")
    full_path = os.path.abspath(__file__)
    directory, filename = os.path.split(full_path)
    
    telescope.find_gamma(32000.0,4.5, os.path.join(directory, '../data/'))
    EPSILON = 0.001

    assert np.abs(telescope.gamma-0.127056)<=EPSILON
    
    
def test_n_data():
    
   telescope = telescopes.Telescope(light_curve=np.array([[0,1,0.1],[3,np.nan,0.1],[5,6,0.1],[7,np.nan,np.nan],[8,1,27.0],[9,2,0.03]]))  
   telescope.lightcurve_flux = telescope.lightcurve_in_flux()
    
   assert telescope.n_data() == 6
   assert telescope.n_data('Flux') == 3


def test_arrange_the_lightcurve_columns_good_columns():
    
   telescope = telescopes.Telescope(light_curve=np.array([[0,1,0.1],[3,0.0,0.1],[5,6,0.1],[7,0.0,0.0],[8,1,27.0],[9,2,0.03]]),
                                    light_curve_dictionnary = {'time': 0, 'mag' : 1, 'err_mag' : 2 })  
   
    
   assert np.allclose(telescope.lightcurve,np.array([[0,1,0.1],[3,0.0,0.1],[5,6,0.1],[7,0.0,0.0],[8,1,27.0],[9,2,0.03]]))
  
def test_arrange_the_lightcurve_columns_invert_time_magnitude():
    
   telescope = telescopes.Telescope(light_curve=np.array([[0,1,0.1],[3,0.0,0.1],[5,6,0.1],[7,0.0,0.0],[8,1,27.0],[9,2,0.03]]),
                                    light_curve_dictionnary = {'time': 1, 'mag' : 0, 'err_mag' : 2 })  
   
    
   assert np.allclose(telescope.lightcurve,np.array([[1,0,0.1],[0.0,3,0.1],[6,5,0.1],[0.0,7,0.0],[1,8,27.0],[2,9,0.03]]))
