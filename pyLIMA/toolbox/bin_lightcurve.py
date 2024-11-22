import numpy as np
from astropy.timeseries import TimeSeries
import os
import sys

def weighted_mean(my_mags, my_errs):
    ''' Function to return the weighted mean for the magnitude array
    (inverse-variance weighting)
    my_mags: magnitude measurements
    my_merrs: uncertainties in the measurements
    '''
    weights = 1.0/my_errs**2
    weighted_mean_mag = np.sum(my_mags * weights) / np.sum(weights)
    
    return weighted_mean_mag


def get_sigma(my_errs):
    ''' Function to return a single value for the error array
    merrs: uncertainties in the measurements
    '''
    weights = 1.0/my_errs**2
    new_sigma = np.sqrt( np.sum(weights**2 * my_errs**2) / np.sum(weights)**2)
    
    return new_sigma

def bin_data(hjds, mags, merrs, tstart, tend, bin_size):
    ''' Function to bin a timeseries data set
    hjds: heliocentric julian dates corresponding to the magnitude measurements
    mags: magnitude measurements at those julian dates 
    merrs: uncertainties in the measurements at those julian dates
    tstart: julian day to start from 
    tend: julian day to stop at
    bin_size: the bin size in days
    '''
    new_hjds = []
    new_mags = []
    new_merrs = []
    timestamp = tstart
    while timestamp < tend:
        next_timestamp = timestamp + bin_size
        timeclips = np.where( (timestamp <= hjds) & (hjds < next_timestamp) )
        hjds_temp = hjds[timeclips]
        mags_temp = mags[timeclips]
        merrs_temp = merrs[timeclips]
        
        if hjds_temp.size != 0:
            new_hjds.append(np.mean(hjds_temp))
            new_merrs.append(get_sigma(merrs_temp))
            new_mags.append(weighted_mean(mags_temp,merrs_temp))
        
        timestamp = next_timestamp
    
    binned_data = np.empty([len(new_hjds),3])
    binned_data[:,0] = new_hjds
    binned_data[:,1] = new_mags
    binned_data[:,2] = new_merrs
    
    return binned_data

if __name__ == '__main__':
    run_check = True # Set to False if you don't want to get a plotting check at the end
    # Check for command-line arguments
    # Assumes the first line in the light curve file includes the column descriptions: time, magnitude, error
    if len(sys.argv) < 2:
        print("Usage: python bin_lightcurve.py <input_lightcurve_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if len(sys.argv) == 3:
        output_file = sys.argv[2]
    else:
        # Create a default output filename by appending '_binned' before the file extension
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_binned{ext}"
    
    # Data points to be masked by index (if any), default is no masking
    bad_data = []
    
    # Read data as astropy TimeSeries Table and remove any masked data points
    ts = TimeSeries.read(input_file, time_column='time', time_format='mjd')
    ts.remove_rows(bad_data)
    
    # Split up data for processing and define binning ranges
    hjds = ts['time'].value
    mags = ts['magnitude'].value
    merrs = ts['error'].value
    
    tstart = hjds[0]
    tend = hjds[-1]+0.5
    
    bin_size = 1.0 # Default value is daily binning when run from the command line
    
    binned_data = bin_data(hjds, mags, merrs, tstart, tend, bin_size)
    
    # Save the binned data
    np.savetxt(output_file, binned_data, delimiter=',')
    
    # Plotting check (if needed)
    if run_check:
        import matplotlib.pyplot as plt
        
        new_data = bin_data(hjds, mags, merrs, tstart, tend, bin_size)    
        
        plt.errorbar(new_data[:,0],new_data[:,1],yerr=new_data[:,2], fmt='ro', label="binned")
        plt.errorbar(hjds,mags,yerr=merrs, fmt='k.',alpha=0.2, label="unbinned")
        plt.gca().invert_yaxis()
        # Add labels to the x-axis and y-axis
        plt.xlabel("Julian Date")
        plt.ylabel("Magnitude")
        plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.show()

