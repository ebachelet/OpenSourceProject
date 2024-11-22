import matplotlib
matplotlib.use('TKAgg')  # Set the backend before importing pyplot
import numpy as np
import matplotlib.pyplot as plt
from sys import argv, exit

# Examine a lightcurve to get the indexes of a list of points that can be used for masking

class PointBrowser:
    """
    A tool for interactive point selection and navigation in a matplotlib 
    lightcurve plot.
    
    Click on a point to highlight it. Navigate through points using 'n'/'right'
    and 'p'/'left' keys for next and previous points. Add a new point to the list
    by pressing 'a' or remove it by pressing 'r'. Display the list of indexes of the
    selected points in the terminal by pressing 't'.
    """
    
    def __init__(self, fig, ax, hjd, mag, idx, line):
        """
        Initializes the PointBrowser instance with plot axis, data, and index.
        
        Parameters:
        - fig: The matplotlib figure object for the plot.
        - ax: The matplotlib axis object for plotting.
        - hjd, mag: Arrays of data points for plotting.
        - idx: Array of indices corresponding to data points.
        - line: The matplotlib line object for interaction.
        """
        self.fig = fig
        self.ax = ax
        self.hjd = hjd
        self.mag = mag
        self.idx = idx
        self.line = line
        self.lastind = 0
        self.text = ax.text(0.05, 0.95, 'selected: none', 
                            transform=ax.transAxes, va='top')
        self.selected, = ax.plot([], [], 'o', ms=8, alpha=0.7, color='teal', visible=False)
        self.added_points, = ax.plot([], [], 'rx', ms=6, mew=2, alpha=0.7)  # For visualizing added points
        self.output = []

    def onpress(self, event):
        """Handles keyboard press events for navigating and modifying points."""
        if self.lastind is None or event.key not in ('n', 'p', 'a', 'r', 't', 'left', 'right'):
            return
        inc = 0
        if event.key in ('n', 'right'):
            inc = 1
        elif event.key in ('p', 'left'):
            inc = -1
        elif event.key == 'a' and self.idx[self.lastind] not in self.output:
            self.output.append(self.idx[self.lastind])
            self.update_added_points()  # Update visualization of added points
            return
        elif event.key == 'r' and self.idx[self.lastind] in self.output:
            self.output.remove(self.idx[self.lastind])
            self.update_added_points()  # Update visualization of added points
            return
        elif event.key == 't':
            self.output.sort()
            print(self.output)
            return
        
        self.lastind = (self.lastind + inc) % len(self.hjd) # Cycle through points
        self.update()
    
    def update_added_points(self):
        """Updates the visualization of added points."""
        if not self.output:
            self.added_points.set_data([], [])  # No points to display
        else:
            # Get the hjd and mag values for added points
            added_hjd = [self.hjd[i] for i in self.output]
            added_mag = [self.mag[i] for i in self.output]
            self.added_points.set_data(added_hjd, added_mag)
        self.fig.canvas.draw()
          
    def onpick(self, event):
        """Handles mouse pick events to select points."""
        if event.artist != self.line or not len(event.ind):
            return True
        
        distances = np.hypot(event.mouseevent.xdata - self.hjd[event.ind],
                             event.mouseevent.ydata - self.mag[event.ind])
        indmin = distances.argmin()
        self.lastind = event.ind[indmin]
        self.update()
    
    def update(self):
        """Updates the plot with the selected point highlighted."""
        if self.lastind is None:
            return
        dataind = self.lastind
        self.selected.set_visible(True)
        self.selected.set_data(self.hjd[dataind], self.mag[dataind])
        self.text.set_text(f'selected: {self.idx[dataind]}, HJD: {self.hjd[dataind]:.2f}')
        self.update_added_points()  # Ensure added points are updated alongside
        self.fig.canvas.draw()

if __name__ == '__main__':
    print('Assumes the first line of the .csv file contains the column headings.')
    if len(argv) != 2:
        print('Usage: python examine_lightcurve.py /path/to/<datafile>.csv')
        exit()

    datafile = argv[1]
    try:
        # Skip the column descriptions in the first row
        try:
            hjd, mag, merr = np.loadtxt(datafile, delimiter=',', skiprows=1, usecols=(0, 1, 2), 
                                        dtype=float, unpack=True)
        except ValueError:
            hjd, mag, merr = np.loadtxt(datafile, delimiter=',', skiprows=1, usecols=(0, 2, 3), 
                                        dtype=float, unpack=True)
        idx = np.arange(len(hjd))
    except Exception as e:
        print(f'Error reading {datafile}: {e}')
        exit()

    fig, ax = plt.subplots()
    ax.set_title(datafile)
    
    # Set the labels for the x-axis and y-axis
    ax.set_xlabel("Julian Date")
    ax.set_ylabel("Magnitude")
    line, = ax.plot(hjd, mag, 'k.', picker=5)  # 5 points tolerance
    plt.ylim(max(mag) + 0.1, min(mag) - 0.1)
    ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Customize the x-axis with exact Julian dates
    #tick_interval = 30  # Adjust this based on your dataset's range and density
    #ticks = np.arange(min(hjd), max(hjd), tick_interval)
    #ax.set_xticks(ticks)
    #ax.set_xticklabels([f'{tick:.0f}' for tick in ticks], rotation=25, ha='right')  # Format for clarity
    
    browser = PointBrowser(fig, ax, hjd, mag, idx, line)
    fig.canvas.mpl_connect('pick_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event', browser.onpress)
    
    plt.show()

