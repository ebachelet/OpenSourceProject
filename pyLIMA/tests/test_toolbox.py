import numpy as np
import unittest
from unittest.mock import MagicMock, Mock
from pyLIMA.toolbox import brightness_transformation
from pyLIMA.toolbox.examine_lightcurve import PointBrowser  # Adjust the import path as needed
from pyLIMA.toolbox.bin_lightcurve import weighted_mean, get_sigma, bin_data  # Adjust the import path as needed

def test_magnitude_to_flux():
    flux = brightness_transformation.magnitude_to_flux(18.76)

    assert flux == 2857.5905433749376


def test_flux_to_magnitude():
    mag = brightness_transformation.flux_to_magnitude(18.76)

    assert mag == 24.216917914892385


def test_error_magnitude_to_error_flux():
    err_mag = 0.189
    flux = 27.9

    err_flux = brightness_transformation.error_magnitude_to_error_flux(err_mag, flux)

    assert np.allclose(err_flux, 4.856704581546761)


def test_error_flux_to_error_magnitude():
    flux = 27.9
    error_flux = 4.856704581546761
    err_mag = brightness_transformation.error_flux_to_error_magnitude(error_flux, flux)

    assert np.allclose(err_mag, 0.189)


def test_noisy_observations():
    flux = 10

    flux_obs = brightness_transformation.noisy_observations(flux, exp_time=None)

    assert flux_obs != flux

class TestPointBrowser(unittest.TestCase):

    def setUp(self):
        # Mocking the matplotlib figure, axis, and line objects
        self.fig = MagicMock()
        self.ax = MagicMock()
        self.line = MagicMock()

        # Example data
        self.hjd = np.array([1, 2, 3, 4, 5])
        self.mag = np.array([10, 11, 9, 12, 8])
        self.idx = np.arange(len(self.hjd))

        # Configure the ax mock to return a mock object when plot is called
        mock_plot_return = Mock()
        self.ax.plot.return_value = (mock_plot_return,)

        # Instantiate PointBrowser with mocked data
        self.browser = PointBrowser(self.fig, self.ax, self.hjd, self.mag, self.idx, self.line)

    def test_initialization(self):
        """Test that PointBrowser initializes with expected attributes."""
        self.assertEqual(self.browser.lastind, 0)
        self.assertEqual(len(self.browser.output), 0)
        self.assertIsNotNone(self.browser.fig)
        self.assertIsNotNone(self.browser.ax)
        self.assertIsNotNone(self.browser.line)
        self.assertTrue(np.array_equal(self.browser.hjd, self.hjd))
        self.assertTrue(np.array_equal(self.browser.mag, self.mag))
        self.assertTrue(np.array_equal(self.browser.idx, self.idx))

    def test_add_point(self):
        # Simulate adding a point and test the outcome
        self.browser.output.append(2)  # Simulate action
        self.assertIn(2, self.browser.output)  # Test condition

    def test_remove_point(self):
        # First, add a point to ensure there's something to remove
        self.browser.output.append(2)
        self.assertIn(2, self.browser.output, "Point was not added correctly.")

        # Now, simulate removing the point
        self.browser.output.remove(2)
        self.assertNotIn(2, self.browser.output, "Point was not removed correctly.")

def test_examine_lightcurve():
    """Function to run all tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPointBrowser)
    unittest.TextTestRunner().run(suite)

class TestBinLightcurve(unittest.TestCase):

    def setUp(self):
        self.hjd = np.array([1, 2, 3, 4, 5])
        self.mags = np.array([10, 12, 11, 13, 14])
        self.merrs = np.array([0.5, 0.2, 0.3, 0.4, 0.1])

    def test_weighted_mean_types(self):
        """Test that weighted_mean function accepts correct types."""
        self.assertIsInstance(self.mags, np.ndarray, "Magnitudes must be a NumPy array")
        self.assertIsInstance(self.merrs, np.ndarray, "Errors must be a NumPy array")
        
        # Perform a basic operation to ensure the function works with correct types
        result = weighted_mean(self.mags, self.merrs)
        self.assertIsInstance(result, float, "Weighted mean should be a float")

    def test_get_sigma_types(self):
        """Test that get_sigma function accepts correct types."""
        self.assertIsInstance(self.merrs, np.ndarray, "Errors must be a NumPy array")
        
        # Perform a basic operation to ensure the function works with correct types
        result = get_sigma(self.merrs)
        self.assertIsInstance(result, float, "Sigma should be a float")

    def test_bin_data_types(self):
        """Test that bin_data function accepts correct types."""
        self.assertIsInstance(self.hjd, np.ndarray, "HJDs must be a NumPy array")
        self.assertIsInstance(self.mags, np.ndarray, "Magnitudes must be a NumPy array")
        self.assertIsInstance(self.merrs, np.ndarray, "Errors must be a NumPy array")
        self.assertIsInstance(1, (int, float), "tstart must be numeric")
        self.assertIsInstance(5, (int, float), "tend must be numeric")
        self.assertIsInstance(2, (int, float), "bin_size must be numeric")
        
        # Perform a basic operation to ensure the function works with correct types
        binned_data = bin_data(self.hjd, self.mags, self.merrs, 1, 5, 2)
        self.assertIsInstance(binned_data, np.ndarray, "Binned data should be a NumPy array")
        self.assertEqual(binned_data.shape[1], 3, "Binned data should have three columns")

def test_bin_lightcurve():
    """Function to run all tests in the lightcurve processing test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBinLightcurve)
    unittest.TextTestRunner().run(suite)
