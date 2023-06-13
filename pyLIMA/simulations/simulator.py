import astropy
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy.time import Time
from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA.priors import parameters_boundaries
from pyLIMA.toolbox import brightness_transformation


def simulate_a_microlensing_event(name='Microlensing pyLIMA simulation', ra=270,
                                  dec=-30):
    """
    Function to find initial DSPL guess

    Parameters
    ----------
    name : str, event name
    ra : float, the event right ascension
    dec : float, the event dec

    Returns
    -------
    fake_event : object, an event object
    """
    fake_event = event.Event(ra=ra, dec=dec)
    fake_event.name = name

    return fake_event


def simulate_a_telescope(name, time_start=2460000, time_end=2460500, sampling=0.25,
                         location=
                         'Earth', filter='I', uniform_sampling=False,
                         timestamps=[], altitude=0, longitude=0, latitude=0,
                         spacecraft_name=None,
                         spacecraft_positions={'astrometry': [], 'photometry': []},
                         bad_weather_percentage=0.0,
                         minimum_alt=20, moon_windows_avoidance=20,
                         maximum_moon_illumination=100.0, photometry=True,
                         astrometry=True, pixel_scale=1, ra=270, dec=-30):
    """
    Function to find initial DSPL guess

    Parameters
    ----------
    name : str, event name
    ra : float, the event right ascension
    dec : float, the event dec

    Returns
    -------
    fake_event : object, an event object
    """

    """ Simulate a telescope. More details in the telescopes module. The observations
    simulation are made for the
        full time windows, then limitation are applied :
            - Sun has to be below horizon : Sun< -18
            - Moon has to be more than the moon_windows_avoidance distance from the
            target
            - Observations altitude of the target have to be bigger than minimum_alt

    :param str name:  the name of the telescope.
    :param object event: the microlensing event you look at
    :param float time_start: the start of observations in JD
    :param float time_end: the end of observations in JD
    :param float sampling: the hour sampling.
    :param str location: the location of the telescope.
    :param str filter: the filter used for observations
    :param boolean uniform_sampling: set it to True if you want no bad weather,
    no moon avoidance etc....

    :param float altitude: the altitude in meters if the telescope
    :param float longitude: the longitude in degree of the telescope location
    :param float latitude: the latitude in degree of the telescope location

    :param str spacecraft_name: the name of your satellite according to JPL horizons

    :param float bad_weather_percentage: the percentage of bad nights
    :param float minimum_alt: the minimum altitude ini degrees that your telescope
    can go to.
    :param float moon_windows_avoidance: the minimum distance in degrees accepted
    between the target and the Moon
    :param float maximum_moon_illumination: the maximum Moon brightness you allow in
    percentage
    :return: a telescope object
    :rtype: object
    """

    if timestamps == []:

        if (uniform_sampling is False) & (location != 'Space'):

            earth_location = EarthLocation(lon=longitude * astropy.units.deg,
                                           lat=latitude * astropy.units.deg,
                                           height=altitude * astropy.units.m)

            target = SkyCoord(ra, dec, unit='deg')

            minimum_sampling = sampling

            time_of_observations = time_simulation(time_start, time_end,
                                                   minimum_sampling,
                                                   bad_weather_percentage)

            time_convertion = Time(time_of_observations, format='jd').isot

            telescope_altaz = target.transform_to(
                AltAz(obstime=time_convertion, location=earth_location))
            altazframe = AltAz(obstime=time_convertion, location=earth_location)
            Sun = get_sun(Time(time_of_observations, format='jd')).transform_to(
                altazframe)
            Moon = get_moon(Time(time_of_observations, format='jd')).transform_to(
                altazframe)
            Moon_illumination = moon_illumination(Sun, Moon)
            Moon_separation = target.separation(Moon)

            observing_windows = \
                np.where((telescope_altaz.alt > minimum_alt * astropy.units.deg)
                         & (Sun.alt < -18 * astropy.units.deg)
                         & (
                                     Moon_separation > moon_windows_avoidance *
                                     astropy.units.deg)
                         & (Moon_illumination < maximum_moon_illumination)
                         )[0]

            time_of_observations = time_of_observations[observing_windows]


        else:

            time_of_observations = np.arange(time_start, time_end, sampling / 24.0)

    else:

        time_of_observations = np.array(timestamps)

    if photometry & (len(time_of_observations) > 0):

        lightcurveflux = np.ones((len(time_of_observations), 3)) * 42
        lightcurveflux[:, 0] = time_of_observations

    else:

        lightcurveflux = None

    if astrometry:

        astrometry = np.ones((len(time_of_observations), 5)) * 42
        astrometry[:, 0] = time_of_observations

    else:

        astrometry = None

    telescope = telescopes.Telescope(name=name, camera_filter=filter,
                                     pixel_scale=pixel_scale,
                                     light_curve=lightcurveflux,
                                     light_curve_names=['time', 'flux', 'err_flux'],
                                     light_curve_units=['JD', 'w/m^2', 'w/m^2'],
                                     astrometry=astrometry,
                                     astrometry_names=['time', 'ra', 'err_ra', 'dec',
                                                       'err_dec'],
                                     astrometry_units=['JD', 'deg', 'deg', 'deg',
                                                       'deg'],
                                     location=location, spacecraft_name=spacecraft_name,
                                     spacecraft_positions=spacecraft_positions)
    return telescope


def time_simulation(time_start, time_end, sampling, bad_weather_percentage):
    """ Simulate observing time during the observing windows, rejecting windows with
    bad weather.

    :param float time_start: the start of observations in JD
    :param float time_end: the end of observations in JD
    :param float sampling: the number of points observed per hour.
    :param float bad_weather_percentage: the percentage of bad nights

    :return: a numpy array which represents the time of observations

    :rtype: array_like

    """

    time_initial = np.arange(time_start, time_end, sampling / 24.)
    total_number_of_days = int(time_end - time_start)

    time_observed = []
    night_begin = time_start

    for i in range(total_number_of_days):

        good_weather = np.random.uniform(0, 1)

        if good_weather > bad_weather_percentage:

            mask = (time_initial >= night_begin) & (time_initial < night_begin + 1)
            time_observed = np.append(time_observed, time_initial[mask])

        else:

            pass

        night_begin += 1

    time_of_observations = np.array(time_observed)

    return time_of_observations


def moon_illumination(sun, moon):
    """The moon illumination expressed as a percentage.

            :param astropy sun: the sun ephemeris
            :param astropy moon: the moon ephemeris

            :return: a numpy array indicated the moon illumination.

            :rtype: array_like

    """

    geocentric_elongation = sun.separation(moon).rad
    selenocentric_elongation = np.arctan2(sun.distance * np.sin(geocentric_elongation),
                                          moon.distance - sun.distance * np.cos(
                                              geocentric_elongation))

    illumination = (1 + np.cos(selenocentric_elongation)) / 2.0

    return illumination


def simulate_microlensing_model_parameters(model):
    """ Simulate parameters given the desired model. Parameters are selected in
    uniform distribution inside
        parameters_boundaries given by the microlguess modules. The exception is 'to'
        where it is selected
        to enter inside telescopes observations.

        :param object event: the microlensing event you look at. More details in
        event module


        :return: fake_parameters, a set of parameters
        :rtype: list
    """

    model.define_model_parameters()
    boundaries = parameters_boundaries.parameters_boundaries(model.event,
                                                             model.model_dictionnary)

    fake_parameters = []

    for ind, key in enumerate(model.model_dictionnary.keys()):

        try:

            if 'fsource' in key:
                break

            fake_parameters.append(
                np.random.uniform(boundaries[ind][0], boundaries[ind][1]))

        except AttributeError:

            pass

    fake_fluxes_parameters = simulate_fluxes_parameters(model.event.telescopes,
                                                        source_magnitude=[10, 20],
                                                        blend_magnitude=[19, 22])
    fake_parameters += fake_fluxes_parameters

    # t_0 limit fix
    mins_time = []
    maxs_time = []

    for telescope in model.event.telescopes:

        if telescope.lightcurve_flux is not None:
            mins_time.append(np.min(telescope.lightcurve_flux['time'].value))
            maxs_time.append(np.max(telescope.lightcurve_flux['time'].value))

        if telescope.astrometry is not None:
            mins_time.append(np.min(telescope.astrometry['time'].value))
            maxs_time.append(np.max(telescope.astrometry['time'].value))

    fake_parameters[0] = np.random.uniform(np.min(mins_time), np.max(maxs_time))

    if model.parallax_model[0] != 'None':
        fake_parameters[0] = np.random.uniform(model.parallax_model[1] - 1,
                                               model.parallax_model[1] + 1)

    return fake_parameters


def simulate_fluxes_parameters(list_of_telescopes, source_magnitude=[10, 20],
                               blend_magnitude=[10, 20]):
    """ Simulate flux parameters (magnitude_source , g) for the telescopes. More
    details in microlmodels module

    :param list list_of_telescopes: a list of telescopes object

    :return: fake_fluxes parameters, a set of fluxes parameters
    :rtype: list

    """

    fake_fluxes_parameters = []

    for telescope in list_of_telescopes:
        magnitude_source = np.random.uniform(source_magnitude[0], source_magnitude[1])
        flux_source = brightness_transformation.magnitude_to_flux(magnitude_source)

        magnitude_blend = np.random.uniform(blend_magnitude[0], blend_magnitude[1])
        flux_blend = brightness_transformation.magnitude_to_flux(magnitude_blend)

        fake_fluxes_parameters.append(flux_source)
        fake_fluxes_parameters.append(flux_blend)

    return fake_fluxes_parameters


def simulate_lightcurve_flux(model, pyLIMA_parameters, add_noise=True,
                             exposure_times=None):
    """ Simulate the flux of telescopes given a model and a set of parameters.
    It updates straight the telescopes object inside the given model.

    :param object model: the microlensing model you desire. More detail in microlmodels.
    :param object pyLIMA_parameters: the parameters used to simulate the flux.
    :param str red_noise_apply: to include or not red_noise

    """

    for ind, telescope in enumerate(model.event.telescopes):

        if telescope.lightcurve_flux is not None:

            theoritical_flux = \
                model.compute_the_microlensing_model(telescope, pyLIMA_parameters)[
                    'photometry']

            if add_noise:

                # if exposure_times is not None:
                #   exp_time = exposure_times[ind]

                observed_flux, err_observed_flux = \
                    brightness_transformation.noisy_observations(
                        theoritical_flux, exposure_times)

            else:

                observed_flux = theoritical_flux
                err_observed_flux = theoritical_flux ** 0.5

            telescope.lightcurve_flux['flux'] = observed_flux
            telescope.lightcurve_flux['err_flux'] = err_observed_flux

            telescope.lightcurve_magnitude = telescope.lightcurve_in_magnitude()


def simulate_astrometry(model, pyLIMA_parameters, add_noise=True):
    """
    """
    from astropy import units as unit

    for telescope in model.event.telescopes:

        if telescope.astrometry is not None:

            theoritical_model = model.compute_the_microlensing_model(telescope,
                                                                     pyLIMA_parameters)

            theoritical_flux = theoritical_model['photometry']
            theoritical_astrometry = theoritical_model['astrometry']

            if add_noise:

                observed_flux, err_observed_flux = \
                    brightness_transformation.noisy_observations(
                        theoritical_flux)

                SNR = observed_flux / err_observed_flux

                err_ra = 1 / SNR / 3600.  # assuming FWHM=1 as
                err_dec = 1 / SNR / 3600.

                obs_ra = np.random.normal(theoritical_astrometry[0], err_ra)
                obs_dec = np.random.normal(theoritical_astrometry[1], err_dec)

            else:

                obs_ra = theoritical_astrometry[0]
                err_ra = theoritical_astrometry[0] * 0.01
                obs_dec = theoritical_astrometry[1]
                err_dec = theoritical_astrometry[1] * 0.01

            telescope.astrometry['ra'] = obs_ra * unit.deg
            telescope.astrometry['err_ra'] = err_ra * unit.deg
            telescope.astrometry['dec'] = obs_dec * unit.deg
            telescope.astrometry['err_dec'] = err_dec * unit.deg
