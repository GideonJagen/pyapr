import pyapr
import numpy as np


def get_apr_interactive(image, dtype='short', rel_error=0.1, gradient_smoothing=2, verbose=True, params=None):

    # check that the image array is c-contiguous
    if not image.flags['C_CONTIGUOUS']:
        print('WARNING: \'image\' argument given to get_apr_interactive is not C-contiguous \n'
              'input image has been replaced with a C-contiguous copy of itself')
        image = np.ascontiguousarray(image)

    # Initialize objects
    io_int = pyapr.filegui.InteractiveIO()
    apr = pyapr.APR()
    if params is None:
        par = pyapr.APRParameters()

        # Set some parameters
        par.auto_parameters = False
        par.rel_error = rel_error
        par.gradient_smoothing = gradient_smoothing
    else:
        par = params

    if dtype in ('float', 'float32'):
        parts = pyapr.FloatParticles()
        converter = pyapr.converter.FloatConverter()
    elif dtype in ('short', 'uint16'):
        parts = pyapr.ShortParticles()
        converter = pyapr.converter.ShortConverter()
    # elif dtype in {'byte', 'uint8'}:  # currently not working
    #     parts = pyapr.ByteParticles()
    #     converter = pyapr.converter.ByteConverter()
    else:
        errstr = 'get_apr_interactive argument dtype must be one of (float, float32, short, uint16), but {} was given'.format(dtype)
        raise Exception(errstr)

    converter.set_parameters(par)
    converter.set_verbose(verbose)

    # launch interactive APR converter
    io_int.interactive_apr(converter, apr, image)

    if verbose:
        print("Total number of particles: {} \n".format(apr.total_number_particles()))
        cr = image.size/apr.total_number_particles()
        print("Compuational Ratio: {:7.2f}".format(cr))
        print("Sampling particles ... \n")

    # sample particles
    parts.sample_image(apr, image)

    return apr, parts


def find_parameters_interactive(image, dtype='short', rel_error=0.1, gradient_smoothing=0, verbose=True, params=None):
    # Initialize objects
    io_int = pyapr.filegui.InteractiveIO()
    apr = pyapr.APR()
    if params is None:
        par = pyapr.APRParameters()

        # Set some parameters
        par.auto_parameters = False
        par.rel_error = rel_error
        par.gradient_smoothing = gradient_smoothing
    else:
        par = params

    if dtype in ('float', 'float32'):
        parts = pyapr.FloatParticles()
        converter = pyapr.converter.FloatConverter()
    elif dtype in ('short', 'uint16'):
        parts = pyapr.ShortParticles()
        converter = pyapr.converter.ShortConverter()
    # elif dtype in {'byte', 'uint8'}:  # currently not working
    #     parts = pyapr.ByteParticles()
    #     converter = pyapr.converter.ByteConverter()
    else:
        errstr = 'get_apr_interactive argument dtype must be one of (float, float32, short, uint16), but {} was given'.format(dtype)
        raise Exception(errstr)

    converter.set_parameters(par)
    converter.set_verbose(verbose)

    # launch interactive APR converter
    par = io_int.find_parameters_interactive(converter, apr, image)

    if verbose:
        print("---------------------------------")
        print("Using the following parameters:")
        print("grad_th = {}, sigma_th = {}, Ip_th = {}".format(par.grad_th, par.sigma_th, par.Ip_th))
        print("---------------------------------")

    return par
