import os
import pyapr
from skimage.external import tifffile


def main():

    # Read in an image
    io_int = pyapr.filegui.InteractiveIO()
    fpath = io_int.get_tiff_file_name()  # get image file path from gui (data type must be float32 or uint16)

    # Specify the z-range to be used to set the parameters
    z_start = 256
    z_end = z_start + 512

    # Read slice range into numpy array
    with tifffile.TiffFile(fpath) as tif:
        img = tif.asarray(key=slice(z_start, z_end))

    # Initialize and set some APRParameters (only Ip_th, grad_th and sigma_th are set interactively)
    par = pyapr.APRParameters()
    par.auto_parameters = False
    par.rel_error = 0.1
    par.gradient_smoothing = 5

    # Interactively set the threshold parameters using the partial image
    par = pyapr.converter.find_parameters_interactive(img, dtype=img.dtype, params=par, verbose=True)

    del img  # Parameters found, we don't need the partial image anymore

    # par.input_dir + par.input_image_name must be the path to the image file
    par.input_dir = ''
    par.input_image_name = fpath

    # Initialize the by-block converter
    converter = pyapr.converter.ShortConverterBatch()
    converter.set_parameters(par)
    converter.set_verbose(True)

    # Parameters controlling the memory usage
    converter.set_block_size(256)   # number of z-slices to process in each block during APR conversion
    converter.set_ghost_size(32)    # number of ghost slices to use on each side of the blocks
    block_size_sampling = 256       # block size for sampling of particle intensities
    ghost_size_sampling = 128       # ghost size for sampling of particle intensities

    # Compute the APR
    apr = pyapr.APR()
    success = converter.get_apr(apr)

    if success:
        cr = apr.computational_ratio()
        print('APR Conversion successful! Computational ratio (#pixels / #particles) = {}'.format(cr))

        print('Sampling particle intensity values')
        parts = pyapr.ShortParticles()
        parts.sample_image_blocked(apr, fpath, block_size_sampling, ghost_size_sampling)
        print('Done!')

        # View the result in the by-slice viewer
        pyapr.viewer.parts_viewer(apr, parts)

        # Write the resulting APR to file
        print("Writing APR to file ... \n")
        fpath_apr = io_int.save_apr_file_name()  # get path through gui
        pyapr.io.write(fpath_apr, apr, parts)

        if fpath_apr:
            # Display the size of the file
            file_sz = os.path.getsize(fpath_apr)
            print("APR File Size: {:7.2f} MB \n".format(file_sz * 1e-6))

            # Compute compression ratio
            mcr = os.path.getsize(fpath) / file_sz
            print("Memory Compression Ratio: {:7.2f}".format(mcr))

    else:
        print('Something went wrong...')


if __name__ == '__main__':
    main()
