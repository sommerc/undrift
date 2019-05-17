import os
import sys
import cv2
import numpy
import argparse
import tifffile
from tqdm import tqdm
from skimage import filters
from skimage import feature
from scipy import ndimage as ndi
from scipy.ndimage.interpolation import geometric_transform



class DriftEstimatorOFFarneback(object):
    FARNEBACK_PARAMETER = {"levels" : 3, 
                           "winsize": 7, 
                           "iterations": 3, 
                           "poly_n" : 5, 
                           "poly_sigma" : 0.4, 
                           "flags" : 0}

    def __init__(self):
        pass

    def calc_flow(self, img1, img2):
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, **self.FARNEBACK_PARAMETER)
        u = flow[...,0]
        v = flow[...,1]
        
        return u, v

    def calc_flow_stack(self, img_stack):
        t_dim, y_dim, x_dim = img_stack.shape
        of = numpy.zeros( (t_dim-1, 2, y_dim, x_dim), dtype=numpy.float32)

        for i, (img1, img2) in tqdm(enumerate(zip(img_stack[:-1], img_stack[1:])), 
                                    total=t_dim-1,
                                    desc="{:40s}".format('  -- Calculate optical flow field')):       
            u, v = self.calc_flow(img1, img2)
            of[i, 0, :, :] = u
            of[i, 1, :, :] = v
        return of

def translational_register(imgstack):
    for i in tqdm(range(1, imgstack.shape[0]), desc="{:40s}".format("  -- Translational registration")):
        (oy, ox), e, p = feature.register_translation(imgstack[i-1].mean(0), imgstack[i].mean(0))

        t_dim, c_dim, y_dim, x_dim = imgstack.shape

        for c in range(c_dim):
            imgstack[i, c] = ndi.shift(imgstack[i, c], (oy, ox), mode="constant")

    return imgstack


def smooth_flow(of, sigma_xy, sigma_t):
    for uv in tqdm(range(2), desc="{:40s}".format("  -- Smoothing vector field ({}xy|{}t)".format(sigma_xy, sigma_t))):
        of[:, 0, :, :] = filters.gaussian(of[:, 0, :, :], sigma=(sigma_t, sigma_xy, sigma_xy), preserve_range=True, mode="nearest")
        of[:, 1, :, :] = filters.gaussian(of[:, 1, :, :], sigma=(sigma_t, sigma_xy, sigma_xy), preserve_range=True, mode="nearest")

    return of

def undrift(imgstack, optflow):
    t_dim, c_dim, y_dim, x_dim = imgstack.shape

    
    
    grid_visu = numpy.zeros((t_dim, y_dim, x_dim), numpy.uint8)

    grid_visu[0,  ::8,  ::8] = 255 
    grid_visu[0, 1::8, 1::8] = 255 
    grid_visu[0, 1::8,  ::8] = 255 
    grid_visu[0,  ::8, 1::8] = 255 

    result = numpy.zeros((t_dim, c_dim, y_dim, x_dim), imgstack.dtype)

    def shift_func_forward(xy, of):
        return (
                xy[0] - of[1, xy[0].astype(numpy.int32).clip(0,511), xy[1].astype(numpy.int32).clip(0,511)],
                xy[1] - of[0, xy[0].astype(numpy.int32).clip(0,511), xy[1].astype(numpy.int32).clip(0,511)],
                )

    def shift_func_backward(xy, of):
        return (
                xy[0] + of[1, xy[0].astype(numpy.int32).clip(0,511), xy[1].astype(numpy.int32).clip(0,511)],
                xy[1] + of[0, xy[0].astype(numpy.int32).clip(0,511), xy[1].astype(numpy.int32).clip(0,511)],
                )

    result[0, :c_dim] = imgstack[0, :c_dim]

    xxyy = numpy.meshgrid(numpy.arange(x_dim), numpy.arange(y_dim))[::-1]
    xxyy2 = numpy.meshgrid(numpy.arange(x_dim), numpy.arange(y_dim))[::-1]
    for k in tqdm(range(1, t_dim), desc="{:40s}".format('  -- Un-drift movie')):
        xxyy = shift_func_backward(xxyy, of=optflow[k-1, ...])
        xxyy2 = shift_func_forward(xxyy2, of=optflow[k-1, ...])

        for c in range(c_dim):
            img  = imgstack[k, c, :, :]
            
            
            result[k, c, :, :] = ndi.map_coordinates(img, xxyy)

            # coords_in_input      = shift_func_forward((xx, yy), of=optflow[:k+1, ...].sum(0))
            grid_visu[k, :, :] = ndi.map_coordinates(grid_visu[0, :, :], xxyy2)

    return result, grid_visu


def run(input_fn, smooth_xy, smooth_t, pre_reg, anti_flicker):
    input_dir = os.path.dirname(os.path.abspath(input_fn))
    base_fn = os.path.splitext(os.path.basename(input_fn))[0]

    print("\n# Undrift file:", base_fn)
    
    img_stack = tifffile.imread(input_fn)

    if pre_reg: img_stack = translational_register(img_stack)

    img_stack_merge = img_stack.mean(axis=1)

    

    if True:
        drift_corrector = DriftEstimatorOFFarneback()
        optical_flow = drift_corrector.calc_flow_stack(img_stack_merge)

        optical_flow = smooth_flow(optical_flow, smooth_xy, smooth_t)
        tifffile.imsave(os.path.join(input_dir, base_fn) + "_optflow_field.tiff", optical_flow[:, None, ...], imagej=True)
    else:
        optical_flow = tifffile.imread(os.path.join(input_dir, base_fn) + "_optflow_field.tiff")

    drift_undone, drift_visu = undrift(img_stack, optical_flow)

    if anti_flicker: drift_undone = translational_register(drift_undone)


    print("  -- Saving output files")
    if True:
        tifffile.imsave(os.path.join(input_dir, base_fn) + "_undrift.tiff", drift_undone[:, None, ...], imagej=True, metadata={'Composite mode': 'composite'})

    if True:
        tifffile.imsave(os.path.join(input_dir, base_fn) + "_drift_visu.tiff", drift_visu[:, None, ...], imagej=True)

description = \
"""
Un-drift tissue with dense optical flow 
"""

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('input_file',  type=str, nargs="+", help="Input 2D-movie(s) with dimensions: TCYX")
    parser.add_argument('--smooth_xy', type=float, default=31, help="Sigma of spatial Gaussian smoothing of the vector field (31)")
    parser.add_argument('--smooth_t',  type=float, default= 1, help="Sigma of temporal Gaussian smoothing of the vector field (1)")
    parser.add_argument('--pre_reg',   type=str2bool, nargs='?', const=True, default=False,   help="Apply translational pre-registration (False)")
    parser.add_argument('--anti_flicker', type=str2bool, nargs='?', const=True, default=False, help="Apply translational post-registration (False)")
    
    return parser.parse_args()


def main():
    # main(sys.argv[1], 25)
    args = get_args()
    for input_fn in args.input_file:
        run(input_fn, args.smooth_xy, args.smooth_t, args.pre_reg, args.anti_flicker)

if __name__ == "__main__":
    main()