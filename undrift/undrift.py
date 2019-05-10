import os
import sys
import cv2
import numpy
import argparse
import tifffile
from tqdm import tqdm
from skimage import filters
from scipy import ndimage as ndi
from scipy.ndimage.interpolation import geometric_transform

FARNEBACK_PARAMETER = {"levels":3, 
                        "winsize":7, 
                        "iterations":3, 
                        "poly_n":5, 
                        "poly_sigma":0.4, 
                        "flags":0}

def calc_flow(img1, img2):
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, **FARNEBACK_PARAMETER)
    u = flow[...,0]
    v = flow[...,1]
    
    return u, v

def calc_flow_stack(img_stack, smooth):
    t_dim, y_dim, x_dim = img_stack.shape
    of = numpy.zeros( (t_dim-1, 2, y_dim, x_dim), dtype=numpy.float32)

    for i, (img1, img2) in tqdm(enumerate(zip(img_stack[:-1], img_stack[1:])), 
                                total=t_dim-1,
                                desc='Calculate optical flow field'):       
        u, v = calc_flow(img1, img2)
        # u_, v_ = smooth_flow(u, v, sigma=smooth)
        of[i, 0, :, :] = u
        of[i, 1, :, :] = v
    return of
        
        

def smooth_flow(u, v, sigma):
    u_smooth = filters.gaussian(u, sigma=sigma, preserve_range=True)
    v_smooth = filters.gaussian(v, sigma=sigma, preserve_range=True)
    return u_smooth, v_smooth


def smooth_flow_(of, sigma_xy, sigma_t):
    of_smooth = numpy.zeros(of.shape, of.dtype)

    of_smooth[:, 0, :, :] = filters.gaussian(of[:, 0, :, :], sigma=(sigma_t, sigma_xy, sigma_xy), preserve_range=True)
    of_smooth[:, 1, :, :] = filters.gaussian(of[:, 1, :, :], sigma=(sigma_t, sigma_xy, sigma_xy), preserve_range=True)

    return of_smooth

def undrift(imgstack, optflow):
    t_dim, c_dim, y_dim, x_dim = imgstack.shape

    xx, yy = numpy.meshgrid(numpy.arange(x_dim), numpy.arange(y_dim))
    
    grid_visu = numpy.zeros((t_dim, y_dim, x_dim), numpy.uint8)

    grid_visu[0,  ::8,  ::8] = 255 
    grid_visu[0, 1::8, 1::8] = 255 
    grid_visu[0, 1::8,  ::8] = 255 
    grid_visu[0,  ::8, 1::8] = 255 

    result = numpy.zeros((t_dim-1, c_dim, y_dim, x_dim), imgstack.dtype)

    def shift_func_forward(xy, of):
        return (xy[1] - of[1, xy[1], xy[0]], 
                xy[0] - of[0, xy[1], xy[0]])

    def shift_func_backward(xy, of):
        return (xy[1] + of[1, xy[1], xy[0]], 
                xy[0] + of[0, xy[1], xy[0]])

    for k in tqdm(range(t_dim-1), desc='Un-drift by optical flow field'):
        for c in range(c_dim):
            img  = imgstack[k, c, :, :]
            

            coords_in_input    = shift_func_backward((xx, yy), of=optflow[:k+1, ...].sum(0))
            result[k, c, :, :] = ndi.map_coordinates(img, coords_in_input)

            coords_in_input      = shift_func_forward((xx, yy), of=optflow[:k+1, ...].sum(0))
            grid_visu[k+1, :, :] = ndi.map_coordinates(grid_visu[0, :, :], coords_in_input)

    return result, grid_visu

    


def run(input_fn, smooth_xy, smooth_t):
    print(input_fn)
    input_dir = os.path.dirname(os.path.abspath(input_fn))
    base_fn = os.path.splitext(os.path.basename(input_fn))[0]

    print("Process", input_fn)
    
    img_stack = tifffile.imread(input_fn)
    img_stack_merge = img_stack.mean(axis=1)
    
    optical_flow = calc_flow_stack(img_stack_merge, smooth=smooth_xy)

    optical_flow = smooth_flow_(optical_flow, smooth_xy, smooth_t)

    if False:
        tifffile.imsave(os.path.join(input_dir, base_fn) + "_optflow_field.tiff", optical_flow[:, None, ...], imagej=True)

    drift_undone, drift_visu = undrift(img_stack, optical_flow)

    if True:
        tifffile.imsave(os.path.join(input_dir, base_fn) + "_undrift.tiff", drift_undone, imagej=True, metadata={'Composite mode': 'composite'})

    if True:
        tifffile.imsave(os.path.join(input_dir, base_fn) + "_drift_visu.tiff", drift_visu, imagej=True)

description = \
"""
Un-drift tissue with dense optical flow 
"""

def get_args():
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('input_file', type=str, nargs="+", help="Input movie(s) with dimensions:TCYX")
    parser.add_argument('--smooth_xy', type=float, default=25, help="Sigma of spatial  smoothing of the vector field")
    parser.add_argument('--smooth_t',  type=float, default= 1, help="Sigma of temporal smoothing of the vector field")
    

    return parser.parse_args()


if __name__ == "__main__":
    # main(sys.argv[1], 25)
    args = get_args()
    for input_fn in args.input_file:
        run(input_fn, args.smooth_xy, args.smooth_t)