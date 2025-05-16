import os
import glob
import math
import numpy as np
import pandas as pd
import skimage as ski
import cv2
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import napari
import skimage as ski
import cv2
import scipy.ndimage as ndi
import seaborn as sns
import scipy as sp
import io
from PIL import Image
import os
import dask

from skimage import io, filters, measure, transform, exposure
from skimage.transform import resize
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm


def get_props(img, labels):
    props = measure.regionprops(labels, intensity_image=img)
    return np.array([p.mean_intensity for p in props])


def save_images_as_avi(image_stack, output_file, fps=30, cmap='cyan'):
    """
    Save a stack of images as an .avi video file.

    Parameters:
    - image_stack: numpy array of shape (num_frames, height, width, channels)
    - output_file: path to the output .avi file
    - fps: frames per second for the output video
    """
    out_stack = (image_stack.astype(np.double))/np.percentile(image_stack, 99.9)
    out_stack = np.clip(out_stack, 0, 1)

    if cmap=='viridis':
        out_stack = plt.get_cmap('viridis')(out_stack)[:,:,:,0:3] * 255
    elif cmap=='red':
        out_stack = np.stack([out_stack, out_stack, out_stack], axis=-1)
        out_stack = out_stack * 255
        out_stack[:,:,:,0] = 0
        out_stack[:,:,:,1] = 0
    elif cmap=='cyan':
        out_stack = np.stack([out_stack, out_stack, out_stack], axis=-1)
        out_stack = out_stack * 255
        out_stack[:,:,:,2] = 0
        
    out_stack = out_stack.astype(np.uint8)
    # Get the dimensions of the images
    num_frames, height, width, channels = out_stack.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for i in range(num_frames):
        # Write each frame to the video file
        out.write(out_stack[i])

    # Release the VideoWriter object
    out.release()

def straighten_image(image, path_points, width=50, napari_order=True, num_points=None):
    """
    Straighten a curved path in an image.
    
    :param image: Input image as a NumPy array.
    :param path_points: List of (x, y) tuples defining the curved path.
    :param width: Width of the straightened output.
    :return: Straightened image.
    """
    # Find the total length of the path in pixel space
    if napari_order:
        path_points = path_points[:,::-1]
    path_length = 0
    for i in range(len(path_points) - 1):
        p1, p2 = path_points[i], path_points[i + 1]
        path_length += np.linalg.norm(np.array(p2) - np.array(p1))

    # Fit a spline to the path points
    path_points = np.array(path_points)
    tck, _ = sp.interpolate.splprep(path_points.T, s=0, k=1)
    if num_points is None:
        num_points = int(np.round(path_length))  # Number of points to interpolate
    u = np.linspace(0, 1, num_points)
    x_new, y_new = sp.interpolate.splev(u, tck)
    path_spline = np.column_stack((x_new, y_new))
    
    # Initialize output image
    straightened = []
    
    tangents = []
    for i in range(len(path_spline) - 1):
        # Get tangent vector
        p1, p2 = path_spline[i], path_spline[i + 1]
        tangent = p2 - p1
        tangent = tangent / np.linalg.norm(tangent)
        
        
        # Get normal vector
        normal = np.array([-tangent[1], tangent[0]])
        tangents.append(normal)
        
        # Extract slice
        slice_coords = [p1 + normal * (-width // 2 + j) for j in range(width)]
        slice_coords = np.clip(slice_coords, 0, np.array(image.shape[1::-1]) - 1).astype(int)

        
        slice_pixels = image[slice_coords[:, 1], slice_coords[:, 0]]
        straightened.append(slice_pixels)
    
    # Stack slices into a straightened image
    straightened = np.array(straightened).T
    tangents = np.array(tangents)
    vectors = np.stack([path_spline[0:-1,::-1], tangents[:,::-1]*100.0], axis=1)
    return straightened



def add_colored_scale_bar(bgr, color='red', width=40, height_ratio=0.3,
                          min_val=0, max_val=45000, label=True, channel_type=None):
    """
    Adds a vertical intensity scale bar in the same color as the image channel.
    If channel_type is provided (e.g., 'TRITC_Deriv'), and min_val/max_val are default,
    it overrides them with appropriate values for derivative data.
    """
    # Optional overrides for derivative kymographs
    if channel_type == 'TRITC_Deriv' and min_val == 0 and max_val == 45000:
        color = 'red'
        min_val = -10000
        max_val = 10000
    elif channel_type == 'YFP_Deriv' and min_val == 0 and max_val == 45000:
        color = 'green'
        min_val = -10000
        max_val = 10000
    elif channel_type == 'Ratio_Deriv' and min_val == 0 and max_val == 45000:
        color = 'blue'
        min_val = -1.0
        max_val = 1.0

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2
    max_label = str(int(max_val))
    (label_width, _) = cv2.getTextSize(max_label, font, font_scale, thickness)[0]

    H, W, _ = bgr.shape
    bar_height = int(H * height_ratio)
    bar_top = int((H - bar_height) / 2)
    bar_left = W - label_width

    gradient = np.linspace(255, 0, bar_height).astype(np.uint8).reshape(-1, 1)
    bar = np.repeat(gradient, width, axis=1)

    bar_bgr = np.zeros((bar_height, width, 3), dtype=np.uint8)
    if color == 'red':
        bar_bgr[..., 2] = bar
    elif color == 'green':
        bar_bgr[..., 1] = bar
    elif color == 'blue':
        bar_bgr[..., 0] = bar
    else:
        bar_bgr[...] = bar[:, :, None]

    bgr[bar_top:bar_top + bar_height, bar_left:bar_left + width] = bar_bgr

    if label:
        color_text = (255, 255, 255)
        cv2.putText(bgr, f'{int(max_val)}', (bar_left - 40, bar_top - 10),
                    font, font_scale, color_text, thickness, cv2.LINE_AA)
        cv2.putText(bgr, f'{int(min_val)}', (bar_left - 40, bar_top + bar_height),
                    font, font_scale, color_text, thickness, cv2.LINE_AA)

    return bgr

def save_channel_avi_with_timestamp(image_stack, output_path, fps=5, color='gray', magnification=1.25):
            """
            Save a 3D image stack (T, Y, X) as an AVI with timestamp in hours.
            """
            # color_map = {
            #         'red': cm.Reds,
            #         'green': cm.Greens,
            #         'blue': cm.Blues,
            #         'gray': cm.gray
            #     }.get(cmap, cm.gray)

            T, H, W = image_stack.shape
            video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (W, H))
            

            for i in range(T):
                    frame = image_stack[i]
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        # It's already RGB
                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        # It's grayscale
                        norm = (frame - np.min(frame)) / (np.max(frame) - np.min(frame) + 1e-8)
                        gray_uint8 = (norm * 255).astype(np.uint8)

                        bgr = np.zeros((gray_uint8.shape[0], gray_uint8.shape[1], 3), dtype=np.uint8)

                        if color == 'red':
                            bgr[..., 2] = gray_uint8  # Red channel
                        elif color == 'green':
                            bgr[..., 1] = gray_uint8  # Green channel
                        elif color == 'blue':
                            bgr[..., 0] = gray_uint8  # Blue channel
                        else:
                            # Default to grayscale
                            bgr = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)

                    #norm = (frame - np.min(frame)) / (np.max(frame) - np.min(frame) + 1e-8)
                    #colored = (color_map(norm)[..., :3] * 255).astype(np.uint8)
                    #bg_mask = norm <= 0.01
                    #colored[bg_mask] = [0, 0, 0]  # Set background to black (RGB)

                    #bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

                    time_in_hours = i * 1  # each frame is 1 hours apart
                    timestamp = f"Time: {time_in_hours:.2f} h"

                    cv2.putText(bgr, timestamp, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                3, (255,255,255), 3, cv2.LINE_AA)
                    
                    bgr = add_colored_scale_bar(bgr, color=color, min_val=0, max_val=45000) #add intensity scale bar
                    # Draw scale bar
                    scale_bar_width = 500.0 # in microns
                    X2 = W - 30
                    Y1 = H - 50
                    Y2 = H - 50
                    X1 = int(X2 - scale_bar_width/(7.0632 / magnification))
                    rect_width = 30
                    cv2.rectangle(bgr, (X1, Y1 - rect_width // 2), (X2, Y2 + rect_width // 2), (255, 255, 255), -1)



                    video_writer.write(bgr)

            video_writer.release()

