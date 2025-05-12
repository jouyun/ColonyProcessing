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
from pystackreg import StackReg
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

def process_color_channel(idf, type, resolution=7.88955):
    df = idf.copy()
    if type == 'YFP':
        color_scheme = 'viridis'
        min_z, max_z = 0, 45000 #adjust as needed
    elif type == 'TRITC':
        color_scheme = 'Reds_r'
        min_z, max_z = 0, 45000 #adjust as needed
    elif type == 'Ratio':
        color_scheme = 'rocket'
        min_z, max_z = 0.0, 1 #adjust as needed
        df.iloc[:, 1:] = np.log10(df.iloc[:, 1:] + 1)

    elif type == 'YFP_Deriv':
        color_scheme = 'viridis'
        min_z, max_z = 0, 45000 #adjust as needed
    elif type == 'TRITC_Deriv':
        color_scheme = 'Reds_r'
        min_z, max_z = 0, 45000 #adjust as needed
    elif type == 'Ratio_Deriv':
        color_scheme = 'rocket'
        min_z, max_z = -1, 1 #adjust as needed
    else:
        raise ValueError(f"Unrecognized channel type: {type}")
    
    df = df.transpose().reset_index()
    df.columns = ['Y'] + list(df.columns[1:])
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Save the read in data to an Excel file in the 'Data read in' directory
    #df.to_excel(os.path.join(data_dir, os.path.basename(file_path) + ".xlsx"), index=False)
    
    if not pd.api.types.is_numeric_dtype(df['Y']):
        raise Exception("'Y' column is not numeric.")
    
    #print(df['x'].max())
    #resolution = 7.0632 / magnification
    df_long = df.melt(id_vars=['Y'], var_name='X', value_name='Z')
    df_long['X'] = df_long['X'] * 0.001*resolution - 0.001*resolution
    
    max_x = df_long['X'].max()
    #df_long = df_long[df_long['X'] <= 0.75 * max_x]
    
    time_interval = 1.16667  # hours per frame
    df_long['Time_hr'] = df_long['Y'] * time_interval

    pivoted = df_long.pivot(index="Time_hr", columns="X", values="Z")
    #end_time = 120
    #df_long = df_long[(df_long['Y'] >= 8) & (df_long['Y'] <= end_time)]
    
    plt.figure(figsize=(10, 8))

    #pivoted = df_long.pivot(index="Y", columns="X", values="Z")
    #mask = np.isnan(pivoted)
    mask = pivoted.isna() | (pivoted == 0) 
    cmap = plt.get_cmap(color_scheme).copy()
    cmap.set_bad(color='gray')

    #fig = sns.heatmap(pivoted, cmap=cmap, mask=mask, vmin=min_z, vmax=max_z, xticklabels=100)
    fig = sns.heatmap(pivoted, cmap=cmap, mask=mask, vmin=min_z, vmax=max_z, xticklabels=100, yticklabels=True)

    # Formatting
    ax = fig
    ax.invert_yaxis()
    ax.set_xlabel("Radius (mm)", fontsize=30)
    ax.set_ylabel("Time (h)", fontsize=30)

    #xrng = (np.arange(0,4,1))
    max_rng = np.floor(max_x).astype(int) + 1
    xrng = (np.arange(0,max_rng,1))
    ax.set_xticks((xrng * 1/(0.001*resolution)).astype(int))
    ax.set_xticklabels(xrng)
    ax.tick_params(axis='x', labelsize=20)

    max_y = pivoted.index.max()
    yrng = np.arange(2,max_y, 10)
    ax.set_yticks((yrng).astype(int))
    ax.set_yticklabels(yrng+8)
    ax.tick_params(axis='y', labelsize=20)

    ax.set_title(f"{type} Intensity", fontsize=30)

    fig = ax.get_figure()
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape((height, width, 4))  # RGBA data

    # If you only need RGB (3 channels), slice off the alpha channel:
    buf = buf[..., :3]

    # Convert to a Pillow Image
    img = Image.fromarray(buf)

    plt.close(fig)  # Close if you don't need the figure anymore

    return img

def create_montage(images, rows, cols):
    widths, heights = zip(*(i.size for i in images))
    total_width = cols * max(widths)
    total_height = rows * max(heights)
    
    montage = Image.new('RGB', (total_width, total_height))
    
    for i, img in enumerate(images):
        x = i % cols * max(widths)
        y = i // cols * max(heights)
        montage.paste(img, (x, y))
    
    return montage

def merge_montages(dir_path):
    # Determine the appropriate path separator based on the operating system
    if os.name == 'nt':  # Windows
        path_separator = '\\'
    else:  # Linux or Mac
        path_separator = '/'
    montages = glob.glob(dir_path +'/*.png')
    montages.sort()

    img_list = []
    for current_montage in montages:
        if 'All' in current_montage:
            continue
        well = current_montage.split(path_separator)[-1].split('_')[0]
        roi = current_montage.split(path_separator)[-1].split('_')[1].split('.')[0]
        img = Image.open(current_montage)
        draw = ImageDraw.Draw(img)
        text = well + '_' + roi
        position = (10, 10)
        font = ImageFont.truetype("arial", 70)
        draw.text(position, text, font=font, fill='black')
        img_list.append(img)
    #montaged = create_montage(img_list, int(len(img_list)/5), 5)
    montaged = create_montage(img_list, math.ceil(len(img_list)/5), 5)
    montaged.save(dir_path + '/All.png')

# Assuming these helpers exist:
# from your_module import process_color_channel, create_montage, straighten_image
def merge_derivative_montages(dir_path):
    """
    Merge derivative kymograph PNGs (e.g., B4_DerivMontage_0.png) into a labeled All_DerivMontages.png.
    """
    montage_paths = glob.glob(os.path.join(dir_path, '*_DerivMontage_*.png'))
    montage_paths.sort()
    img_list = []
    for path in montage_paths:
        filename = os.path.basename(path).replace('.png', '')
        parts = filename.split('_')  # e.g., ['B4', 'DerivMontage', '0']
        if len(parts) < 3:
            continue
        well, angle = parts[0], parts[-1]

        img = Image.open(path)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial", 50)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), f"{well}_Δ_Angle{angle}", fill='black', font=font)
        img_list.append(img)

    if img_list:
        montage = create_montage(img_list, math.ceil(len(img_list) / 5), 5)
        montage.save(os.path.join(dir_path, 'All_DerivMontages.png'))
        print("Saved All_DerivMontages.png")
    else:
        print("No derivative montages found to merge.")

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

def make_derivative_kymograph_montages_from_csv(well, csv_dir, output_dir, resolution=7.0632 / 1.25):
    """
    Creates 1×3 TIFF montages for each angle using TRITC_Deriv, YFP_Deriv, and Ratio_Deriv CSVs.
    """
    channel_names = ['TRITC_Deriv', 'YFP_Deriv', 'Ratio_Deriv']
    for idx in range(4):  # Four angles
        images = []

        for ch in channel_names:
            path = os.path.join(csv_dir, f"{well}_{ch}_{idx}.csv")
            if not os.path.exists(path):
                print(f"Missing: {path}")
                continue
            df = pd.read_csv(path)
            df_long = df.melt(id_vars=['x'], var_name='Y', value_name='Z')
            df_long['Y'] = pd.to_numeric(df_long['Y'], errors='coerce')
            df_long['Time_hr'] = df_long['Y']
            pivoted = df_long.pivot(index="Time_hr", columns="x", values="Z")
            mask = pivoted.isna() | (pivoted == 0)

            if ch == 'TRITC_Deriv':
                cmap = plt.get_cmap('Reds_r').copy()
                vmin, vmax = -10000, 10000
            elif ch == 'YFP_Deriv':
                cmap = plt.get_cmap('viridis').copy()
                vmin, vmax = -10000, 10000
            elif ch == 'Ratio_Deriv':
                cmap = plt.get_cmap('seismic').copy()
                vmin, vmax = -1.0, 1.0
            else:
                cmap = plt.get_cmap('gray').copy()
                vmin, vmax = None, None

            cmap.set_bad(color='gray')

            fig = sns.heatmap(pivoted, cmap=cmap, mask=mask, center=0, vmin=vmin, vmax=vmax, cbar=True)

            ax = fig
            ax.invert_yaxis()
            ax.set_title(ch.replace("_Deriv", ""), fontsize=20)
            ax.axis('off')

            fig = ax.get_figure()
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            images.append(Image.fromarray(buf[..., :3]))
            plt.close(fig)
        if len(images) == 3:
            montage = create_montage(images, 1, 3)
            draw = ImageDraw.Draw(montage)
            try:
                font = ImageFont.truetype("arial", 50)
            except:
                font = ImageFont.load_default()

            draw.text((10, 10), f"{well}_Angle{idx}", font=font, fill='black')
            out_path = os.path.join(output_dir, f"{well}_DerivMontage_{idx}.png")
            montage.save(out_path)
            print(f"✅ Saved: {out_path}")


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
