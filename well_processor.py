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
        min_z, max_z = 0, 55000
    elif type == 'TRITC':
        color_scheme = 'turbo'
        min_z, max_z = 0, 25000
    else:
        color_scheme = 'rocket'
        min_z, max_z = 0.0, 1.25
        df.iloc[:, 1:] = np.log10(df.iloc[:, 1:] + 1)
    
    df = df.transpose().reset_index()
    df.columns = ['Y'] + list(df.columns[1:])
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Save the read in data to an Excel file in the 'Data read in' directory
    #df.to_excel(os.path.join(data_dir, os.path.basename(file_path) + ".xlsx"), index=False)
    
    if not pd.api.types.is_numeric_dtype(df['Y']):
        raise Exception("'Y' column is not numeric.")
    
    #print(df['x'].max())
    #7.0632 / magnification
    df_long = df.melt(id_vars=['Y'], var_name='X', value_name='Z')
    df_long['X'] = df_long['X'] * 0.001*resolution - 0.001*resolution
    
    max_x = df_long['X'].max()
    #df_long = df_long[df_long['X'] <= 0.75 * max_x]
    
    end_time = 80
    df_long = df_long[(df_long['Y'] >= 8) & (df_long['Y'] <= end_time)]
    
    # R code appears to be ignoring the hardcoded min_z and max_z values
    #min_z, max_z = df_long['Z'].min(), df_long['Z'].max()
    
    plt.figure(figsize=(10, 8))

    pivoted = df_long.pivot(index="Y", columns="X", values="Z")
    mask = np.isnan(pivoted)
    fig = sns.heatmap(pivoted, cmap=color_scheme, mask=mask, vmin=min_z, vmax=max_z, xticklabels=100)

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

    yrng = np.arange(2,50, 10)
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

                    time_in_hours = i * 1  # each frame is 4 hours apart
                    timestamp = f"Time: {time_in_hours:.2f} h"

                    cv2.putText(bgr, timestamp, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (255,255,255), 2, cv2.LINE_AA)
                    
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


class WellProcessor:
    """
    A class to process microscopy images from a particular well directory.
    It handles the loading, registration, masking, and generation of intensity
    plots (kymographs) for TRITC and YFP channels.
    """
    def __init__(self, well, path, time_clip=2, magnification=4.0):
        """
        Initialize the processor with a given well ID and file path.

        Parameters
        ----------
        well : str
            The well identifier (e.g., 'A1').
        path : str
            The path to the directory containing the .tif images.
        """
        self.well = well
        self.path = path
        self.time_clip = time_clip

        # Files present in the directory
        self.files = glob.glob(os.path.join(path, '*.tif'))
        self.files.sort()

        # Attributes that will be set during processing
        self.tritc = None
        self.yfp = None
        self.tritc_registered = None
        self.yfp_registered = None
        self.masks = None
        self.tritc_cleared = None
        self.yfp_cleared = None
        self.output_img = None

        # Kymographs
        self.tritc_kymographs = []
        self.yfp_kymographs = []

        # DataFrames
        self.tritc_tdfs = []
        self.yfp_tdfs = []
        self.ratio_tdfs = []

        # Geometric parameters of the region
        self.x = None
        self.y = None
        self.r = None
        self.magnification = magnification
        self.resolution = 7.0632 / magnification # in microns per pixel

        # Directories
        self.montage_dir = None
        self.csv_dir = None
        self.projection_dir = None
        self.segmentation_dir = None
        self.cleared_dir = None

    def crop_and_subtract_background(self, img):
        """
        Crop the input stack to remove unneeded edges and then subtract background
        using the 10th percentile for each slice.

        Parameters
        ----------
        img : np.ndarray
            The image stack to be cropped and background-subtracted (dimensions: Z,Y,X).

        Returns
        -------
        np.ndarray
            The processed image stack (uint16).
        """
        # Crop: [Z-slices, Y, X] => use whatever cropping is appropriate
        img = img[self.time_clip:110, 0:1992, 0:1992]
        fimg = img.astype(np.float32)

        # Subtract background
        backgrounds = np.percentile(fimg, 10, axis=(1, 2))
        fimg = np.clip(fimg - backgrounds[:, None, None], 0, None)
        return fimg.astype(np.uint16)

    def get_well_images(self):
        """
        Load TRITC and YFP images from the given well directory and
        apply background subtraction to each stack.
        """
        # Load TRITC
        tritc_paths = glob.glob(os.path.join(self.path, f'{self.well}_*TRITC*.tif'))
        tritc_paths.sort()
        tritc_stack = np.array([io.imread(f) for f in tritc_paths])
        self.tritc = self.crop_and_subtract_background(tritc_stack)

        # Load YFP
        yfp_paths = glob.glob(os.path.join(self.path, f'{self.well}_*YFP*.tif'))
        yfp_paths.sort()
        yfp_stack = np.array([io.imread(f) for f in yfp_paths])
        self.yfp = self.crop_and_subtract_background(yfp_stack)

    def register_images(self):
        """
        Register YFP stack to itself using 'previous' reference, then
        apply the same transform to TRITC.
        """
        sr = StackReg(StackReg.TRANSLATION)
        self.yfp_registered = sr.register_transform_stack(
            self.yfp,
            reference='previous',
            verbose=True
        )
        self.tritc_registered = sr.transform_stack(self.tritc)

    def get_center_and_feret_slice(self, mask):
        """
        Given a single 2D mask, compute the centroid and maximum Feret diameter.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary 2D mask.

        Returns
        -------
        centroid_y : float
            The Y-coordinate of the largest region's centroid.
        centroid_x : float
            The X-coordinate of the largest region's centroid.
        feret_diameter : float
            The maximum Feret diameter of the largest region.
        """
        labels = ndi.label(mask)[0]
        props = measure.regionprops_table(
            labels,
            properties=('centroid', 'orientation', 'area', 
                        'major_axis_length', 'minor_axis_length', 
                        'feret_diameter_max')
        )
        df = pd.DataFrame(props)
        df = df.sort_values(by='area', ascending=False)
        return (
            df.iloc[0]['centroid-0'],
            df.iloc[0]['centroid-1'],
            df.iloc[0]['feret_diameter_max']
        )

    def get_center_and_radius_stack(self, masks):
        """
        Compute center (x, y) and approximate radius across multiple slices.

        Parameters
        ----------
        masks : np.ndarray
            3D binary mask stack (Z,Y,X).

        Returns
        -------
        center_x : float
        center_y : float
        radius : int
        """
        num_slices = masks.shape[0]

        # Sample two slices to get orientation, e.g. slice 5 and 75% of total
        slice_1 = 5
        slice_2 = int(num_slices * 0.75)

        y1, x1, d1 = self.get_center_and_feret_slice(masks[slice_1])
        y2, x2, d2 = self.get_center_and_feret_slice(masks[slice_2])

        # For demonstration, we only use one radius—adjust as needed
        chosen_d = max(d1, d2)
        return x1, y1, int(chosen_d / 2)
    
    def get_masks(self):
        """
        Build a simple mask for each registered TRITC slice by thresholding and 
        applying a Gaussian filter. Then apply masks to TRITC and YFP data.
        """
        thresholds = np.array([
            filters.threshold_triangle(slice_) for slice_ in self.tritc_registered
        ])
        masks = [
            filters.gaussian(slice_, sigma=10) > thr 
            for slice_, thr in zip(self.tritc_registered, thresholds)
        ]
        self.masks = np.array(masks)

        # Force the first few slices to match slice 4
        self.masks[0:4] = self.masks[4]

        self.tritc_cleared = self.tritc_registered * self.masks
        self.yfp_cleared = self.yfp_registered * self.masks

                # Combine TRITC and YFP for timelapse
        combined = np.stack([self.tritc_cleared, self.yfp_cleared, self.yfp_cleared/self.tritc_cleared], axis=1)
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        out_path = os.path.join(self.cleared_dir, f'{self.well}.tif')
        io.imsave(
            out_path,
            combined.astype(np.single),
            imagej=True,
            resolution = (1.0/self.resolution/.001, 1.0/self.resolution/.001),
            metadata={
                'unit': 'mm',
                'axes': 'TCYX',
                'mode': 'composite',
                'min': float(np.min(combined)),
                'max': float(np.max(combined))
            }
        )

        # #Make .avis
        # save_images_as_avi(self.tritc_cleared, os.path.join(self.cleared_dir, f'{self.well}_TRITC.avi'), fps=20, cmap='red')
        # save_images_as_avi(self.yfp_cleared, os.path.join(self.cleared_dir, f'{self.well}_YFP.avi'), fps=20, cmap='cyan')
        ratio_img = self.yfp_cleared / self.tritc_cleared
        ratio_img = np.nan_to_num(ratio_img, nan=0.0, posinf=0.0, neginf=0.0)
        # save_images_as_avi(ratio_img, os.path.join(self.cleared_dir, f'{self.well}_Ratio.avi'), fps=30, cmap='viridis')

        save_channel_avi_with_timestamp(
            self.tritc_cleared,  # shape (T, Y, X)
            os.path.join(self.cleared_dir, f'{self.well}_TRITC.avi'),
            fps=5,
            color='red',
            magnification=self.magnification
        )

        save_channel_avi_with_timestamp(
            self.yfp_cleared,  # shape (T, Y, X)
            os.path.join(self.cleared_dir, f'{self.well}_YFP.avi'),
            fps=5,
            color='green',
            magnification= self.magnification
        )

        save_channel_avi_with_timestamp(
            ratio_img,  # shape (T, Y, X)
            os.path.join(self.cleared_dir, f'{self.well}_ratio.avi'),
            fps=5,
            color='blue',
            magnification= self.magnification
        )


        # Make 'derivative' images
        combined = np.stack([self.tritc_cleared, self.yfp_cleared, self.yfp_cleared/self.tritc_cleared], axis=1)
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        combined = np.diff(combined, axis=0)
        out_path = os.path.join(self.cleared_dir, f'{self.well}_derivative.tif')
        io.imsave(
            out_path,
            combined.astype(np.single),
            imagej=True,
            resolution = (1.0/self.resolution/.001, 1.0/self.resolution/.001),
            metadata={
                'unit': 'mm',
                'axes': 'TCYX',
                'mode': 'composite',
                'min': float(np.min(combined)),
                'max': float(np.max(combined))
            }
        )

        

    def get_prop_stack(self, labels, img_stack):
        """
        Compute regionprops-based statistics for each slice in the stack.

        Parameters
        ----------
        labels : np.ndarray
            2D integer label image (same shape as each slice).
        img_stack : np.ndarray
            3D image stack (Z,Y,X).

        Returns
        -------
        np.ndarray
            A 2D array of mean intensities for each label across slices.
        """
        # OLD WAY
        rtn_vals = []
        for img in img_stack:
            props = measure.regionprops(labels, intensity_image=img)
            intensities = np.array([p.mean_intensity for p in props])
            rtn_vals.append(intensities)
        return np.array(rtn_vals)
    
        # process = [dask.delayed(get_props)(i, labels) for i in img_stack]
        # rslt = np.array(dask.compute(*process))
        # return rslt

    def get_kymographs(self, display=False, viewer=None):
        """
        Generate multiple kymographs along various angles plus a radial "shell" kymograph.
        Optionally add shapes to a napari-like viewer for inspection.
        """
        lines = []
        self.tritc_kymographs = []
        self.yfp_kymographs = []

        # Create an output overlay image from the 5th slice of TRITC (index=5).
        self.output_img = exposure.rescale_intensity(
            self.tritc_cleared[5].copy(), out_range=(0, 255)
        ).astype(np.uint8)

        # Standard lines at specified angles
        angles = [-45, 65, 90, -10]
        for angle_ in angles:
            pt0 = [self.y, self.x]
            pt00 = [
                self.y + (self.r/2) * math.sin(math.radians(angle_)),
                self.x + (self.r/2) * math.cos(math.radians(angle_))
            ]
            pt1 = [
                self.y + self.r * math.sin(math.radians(angle_)),
                self.x + self.r * math.cos(math.radians(angle_))
            ]

            lines.append([pt0, pt1])

            yfp_kymo = np.array([
                straighten_image(slice_, np.array([pt0, pt00, pt1]),
                                 num_points=int(self.r), width=27) 
                for slice_ in self.yfp_cleared
            ]).mean(axis=1)

            tritc_kymo = np.array([
                straighten_image(slice_, np.array([pt0, pt00, pt1]),
                                 num_points=int(self.r), width=27) 
                for slice_ in self.tritc_cleared
            ]).mean(axis=1)

            self.yfp_kymographs.append(yfp_kymo)
            self.tritc_kymographs.append(tritc_kymo)

            # Draw line on output image for reference
            cv2.line(
                self.output_img,
                (int(pt0[1]), int(pt0[0])),
                (int(pt1[1]), int(pt1[0])),
                (255, 255, 255),
                2
            )

        include_shell_kymo = False
        if include_shell_kymo:
            # Construct distance-based shell kymograph
            xgrid, ygrid = np.meshgrid(
                np.arange(self.tritc_cleared.shape[2]),
                np.arange(self.tritc_cleared.shape[1])
            )
            distances = np.sqrt((xgrid - self.x)**2 + (ygrid - self.y)**2)
            distances = distances.astype(np.uint16)

            # Mask distances > r and ensure no zeros
            distances = distances * (distances < self.r) + 1

            tritc_kymo_shell = self.get_prop_stack(distances, self.tritc_cleared)
            yfp_kymo_shell = self.get_prop_stack(distances, self.yfp_cleared)
            self.tritc_kymographs.append(tritc_kymo_shell[:, :-1])
            self.yfp_kymographs.append(yfp_kymo_shell[:, :-1])

        # Build DataFrames for each kymograph
        self.tritc_tdfs = []
        self.yfp_tdfs = []

        for idx in range(len(self.yfp_kymographs)):
            tritc_kymo_ = self.tritc_kymographs[idx]
            yfp_kymo_ = self.yfp_kymographs[idx]

            tritc_tdf = pd.DataFrame(
                data=tritc_kymo_.T, 
                columns=6 + np.arange(tritc_kymo_.shape[0])
            )
            yfp_tdf = pd.DataFrame(
                data=yfp_kymo_.T, 
                columns=6 + np.arange(yfp_kymo_.shape[0])
            )

            # Insert distance column
            tritc_tdf.insert(0, 'x', self.resolution * np.arange(tritc_tdf.shape[0]))
            yfp_tdf.insert(0, 'x', self.resolution * np.arange(yfp_tdf.shape[0]))

            # Adjust DataFrame indices
            tritc_tdf.index = tritc_tdf.index + 1
            yfp_tdf.index = yfp_tdf.index + 1

            self.tritc_tdfs.append(tritc_tdf)
            self.yfp_tdfs.append(yfp_tdf)

        # Rescale each kymograph in the Z dimension to 10x
        for idx in range(len(self.yfp_kymographs)):
            self.yfp_kymographs[idx] = transform.rescale(
                self.yfp_kymographs[idx],
                [10, 1]
            )
            self.tritc_kymographs[idx] = transform.rescale(
                self.tritc_kymographs[idx],
                [10, 1]
            )

        # Optional: add shapes to a napari-like viewer
        if display and viewer is not None:
            viewer.add_shapes(data=lines, shape_type='line',
                              edge_color='blue', face_color='blue',
                              edge_width=2)
            viewer.add_labels(distances)

    def get_ratio_df(self):
        """
        Create ratio DataFrames (YFP / TRITC) for each pair of YFP/TRITC DataFrames.
        """
        self.ratio_tdfs = []
        for tritc_tdf, yfp_tdf in zip(self.tritc_tdfs, self.yfp_tdfs):
            ratio_tdf = tritc_tdf.copy()
            ratio_tdf.iloc[:, 1:] = yfp_tdf.iloc[:, 1:] / tritc_tdf.iloc[:, 1:]
            self.ratio_tdfs.append(ratio_tdf)

    def setup_directories(self):
        """
        Create/check directories for montages, csvs, projections, and segmentations.
        
        Returns
        -------
        list
            A list of the created directory paths in the order
            [montage_dir, csv_dir, projection_dir, segmentation_dir].
        """
        parent_dir = os.path.join(self.path, '../')
        directories = ["montages", "csvs", "projections", "segmentations", "cleared"]
        dir_paths = []

        for directory in directories:
            dir_path = os.path.join(parent_dir, directory)
            dir_paths.append(dir_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")
            else:
                print(f"Directory already exists: {dir_path}")

        return dir_paths

    def make_projections(self):
        """
        Make maximum-intensity projections every 4 slices and save as .tif files.
        Also make a max projection of the latter half of the stack (tail).
        """
        combined = np.stack([self.tritc_cleared, self.yfp_cleared], axis=1)
        num_projections = int(np.ceil(combined.shape[0] / 4))
        projection_list = []

        for i in range(num_projections):
            subset = combined[i*4 : (i+1)*4]
            projections = np.max(subset, axis=0)
            projection_list.append(projections)

        # Save the projection stack
        projections = np.array(projection_list)
        out_path = os.path.join(self.projection_dir, f'{self.well}.tif')
        io.imsave(
            out_path,
            projections.astype(np.single),
            imagej=True,
            resolution = (1.0/self.resolution/.001, 1.0/self.resolution/.001),
            metadata={
                'unit': 'mm',
                'axes': 'TCYX',
                'mode': 'composite',
                'min': float(np.min(projections)),
                'max': float(np.max(projections))
            }
        )

        # "Tail" projection of second half
        half_idx = int(np.floor(combined.shape[0] / 2))
        back_half = combined[half_idx:].max(axis=0)
        out_tail = os.path.join(self.projection_dir, f'{self.well}_Tail.tif')
        io.imsave(
            out_tail,
            back_half.astype(np.single),
            imagej=True,
            resolution = (1.0/self.resolution/.001, 1.0/self.resolution/.001),
            metadata={
                'unit': 'mm',
                'axes': 'CYX',
                'mode': 'composite',
                'min': float(np.min(back_half)),
                'max': float(np.max(back_half))
            }
        )

    def process_well(self):
        """
        Main pipeline to process the well data from reading images
        through final saves of projections, CSV files, and montages.
        """
        # Create output directories
        (self.montage_dir,
         self.csv_dir,
         self.projection_dir,
         self.segmentation_dir,
         self.cleared_dir) = self.setup_directories()

        # Load and register images
        self.get_well_images()
        print('get_well done')
        self.register_images()
        print('register_images done')

        # Threshold and mask
        self.get_masks()
        print('get_masks done')

        # Build projections
        self.make_projections()
        print('make_projections done')

        # Determine center and radius for radial analyses
        self.x, self.y, self.r = self.get_center_and_radius_stack(self.masks)
        print('get_center_and_radius_stack done')

        # Create kymographs
        # viewer must be defined in your environment if display=True is desired
        self.get_kymographs(display=False, viewer=None)
        print('get_kymographs done')

        # Compute ratio DataFrames
        self.get_ratio_df()
        print('get_ratio_df done')

        # Debug / check the final projection directory
        print(self.projection_dir)

        # Save CSVs and montages for each set of kymographs
        for idx, (tritc_tdf, yfp_tdf, ratio_tdf) in enumerate(
            zip(self.tritc_tdfs, self.yfp_tdfs, self.ratio_tdfs)
        ):
            # CSVs
            tritc_out = os.path.join(self.csv_dir, f'{self.well}_TRITC_{idx}.csv')
            yfp_out   = os.path.join(self.csv_dir, f'{self.well}_YFP_{idx}.csv')
            ratio_out = os.path.join(self.csv_dir, f'{self.well}_Ratio_{idx}.csv')
            tritc_tdf.to_csv(tritc_out, index=False)
            yfp_tdf.to_csv(yfp_out, index=False)
            ratio_tdf.to_csv(ratio_out, index=False)

            # Montages
            yfp_img   = process_color_channel(yfp_tdf, 'YFP', resolution=self.resolution)
            tritc_img = process_color_channel(tritc_tdf, 'TRITC', resolution=self.resolution)
            ratio_img = process_color_channel(ratio_tdf, 'Ratio', resolution=self.resolution)
            montage   = create_montage([tritc_img, yfp_img, ratio_img], 1, 3)
            montage.save(os.path.join(self.montage_dir, f'{self.well}_{idx}.png'))

        # Save segmentation overlay
        seg_out = os.path.join(self.segmentation_dir, f'{self.well}.tif')
        io.imsave(seg_out, self.output_img)
