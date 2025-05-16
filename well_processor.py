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
from helper import save_channel_avi_with_timestamp, straighten_image


class WellProcessor:
    """
    A class to process microscopy images from a particular well directory.
    It handles the loading, registration, masking, and generation of intensity
    plots (kymographs) for TRITC and YFP channels.
    """
    def __init__(self, well, path, time_clip=2, magnification=1.25):
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
        self.derivative_stack = None

        # Kymographs
        self.tritc_kymographs = []
        self.yfp_kymographs = []

        # DataFrames
        self.tritc_tdfs = []
        self.yfp_tdfs = []
        self.ratio_tdfs = []

        # Derivative kymographs
        self.tritc_deriv_kymographs = []
        self.yfp_deriv_kymographs = []
        self.ratio_deriv_kymographs = []

        # Derivative DataFrames (CSV outputs)
        self.tritc_deriv_tdfs = []
        self.yfp_deriv_tdfs = []
        self.ratio_deriv_tdfs = []

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
        self.derivative_dir = None
        self.derivative_csv_dir = None    
        self.derivative_kymograph_dir = None    


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
        
        # Resize to match the first image in the stack
        # target_shape = io.imread(tritc_paths[0]).shape
        # tritc_stack = [resize(io.imread(f), target_shape, preserve_range=True).astype(np.uint16)
        #                 for f in tritc_paths]
        # self.tritc = np.stack(tritc_stack, axis=0)


        self.tritc = self.crop_and_subtract_background(tritc_stack)

        # Load YFP
        yfp_paths = glob.glob(os.path.join(self.path, f'{self.well}_*YFP*.tif'))
        yfp_paths.sort()
        yfp_stack = np.array([io.imread(f) for f in yfp_paths])
        self.yfp = self.crop_and_subtract_background(yfp_stack)

        # target_shape = io.imread(yfp_paths[0]).shape
        # yfp_stack = [resize(io.imread(f), target_shape, preserve_range=True).astype(np.uint16)
        #                 for f in yfp_paths]
        # self.yfp = np.stack(yfp_stack, axis=0)

        #self.yfp = self.crop_and_subtract_background(yfp_stack)


    def register_images(self):
        """
        Register TRITC stack to itself using 'previous' reference, then
        apply the same transform to YFP.
        """
        sr = StackReg(StackReg.TRANSLATION)
        # self.yfp_registered = sr.register_transform_stack(
        #     self.yfp,
        #     reference='previous',
        #     verbose=True
        # )
        self.tritc_registered = sr.register_transform_stack(
            self.tritc,
            reference='previous',
            verbose=True
        )
        #self.tritc_registered = sr.transform_stack(self.tritc)
        self.yfp_registered = sr.transform_stack(self.yfp)

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
        self.masks[0:2] = self.masks[2]

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
        Generate multiple kymographs along various angles
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
        Create/check directories for montages, csvs, projections, segmentations,
        cleared image stacks, and derivative kymograph outputs.

        Returns
        -------
        list
        A list of the created directory paths in the order:
        [montage_dir, csv_dir, projection_dir, segmentation_dir, cleared_dir, derivative_dir, derivative_csv_dir, derivative_kymograph_dir]
        """

        parent_dir = os.path.join(self.path, '../')
        directories = ["montages", 
                       "csvs", 
                       "projections", 
                       "segmentations", 
                       "cleared", ]
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

    def process_well(self, remake=False):
        """
        Main pipeline to process the well data from reading images
        through final saves of projections, CSV files, and montages.
        """
        # Create output directories
        (self.montage_dir,
         self.csv_dir,
         self.projection_dir,
         self.segmentation_dir,
         self.cleared_dir,) = self.setup_directories()

        # Check cleared_dir, if any files are already inside and (if remake is also False), just return
        if not remake and os.path.exists(self.cleared_dir):
            cleared_files = glob.glob(os.path.join(self.cleared_dir, f'{self.well}*'))
            if cleared_files:
                print(f"Cleared directory already contains files for {self.well}.")
                return


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

        # Save CSVs and for each set of kymographs
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

        # Save segmentation overlay
        seg_out = os.path.join(self.segmentation_dir, f'{self.well}.tif')
        io.imsave(seg_out, self.output_img)