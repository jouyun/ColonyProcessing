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



def process_color_channel(idf, type, resolution=7.88955, max_z=None, min_z=None, color_scheme='viridis', ratio=False):
    df = idf.copy()
        
    if ratio:
        df.iloc[:, 1:] = np.log10(df.iloc[:, 1:] + 1)
    
    df = df.set_index('x').T.reset_index()
    df.columns = ['T'] + list(np.arange(len(df.columns[1:])))
    df = df.apply(pd.to_numeric, errors='coerce')
    
    if not pd.api.types.is_numeric_dtype(df['T']):
        raise Exception("'Y' column is not numeric.")
    
    df_long = df.melt(id_vars=['T'], var_name='X', value_name='Z')
    df_long['X'] = df_long['X'] * 0.001*resolution - 0.001*resolution
    
    max_x = df_long['X'].max()
    
    time_interval = 1.16667  # hours per frame
    df_long['Time_hr'] = df_long['T'] * time_interval

    pivoted = df_long.pivot(index="Time_hr", columns="X", values="Z")

    plt.figure(figsize=(10, 8))


    mask = pivoted.isna() | (pivoted == 0) 
    cmap = plt.get_cmap(color_scheme).copy()
    cmap.set_bad(color='gray')

    fig = sns.heatmap(pivoted, cmap=cmap, mask=mask, vmin=min_z, vmax=max_z, xticklabels=100, yticklabels=True)

    # Formatting
    ax = fig
    ax.invert_yaxis()
    ax.set_xlabel("Radius (mm)", fontsize=30)
    ax.set_ylabel("Time (h)", fontsize=30)

    max_rng = np.floor(max_x).astype(int) + 1
    xrng = (np.arange(0,max_rng,1))
    ax.set_xticks((xrng * 1/(0.001*resolution)).astype(int))
    ax.set_xticklabels(xrng)
    ax.tick_params(axis='x', labelsize=20)

    max_y = pivoted.index.max()
    yrng = np.arange(2,max_y, 10)
    ax.set_yticks((yrng * 1/time_interval).astype(int))
    ax.set_yticklabels(yrng+8)
    ax.tick_params(axis='y', labelsize=20)

    ax.set_title(f"{type}", fontsize=30)

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
    print(len(img_list))
    montaged = create_montage(img_list, math.ceil(len(img_list)/4), 4)
    montaged.save(dir_path + '/All.png')

def get_diff_df(df):
    diff_df = df.iloc[:,1:].diff(axis=1)
    diff_df.columns = df.columns[0:-1]
    diff_df['x'] = df['x']
    return diff_df

def make_montages(csv_dir, montage_dir, resolution, maxz=[45000,45000,1], minz=[0,0,0], dmaxz=[15000,15000,2], dminz=[0,0,0]):
    csvs = glob.glob(csv_dir + '*_TRITC_*.csv')
    csvs.sort()
    for idx, csv in enumerate(csvs):
        well = csv.split('\\')[-1].split('_')[0]
        yfp_tdf = pd.read_csv(csv.replace('_TRITC_', '_YFP_'))
        tritc_tdf = pd.read_csv(csv)
        ratio_tdf = pd.read_csv(csv.replace('_TRITC_', '_Ratio_'))

        # Standard montage
        tritc_img = process_color_channel(tritc_tdf, 'TRITC Intensity', resolution=resolution, max_z=maxz[0], min_z=minz[0], color_scheme='Reds_r')
        yfp_img   = process_color_channel(yfp_tdf, 'YFP Intensity', resolution=resolution, max_z=maxz[1], min_z=minz[1], color_scheme='viridis')
        ratio_img = process_color_channel(ratio_tdf, 'Ratio Intensity', resolution=resolution, max_z=maxz[2], min_z=minz[2], color_scheme='rocket', ratio=True)
        montage   = create_montage([tritc_img, yfp_img, ratio_img], 1, 3)
        montage.save(os.path.join(montage_dir, f'{well}_{idx}.png'))

        # Difference montage
        diff_yfp_df = get_diff_df(yfp_tdf)
        diff_tritc_df = get_diff_df(tritc_tdf)
        diff_ratio_df = get_diff_df(ratio_tdf)
        tritc_img = process_color_channel(diff_tritc_df, 'TRITC Derviative', resolution=resolution, max_z=dmaxz[0], min_z=dminz[0], color_scheme='Reds_r')
        yfp_img   = process_color_channel(diff_yfp_df, 'YFP Derviative', resolution=resolution, max_z=dmaxz[1], min_z=dminz[1], color_scheme='viridis')
        ratio_img = process_color_channel(diff_ratio_df, 'Ratio Derviative', resolution=resolution, max_z=dmaxz[2], min_z=dminz[2], color_scheme='rocket', ratio=False)
        montage   = create_montage([tritc_img, yfp_img, ratio_img], 1, 3)
        montage.save(os.path.join(montage_dir, f'{well}_{idx}_diff.png'))

