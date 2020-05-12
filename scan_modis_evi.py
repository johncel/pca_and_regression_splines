import sys
import logging
import xarray as xr
import xml.etree.ElementTree as ET 
import numpy as np
import rasterio
import pycrs
from pyproj import Proj, transform
from functools import lru_cache
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from mpl_toolkits.basemap import Basemap

import rasterio
from rasterio import Affine as A, transform
from rasterio.warp import reproject, Resampling, calculate_default_transform


logging.basicConfig(level=logging.INFO)


DATASET = 'data/evi-granules/MOD13A3.A2020061.h08v05.006.2020100214130.hdf'


@lru_cache(maxsize=128)
def get_ll_proj():
    ll_proj = Proj(init='epsg:4326')
    return ll_proj


def convert_proj_ll(points, in_proj):
    ll_proj = get_ll_proj()

    ll_points = []
    for point in points:
        ll_points.append(in_proj(point[0], point[1], inverse=True))

    return ll_points


# g_ds = None
# @lru_cache(maxsize=None)
# def xr_sel(y,x):
#     val = g_ds[0].sel(y=y, x=x, method='nearest')
#     return val


class MODISEVI:
    def __init__(self, dataset_path):
        # attributes we will set
        self.ll_xy = [] # list of lat lon projected points for each grid point in this set of datasets
        self.modis_proj = None
        self.ds_dict = {} # dictionary of the datasets in this hdf file 
        self.min_lat = 9999
        self.max_lat = -9999
        self.min_lon = 9999
        self.max_lon = -9999
        self.ul = None
        self.ur = None
        self.ll = None
        self.lr = None

        first_dataset = None
        with rasterio.open(dataset_path) as src:
            subdatasets = src.subdatasets
            logging.debug(subdatasets)
            for subdataset in subdatasets:
                key = subdataset.split(':')[-1]
                if first_dataset is None:
                    first_dataset = key
                    self.ds = xr.open_rasterio(subdataset)
                self.ds_dict[key] = xr.open_rasterio(subdataset)
                logging.info(f'ds_dict adding :{key}:')

        logging.debug(self.ds)

        proj_str = self.ds.attrs['crs']
        logging.info(f'proj str: {proj_str}')
        self.modis_proj = Proj(proj_str)

        xy = list(zip(np.array(self.ds['x']), np.array(self.ds['y'])))
        logging.debug(f' xy: {xy}')
        self.ll_xy = convert_proj_ll(xy, self.modis_proj)

        for i, xy_i in enumerate(xy):
            logging.debug(f' xy: {xy_i} ll_xy: {self.ll_xy[i]}')

        # open the meta data to get the bounding box info
        meta_path = f'{dataset_path}.xml'
        tree = ET.parse(meta_path)
        root = tree.getroot()
        logging.debug(root)
        # boundaries = root.findall('./GranuleMetaDataFile/GranuleURMetaData/SpatialDomainContainer/HorizontalSpatialDomainContainer/GPolygon/Boundary')
        boundaries = root.findall('./GranuleURMetaData/SpatialDomainContainer/HorizontalSpatialDomainContainer/GPolygon/Boundary/Point')
        logging.debug(boundaries)
        points = []
        for point in boundaries:
            lat = float(point[1].text)
            lon = float(point[0].text)
            logging.debug(f'lat: {lat} lon:{lon}')
            if lat < self.min_lat:
                self.min_lat = lat
            if lon < self.min_lon:
                self.min_lon = lon
            if lat > self.max_lat:
                self.max_lat = lat
            if lon > self.max_lon:
                self.max_lon = lon
            points.append((lat,lon))
    
        logging.info(f'min/max lat: {self.min_lat}, {self.max_lat}')
        logging.info(f'min/max lon: {self.min_lon}, {self.max_lon}')

        # sort the points by latitude
        points = sorted(points, key=lambda point: (point[0],point[1]))
        logging.debug(f'points: {points}')

        self.ul = points[2]
        self.ur = points[3]
        self.ll = points[0]
        self.lr = points[1]
        logging.info(f'points: ul:{self.ul} ur:{self.ur} ll:{self.ll} lr:{self.lr}')

    def plot_dataset(self, key):
        # fig = plt.figure(figsize=(7,8))
        # ax = fig.add_subplot(111)
        # ax.set_title(f'MODIS {key}')
        # ax.axis('equal')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        ds = self.ds_dict[key]
        print(f'THIS IS THE ARRAY {ds[0]} {ds[0,0,0]}')
        # ax.imshow(ds[0])

        ds_reproject = np.zeros((ds.shape[1:])) - 3000
        dy = (self.max_lat - self.min_lat) / ds_reproject.shape[0]
        dx = (self.max_lon - self.min_lon) / ds_reproject.shape[1]

        rows, cols = src_shape = ds.shape[1:]
        d = 1 / 111.0 # degrees per pixel (1 km / 111 km / degree)
        # src_transform = A.translation(-cols*d/2, rows*d/2) * A.scale(d, -d)
        left = float(min(ds[0]['x']))
        right = float(max(ds[0]['x']))
        top = float(max(ds[0]['y']))
        bottom = float(min(ds[0]['y']))
        print(f'left: {left} right: {right} top: {top} bottom: {bottom}')
        src_transform = transform.from_bounds(left, bottom, right, top, cols, rows)
        src_crs = self.ds.attrs['crs']
        print(f'source crs: {src_crs}')
        source = np.array(ds[0])
        dst_shape = ds.shape
        # dst_transform = A.translation(-237481.5, 237536.4) * A.scale(425.0, -425.0)
        dst_crs = {'init': 'EPSG:4326'}
        dst_transform, dst_width, dst_height = calculate_default_transform(src_crs, dst_crs, cols, rows, 
                                                                           left=left, right=right, top=top, bottom=bottom)
        destination = np.zeros((dst_height, dst_width))
        reproject(
            source,
            destination,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)

        print(f' dst_transform: {dst_transform} destination: {destination} shape: {destination.shape} max: {np.max(destination)} min: {np.min(destination)}')


        # draw a basemap
        m = Basemap(projection='cyl',llcrnrlat=self.min_lat,urcrnrlat=self.max_lat,\
           llcrnrlon=self.min_lon,urcrnrlon=self.max_lon,resolution='h')
        m.drawcoastlines(linewidth=0.25)
        m.drawcountries(linewidth=0.25)
        m.drawstates(linewidth=0.20)
        m.drawlsmask(land_color='0.8', ocean_color='w', lsmask=None, lsmask_lons=None, lsmask_lats=None, lakes=True, resolution='i', grid=1.25)
        m.drawmeridians(np.arange(0,360,5), labels=[True, True, True, True])
        m.drawparallels(np.arange(-90,90,5), labels=[True, True, True, True])

        ax = plt.gca()

        # ax.plot([-113, -100], [36, 40], linewidth=4)
        extent = [self.min_lon, self.max_lon, self.min_lat, self.max_lat]
        # ax.imshow(destination, extent=extent, zorder=3)
        ax.imshow(destination, extent=extent)

        plt.show()
        

evi = MODISEVI(DATASET)

evi.plot_dataset('1 km monthly EVI')
