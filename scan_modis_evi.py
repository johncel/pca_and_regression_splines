import sys
import logging
import xarray as xr
import xml.etree.ElementTree as ET 
import numpy as np
import rasterio
import pycrs
from pyproj import Proj, transform
from functools import lru_cache


logging.basicConfig(level=logging.INFO)


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


DATASET = 'data/evi-granules/MOD13A3.A2020061.h08v05.006.2020100214130.hdf'

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
        

evi = MODISEVI(DATASET)
