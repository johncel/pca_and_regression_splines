import sys
import xarray as xr
import xml.etree.ElementTree as ET 
import numpy as np
import rasterio
import pycrs
from pyproj import Proj, transform
from functools import lru_cache
# import gdal


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
        with rasterio.open(dataset_path) as src:
            subdatasets = src.subdatasets
            print(subdatasets)
            for subdataset in subdatasets:
                self.ds = xr.open_rasterio(subdataset)
                break


        #self.ds = xr.open_dataset(dataset_path)
        print(self.ds)

        proj_str = self.ds.attrs['crs']
        print(f'proj str: {proj_str}')
        modis_proj = Proj(proj_str)

        xy = list(zip(np.array(self.ds['x']), np.array(self.ds['y'])))
        print(f' xy: {xy}')
#        sys.exit(1)
        ll_xy = convert_proj_ll(xy, modis_proj)

        for i, xy_i in enumerate(xy):
            print(f' xy: {xy_i} ll_xy: {ll_xy[i]}')

#        nx = self.ds[0].dims['XDim:MOD_Grid_monthly_1km_VI']
#        ny = self.ds[0].dims['YDim:MOD_Grid_monthly_1km_VI']

        # open the meta data to get the projection info
        meta_path = f'{dataset_path}.xml'
        tree = ET.parse(meta_path)
        root = tree.getroot()
        print(root)
        # boundaries = root.findall('./GranuleMetaDataFile/GranuleURMetaData/SpatialDomainContainer/HorizontalSpatialDomainContainer/GPolygon/Boundary')
        boundaries = root.findall('./GranuleURMetaData/SpatialDomainContainer/HorizontalSpatialDomainContainer/GPolygon/Boundary/Point')
        print(boundaries)
        min_lat = 9999
        max_lat = -9999
        min_lon = 9999
        max_lon = -9999
        points = []
        for point in boundaries:
            # print(f'lat: {point.attrib["PointLatitude"]} lon: {point.attrib["PointLongitude"]}')
            lat = float(point[1].text)
            lon = float(point[0].text)
            print(f'lat: {lat} lon:{lon}')
            if lat < min_lat:
                min_lat = lat
            if lon < min_lon:
                min_lon = lon
            if lat > max_lat:
                max_lat = lat
            if lon > max_lon:
                max_lon = lon
            points.append((lat,lon))
    
        print(f'min/max lat: {min_lat}, {max_lat}')
        print(f'min/max lon: {min_lon}, {max_lon}')

        # sort the points by latitude
        points = sorted(points, key=lambda point: (point[0],point[1]))
        print(f'points: {points}')

        ul = points[2]
        ur = points[3]
        ll = points[0]
        lr = points[1]
        print(f'points: ul:{ul} ur:{ur} ll:{ll} lr:{lr}')
        # x = (lon - lon0) cos (lat)
        # y = lat


        

evi = MODISEVI(DATASET)
