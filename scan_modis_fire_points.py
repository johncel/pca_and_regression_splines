import logging
import shapefile


logging.basicConfig(level=logging.INFO)


DATASET = 'data/MODIS_C6_Global_24h'


class MODISFireShapes:
    def __init__(self, dataset_path):
        self.sf = shapefile.Reader(dataset_path)
        self.field_names = [x[1] for x in self.sf.fields]
        self.full_records = []
        for i,shp in enumerate(self.sf.shapes()):
            self.full_records.append(dict(zip(self.field_names, self.sf.record(i))))

    def print_contents(self):
        print(f'Bounding Box: {self.sf.bbox} shape type: {self.sf.shapeType} (points)')
        print(f'Record fields: {self.sf.fields}')
        for i,shp in enumerate(self.sf.shapes()):
            print(f'record {self.sf.record(i)}')
#            if shp.shapeType == shapefile.POLYGON:
#                zone_name = self.zones_sf.record(i)[1]


fires_shp = MODISFireShapes(DATASET)
fires_shp.print_contents()
