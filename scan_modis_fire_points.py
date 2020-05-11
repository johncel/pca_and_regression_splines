import shapefile

DATASET = 'data/MODIS_C6_Global_24h'

class MODISFireShapes:
    def __init__(self, dataset_path):
        self.sf = shapefile.Reader(dataset_path)

    def print_contents(self):
        print(f'Bounding Box: {self.sf.bbox} shape type: {self.sf.shapeType} (points)')
        print(f'Record fields: {self.sf.fields}')
        field_names = [x[1] for x in self.sf.fields]
        for i,shp in enumerate(self.sf.shapes()):
            full_record = dict(zip(field_names, self.sf.record(i)))
            print(f'record {self.sf.record(i)}')
#            if shp.shapeType == shapefile.POLYGON:
#                zone_name = self.zones_sf.record(i)[1]


fires_shp = MODISFireShapes(DATASET)
fires_shp.print_contents()
