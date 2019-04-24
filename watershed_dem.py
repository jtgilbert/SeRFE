# imports
import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona
import geopandas as gpd
import os


def watershed_dem(dem, watershed, crs_epsg, out_dem):
    """
    This function projects a dem to a coordinate reference system and then clips
    it to a watershed boundary.
    PARAMS
    :dem: string - path to a dem raster file (merged from individual tiles if necessary)
    :watershed: string - path to a shapefile of a watershed boundary
    :crs_epsg: int - desired coordinate reference system (epsg number) for output
    :out_dem: string - path to clipped output dem
    :return:
    """

    # convert epsg number into crs dict
    dst_crs = {'init': 'epsg:{}'.format(crs_epsg)}
    projected = os.path.dirname(dem) + '/projected.tif'

    # reproject dem to desired projected coordinate system
    with rasterio.open(dem) as src:
        if src.crs == dst_crs:
            pass
        else:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(projected, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear)

    # project watershed boundary shapefile to same projection as dem
    wats = gpd.read_file(watershed)
    wats_proj = os.path.dirname(watershed) + '/watershed_proj.shp'
    if wats['geometry'].crs == dst_crs:
        wats.to_file(wats_proj)
    else:
        wats_proj = wats.to_crs(dst_crs)
        wats_proj.to_file(wats_proj)

    # clip projected dem to projected watershed boundary extent
    with fiona.open(wats_proj, 'r') as shapefile:
        features = [feature['geometry'] for feature in shapefile]

    with rasterio.open(projected) as src:
        out_image, out_transform = rasterio.mask.mask(src, features, crop=True)
        out_meta = src.meta.copy()

    if dem[-3:] == 'tif':
        driver = 'GTiff'
    elif dem[-3:] == 'img':
        driver = 'HFA'
    else:
        raise Exception('DEM is not a .tif or .img file')

    out_meta.update({'driver': driver,
                     'height': out_image.shape[1],
                     'width': out_image.shape[2],
                     'transform': out_transform})
    with rasterio.open(out_dem, 'w', **out_meta) as dest:
        dest.write(out_image)

    # clean temp files
    # os.remove(wats_proj)  # this only gets rid of the .shp file, find better way
    # os.remove(projected)

    return

