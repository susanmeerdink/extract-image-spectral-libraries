# Import Modules
import glob
import numpy as np
import rasterio.plot
from rasterio import features
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import fiona
from descartes import PolygonPatch
from shapely.geometry import shape, MultiPolygon
from shapely.ops import transform
import pyproj
from functools import partial
import random
import gc
from os.path import basename
import os

def load_polygons(polyLocation):
    '''
    This function opens the polygon shapefile and transforms it to the appropriate coordinate system.
    Polygons MUST be in the same georeferencing format as images!
    
    Parameters:
    -----------
    1) polyLocation: string containing the directory location of the shapefile
    
    Returns:
    --------
    1) polygons: the reference polygons as a shapely MultiPolygon object
    2) polyOrig: the original shapefile with polygons as Fiona Object - contains metadata still
    3) metadata: a numpy array of metadata from shapefile
    4) header: a list of strings with the metadata headers
    '''
    # Open shapefile and transform to appropriate coordinate system
    polyOrig = fiona.open(polyLocation + '.shp', "r")  # Open polygon to get crs information for transformation
    
    # Determine if polygons need to be transformed to match imagery
    polygons = MultiPolygon([shape(pol['geometry']) for pol in fiona.open(polyLocation + '.shp')])  # Open polygon as a MultiPolygon for processing purposes
        
        # code to transform polygons if necessary
        #         polyMP = MultiPolygon([shape(pol['geometry']) for pol in fiona.open(polyLocation + '.shp')])  # Open polygon as a MultiPolygon for processing purposes
        #         project = partial(  # Define function for projection process
        #             pyproj.transform,
        #             pyproj.Proj(polyOrig.crs),  # source coordinate system
        #             pyproj.Proj(init='epsg:32611'))  # destination coordinate system, UTM Zone 11 WGS 84
        #         polygons = transform(project, polyMP)  # apply projection
        #         print('Reference Polygons Transformed to UTM Zone 11 WGS 84')
    
    # Get Metadata from Polygons
    metadata = np.empty([0, len(list(polyOrig[0]['properties'].values())) ])
    for i in range(0, len(polygons)):
        metadata = np.vstack((metadata, list(polyOrig[i]['properties'].values())))
    meta = polyOrig.meta
    header = list(meta['schema']['properties'].keys())
    
    print(len(polygons), "Reference Polygons Found")
    
    return polygons, polyOrig, metadata, header

def load_metadata(metaLocation, column=0):
    '''
    This function reads in the metadata into a numpy array for processing.
    
    Parameters:
    -----------
    1) metaLocation: a string containing the directory location of the metadata
    2) column: a string used to find the column for matching to polygon metadata
                If no string is provided, the first column will be used
    
    Returns:
    --------
    1) metadata: a numpy array containing the metadata for the reference polygons
    2) headers: Save the column names/headers of the metadata
    3) polyIndex: The column that contains the polygon name to match with shapefile
    '''
    # Load Metadata
    metadata = np.loadtxt(metaLocation, dtype=object, delimiter=',')  # Load in metadata
    headers = metadata[0, :]  # save headers separate of metadata
    headers = np.char.strip(headers.astype(str))  # remove whitespace from headers
    metadata = np.delete(metadata, 0, 0)  # remove the headers
    if column == 0:
        polyIndex = 0
    else:
        polyIndex = metadata[:, (np.where(headers == column)[0][0])]  # pull out what class
    
    return metadata, headers, polyIndex

def write_spectra(outFilename, header, spectra, meta):
    """
    Function that writes spectra to .csv file
    
    Parameters:
    1) outFilename: string containing the full path to save the spectra
    2) header: a list of string to be used for file header
    3) spectra: a numpy array of the extract spectra (output from extract_spectra_single or extract_spectra_batch)
    4) meta: a numpy array of the metadata associated with the spectra
    """
    fileOutSpec = open((outFilename + '_spectra.csv'), 'wb')
    headerOutSpec = 'ImageNumber, PolygonName, X, Y,' + ','.join(map(str, header))
    allSpec = np.hstack((meta, spectra))
    np.savetxt(fileOutSpec, allSpec, header=headerOutSpec, fmt='%s', delimiter=",")
    fileOutSpec.close()

def write_metadata(outFilename, header, meta):
    """
    Function that writes metadata to .csv file
    
    Parameters:
    1) outFilename: string containing the full path to save the spectra
    2) header: a list of string to be used for file header
    3) meta: a numpy array of the metadata associated with the spectra
    """
    fileOutMeta = open((outFilename + '_metadata.csv'), 'wb')
    headerOutMeta = 'ImageNumber, PolygonName, X, Y,' + ','.join(header)
    np.savetxt(fileOutMeta, meta, header=headerOutMeta, fmt='%s', delimiter=",")
    fileOutMeta.close()

def extract_spectra_single(imgLocation, polygons, polygonsOriginal, metadata=0, polyIndex=0, matchIndex=0):
    '''
    Function that reads in a single image and extracts the data from reference polygon locations.
    
    Parameters:
    -----------
    1) imgLocation: string with main directory and filename of image to be processed
    2) polygons: an object that contains all the polygons to be used in extracting spectra
    3) polygonsOriginal: a Fiona object containing the original polygons with metadata from shapefile
    4) metadata: a numpy array containing all the metadata for the polygons
                default is no metadata (0), if not provided only spectra will be extracted and NOT linked to metadata.
    5) polyIndex: an integer specifying which column should be used to link spectra to metadata
    6) matchIndex: an integer specifying which column should be used to link metadata to spectra
    
    Returns:
    --------
    1) spectralLibData: numpy array that returns all the spectra for a polygon
    2) spectralLibName: numpy array that returns an identifier for a polygon [Image#, PolygonID, x, y]
    3) spectralLibMeta: numpy array that returns all the metadata for a polygon
    4) headers: Save the column names/headers of the spectra or wavelengths
    '''

    print('Extracting spectra from ' + basename(imgLocation))
    imgFile = rasterio.open(imgLocation, 'r')  # Open raster image
    headers = imgFile.indexes  # column header for wavelength names
    shortName = basename(imgLocation)  # Get file name
    propname = [i for i in polygonsOriginal[0]['properties'].keys()]
    propname = propname[matchIndex]
    
    # Variables to hold the entire spectral library with metadata
    spectralLibData = np.empty([0, imgFile.count])
    spectralLibName = np.empty([0, 4])
    if type(metadata) is int:
        spectralLibMeta = ''
    else:
        spectralLibMeta = np.empty([0, metadata.shape[1] + 4])
        
    spectraCount = 0
    for idx in range(0, len(polygons)):  # Loop through polygons
        gc.collect()
        polyIn = polygons[idx]
        polyName = polygonsOriginal[idx]['properties'][propname]
        pixelCount = 0

        # Create Mask that has 1 for locations with polygons and 0 for non polygon locations
        polygonMask = rasterio.features.rasterize([(polyIn, 1)], out_shape=imgFile.shape,
                                                  transform=imgFile.transform, all_touched=False)
        test = np.count_nonzero(polygonMask)  # Get the number of elements that are not zero
        if test > 0:  # If there is data for this polygon assign the data
            indices = np.nonzero(polygonMask)
            for i in range(0, len(indices[0])): 
                x = indices[0][i]
                y = indices[1][i]
                window = ((x, x + 1), (y, y + 1))
                data = imgFile.read(window=window)  # Extract spectra from image
                pixel = np.transpose(data[:, 0, 0])
                if any(pixel):  # If there are non zero values save them to spectral library
                    pixelCount += 1  # How many pixels are in this polygon
                    spectraCount += 1  # How many spectra were collected from flightline
                    inName = [0, polyName, x, y]
                    spectralLibData = np.vstack((spectralLibData, pixel))
                    spectralLibName = np.vstack((spectralLibName, inName))
                    
                    if type(metadata) is not int:
                        inMeta = np.hstack((inName, metadata[idx, :]))
                        spectralLibMeta = np.vstack((spectralLibMeta, inMeta))
            
    return spectralLibData, spectralLibName, spectralLibMeta, headers 

def extract_spectra_batch(dirLocation, polygons, polygonsOriginal, metadata=0, polyIndex=0, matchIndex=0):
    """
    Function that extracts spectra from multiple images and saves them into a single file.
    
    Parameters:
    -----------
    1) dirLocation: string with main directory containing the images to be processed
    2) polygons: an object that contains all the polygons to be used in extracting spectra
    3) polygonsOriginal: a Fiona object containing the original polygons with metadata from shapefile
    4) metadata: a numpy array containing all the metadata for the polygons
                default is no metadata (0), if not provided only spectra will be extracted and NOT linked to metadata.
    5) polyIndex: an integer specifying which column should be used to link spectra and metadata
    
    Returns:
    --------
    1) spectralLibData: numpy array that returns all the spectra for a polygon
    2) spectralLibName: numpy array that returns an identifier for a polygon [imageNumber, PolygonName, x, y]
    3) spectralLibMeta: numpy array that returns all the metadata for a polygon
    4) headers: Save the column names/headers of the spectra or wavelengths
    """
    count = 0
    for singlefile in glob.glob(dirLocation):
        if '.hdr' not in singlefile:
            print ('Extracting spectra from ' + os.path.basename(singlefile))
            imgFile = rasterio.open(singlefile, 'r')  # Open raster image
            headers = imgFile.indexes  # column header for wavelength names
            shortName = os.path.basename(singlefile)  # Get file name

            if count == 0:
                # Variables to hold the entire spectral library with metadata
                spectralLibData = np.empty([0, imgFile.count])
                spectralLibName = np.empty([0, 4])
                if metadata != 0:
                    spectralLibMeta = np.empty([0, metadata.shape[1] + 4])
                else:
                    spectralLibMeta = ''

            count = count + 1
            spectraCount = 0
            for idx in range(0, len(polygons)):  # Loop through polygons
                gc.collect()
                polyIn = polygons[idx]  
                polyName = polygonsOriginal[idx]['properties'][matchIndex]
                pixelCount = 0

                # Create Mask that has 1 for locations with polygons and 0 for non polygon locations
                polygonMask = rasterio.features.rasterize([(polyIn, 1)], out_shape=imgFile.shape,
                                                          transform=imgFile.transform, all_touched=False)
                test = np.count_nonzero(polygonMask)  # Get the number of elements that are not zero
                if test > 0:  # If there is data for this polygon assign the data
                    indices = np.nonzero(polygonMask)
                    for i in range(0, len(indices[0])): 
                        x = indices[0][i]
                        y = indices[1][i]
                        window = ((x, x + 1), (y, y + 1))
                        data = imgFile.read(window=window)  # Extract spectra from image
                        pixel = np.transpose(data[:, 0, 0])
                        if any(pixel):  # If there are non zero values save them to spectral library
                            pixelCount += 1  # How many pixels are in this polygon
                            spectraCount += 1  # How many spectra were collected from flightline
                            inName = [count, shortName, polyName, x, y]
                            spectralLibData = np.vstack((spectralLibData, pixel))
                            spectralLibName = np.vstack((spectralLibName, inName))

                            if metadata != 0:
                                inMeta = np.hstack((inName, metadata[np.where(polyIndex == polyName)[0][0], :]))
                                spectralLibMeta = np.vstack((spectralLibMeta, inMeta))
            
    return spectralLibData, spectralLibName, spectralLibMeta, headers 

def create_spec_lib(polyLocation, imageLocation, outLocation, mode=0, metaLocation=1):
    '''
    This function calls other functions to create a spectral library for an image or set of images.
    
    Parameters:
    -----------
    1) polyLocation: a string containing the location and filename of the reference polygon shapefile,
                     Polygon MUST be in the same georeferencing as images!
    2) imageLocation: a string containing the location of all the images
    3) outLocation: a string containing the location you want the output files to be saved
    4) mode: Only used when providing a directory and not a single file! Flag that changes how output files will be stored.
            Mode 0: images will be saved as separate .csv files
            Mode 1: images will be saved as a single .csv file
    5) metaLocation: There are three options for this paramter:
            Option 1: DEFAULT, uses metadata contained in the shapefile, change nothing in variables
            Option 2: provide a .csv of metadata variables to do so provide a list object with three items in this order
                      1) a string containing the location and filename of the metadata csv
                      2) a string containing the header value for metadata to match with polygon ID
                      3) a string containing the header value for polygon ID to match with metadata
            Option 3: provide no metadata by setting value to 0, if not provided only spectra will be extracted and NOT linked to metadata.
    
    Outputs:
    --------
    1) Metadata CSV: Saves a csv with all the metadata associated with each column of spectra extracted
    2) Spectra CSV: Saves a csv with all the spectra extracted from the data with 5 columns designated for identification.    
    '''
    
    # load in polygons
    polygons, polygonsOriginal, polyMeta, polyHeader = load_polygons(polyLocation)
    
    # Extract Spectra for single image
    if os.path.isfile(imageLocation):
        outName = (str.rstrip(imageLocation, '.') + '_spectral_library')
        
        # Load in metadata from .csv
        if isinstance(metaLocation,str):
            metadata, metaColumnHeader, polyIndex = load_metadata(metaLocation[0], metaLocation[1])
            spec, name, meta, specColumnHeader,headers = extract_spectra_single(imageLocation, polygons, polygonsOriginal, metadata, polyIndex, metaLocatin[2])
            write_spectra(outName, specColumnHeader, spec, name)
            write_metadata(outName, metaColumnHeader, meta)
              
        else:
            # Do not use metadata 
            if metaLocation == 0:
                spec, name, specColumnHeader,headers = extract_spectra_single(imageLocation, polygons, polygonsOriginal)
                write_spectra(outName, specColumnHeader, spec, name)            
            
            # Load in metadata from shapefile
            else:
                spec, name, meta, specColumnHeader, headers = extract_spectra_single(imageLocation, polygons, polygonsOriginal, polyMeta)
                write_spectra(outName, specColumnHeader, spec, name)
                write_metadata(outName, polyHeader, meta)

    
    # Extract Spectra from directory
    else:
        # If in mode 0, process each image separately
        if mode == 0: 
            for singlefile in glob.glob(imageLocation):
                if '.hdr' not in singlefile:
                    outName = (str.rstrip(singlefile, '.') + '_spectral_library')

                    # Load in metadata from .csv
                    if len(metaLocation) > 1:
                        metadata, metaColumnHeader, polyIndex = load_metadata(metaLocation)
                        spec, name, meta, specColumnHeader,headers = extract_spectra_single(singlefile, polygons, polygonsOriginal, metadata, polyIndex)
                        write_spectra(outName, specColumnHeader, spec, name)
                        write_metadata(outName, metaColumnHeader, meta)

                    
                    else:
                        # Do not use metadata 
                        if metaLocation == 0:
                            spec, name, specColumnHeader, headers = extract_spectra_single(imageLocation, polygons, polygonsOriginal)
                            write_spectra(outName, specColumnHeader, spec, name)            

                        # Load in metadata from shapefile
                        else:
                            spec, name, meta, specColumnHeader, headers = extract_spectra_single(imageLocation, polygons, polygonsOriginal, polyMeta)
                            write_spectra(outName, specColumnHeader, spec, name)
                            write_metadata(outName, polyHeader, meta)
                        
        # If in mode 1, process images in directory together
        if mode == 1:
            # Load in metadata from .csv
            if len(metaLocation) > 1:
                metadata, metaColumnHeader, polyIndex = load_metadata(metaLocation)
                spec, name, meta, specColumnHeader = extract_spectra_batch(imageLocation, polygons, polygonsOriginal, metadata, polyIndex)
                write_spectra(outName, specColumnHeader, spec, name)
                write_metadata(outName, metaColumnHeader, meta)
            
            
            else:
                # Do not use metadata 
                if metaLocation == 0:
                    spec, name, specColumnHeader = extract_spectra_batch(imageLocation, polygons, polygonsOriginal)
                    write_spectra(outName, specColumnHeader, spec, name)            

                # Load in metadata from shapefile
                else:
                    spec, name, meta, specColumnHeader = extract_spectra_batch(imageLocation, polygons, polygonsOriginal, polyMeta)
                    write_spectra(outName, specColumnHeader, spec, name)
                    write_metadata(outName, polyHeader, meta)

    print('Completed processing')