#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "gisCoordinates"
__module__ = "etl"
__package__ = "transform"
__app__ = "rezaware"
__ini_fname__ = "app.ini"
__conf_fname__ = "app.cfg"

''' Load necessary and sufficient python librairies that are used throughout the class'''
try:
    import os
    import sys
    import configparser    
    import logging
    import traceback
    import functools
    import findspark
    findspark.init()
    from pyspark.sql import functions as F
    from pyspark.sql import DataFrame
    import pandas as pd
#     from geopandas import GeoDataFrame # Loading boundaries Data
    from shapely.geometry import Point, Polygon, shape # creating geospatial data
    from shapely import wkb, wkt # creating and parsing geospatial data
    import numpy as np
    import json
    import math

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))

'''
    CLASS transform geographic locations: country and subdivisions to ISO 3166-2 and
        3166-3 codes using fuzzy matching.

    Contributors:
        * nuwan.waidyanatha@rezgateway.com
        * samana.thetha@gmail.com

    Resources:
        https://pypi.org/project/pycountry/
'''
class dataWorkLoads():
    ''' Function --- INIT ---
    
        author: <nuwan.waidyanatha@rezgateway.com>

    '''
    def __init__(
        self, 
        desc : str = None,   # identifier for the instances
        **kwargs,
    ):
        """
        Description:
            Initializes the postGIS dataWorkloads: class property attributes, app 
            configurations, logger function, data store directory paths, and global classes
        Attributes:
            desc (str) to change the instance description for identification
        Returns:
            None
        """

        self.__name__ = __name__
        self.__package__ = __package__
        self.__module__ = __module__
        self.__app__ = __app__
        self.__ini_fname__ = __ini_fname__
        self.__conf_fname__ = __conf_fname__
        if desc is not None or "".join(desc.split())!="":
            self.__desc__ = desc
        else:
            self.__desc__ = " ".join([self.__app__, self.__module__, self.__package__, self.__name__])

        ''' Initialize property var to hold the data '''
        self._data = None

        ''' initiate to load app.cfg data '''
        global logger  # inherits the utils logger class
        global pkgConf # inherits package app.ini config data
        global clsSpark  # inherits the loader sparkRDBM dataWorkLoads

        __s_fn_id__ = f"{self.__name__} function <__init__>"

        try:
            self.cwd=os.path.dirname(__file__)
            pkgConf = configparser.ConfigParser()
            pkgConf.read(os.path.join(self.cwd,__ini_fname__))

            self.rezHome = pkgConf.get("CWDS","PROJECT")
            sys.path.insert(1,self.rezHome)

            ''' innitialize the logger '''
            from rezaware.utils import Logger as logs
            logger = logs.get_logger(
                cwd=self.rezHome,
                app=self.__app__, 
                module=self.__module__,
                package=self.__package__,
                ini_file=self.__ini_fname__)
            ''' set a new logger section '''
            logger.info('########################################################')
            logger.info("%s %s",self.__name__,self.__package__)

            ''' instantiate sparkRDBM dataWorkLoads '''
            from rezaware.modules.lib.spark import execSession as spark
            clsSpark = spark.Spawn(desc=self.__desc__)

            logger.info("%s Connection complete! ready to load data.",__s_fn_id__)
            logger.debug("%s initialization for %s module package %s %s done.\nStart workloads: %s."
                         %(self.__app__.upper(),
                           self.__module__.upper(),
                           self.__package__.upper(),
                           self.__name__.upper(),
                           self.__desc__.upper()))

            print("%s Class initialization complete" % self.__name__)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None


    ''' Function -- DATA --
            TODO: 
            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    @property
    def data(self):
        """ @propert data function

            supports a class decorator @property that is used for getting the
            instance specific datafame. The data must be a pyspark dataframe
            and if it is not one, the function will try to convert the to a 
            pyspark dataframe.

            return self._data (pyspark dataframe)
        """

        __s_fn_id__ = f"{self.__name__} function <@property data>"

        try:
            if not isinstance(self._data,DataFrame):
                self._data = clsSpark.session.createDataFrame(self._data)
            if self._data.count() <= 0:
                raise ValueError("No records found in data") 

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data

    @data.setter
    def data(self,data):
        """ @data.setter function

            supports the class propert for setting the instance specific data. 
            The data must not be None-Type and must be a pyspark dataframe.

            return self._data (pyspark dataframe)
        """

        __s_fn_id__ = f"{self.__name__} function <@data.setter>"

        try:
            if data is None:
                raise AttributeError("Dataset cannot be empty")
            if not isinstance(data,DataFrame):
                self._data = clsSpark.session.createDataFrame(data)
                logger.warning("%s None spark data property set to %s",__s_fn_id__,type(self._data))
            else:
                self._data = data
                
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data


    ''' Function -- CONVERT COORDINATES --

            author: <nuwan.waidyanatha@rezgateway.com>
    '''

    ''' --- DECIMAL TO GEOMETRY --- ''' 
    @staticmethod
    def decimal_to_geometry(
        data:DataFrame = None,
        dec_lon_attr:str=None,
        dec_lat_attr:str=None,
        geom_type : str ='POINT',
#         deg_lon_attr:str=None,
#         deg_lat_attr:str=None,
        **kwargs,
    ) -> DataFrame:

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <decimal_to_geometry>"
        __def_geom_attr__ = "geom"

        def dec2point_wrapper(lon,lat):

            return Point(lon, lat).wkt

        try:
            if "GEOMATTR" not in kwargs.keys() \
                or "".join(kwargs['GEOMATTR'].split())=="":
                kwargs['GEOMATTR']=__def_geom_attr__
            if kwargs['GEOMATTR'] in data.columns:
                err_txt = "default column name {} ".format(kwargs['GEOMATTR'])
                err_txt+= "already in data columns. Define another column name"
                raise AttributeError(err_txt)
            logger.debug("%s New geometry column name is %s", 
                         __s_fn_id__, kwargs['GEOMATTR'].upper())

            if geom_type.upper() == 'POINT':
                ''' POINT '''
                convertUDF = F.udf(lambda lon, lat: dec2point_wrapper(lon, lat))
                data = data.withColumn(kwargs['GEOMATTR'], \
                            convertUDF(F.col(dec_lon_attr),F.col(dec_lat_attr))\
                                       .alias(kwargs['GEOMATTR']))
                if data is None and data.count()<=0:
                    raise RuntimeError("Failed convert column %s %s to POINT geometry; received empty %s" 
                                       % (dec_lon_attr.upper(), dec_lat_attr.upper(), type(data)))
                logger.debug("%s augmented %s column to data with %d rows", 
                             __s_fn_id__, kwargs['GEOMATTR'].upper(), data.count())
            else:
                raise AttributeError("Unrecognized deometry type %s" % geom_type.upper())

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return data

    ''' --- DECIMAL TO DEGREE --- ''' 
    @staticmethod
    def decimal_to_degree(
        data:DataFrame = None,
        dec_lon_attr:str=None,
        dec_lat_attr:str=None,
#         deg_lon_attr:str=None,
#         deg_lat_attr:str=None,
        **kwargs,
    ) -> DataFrame:

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <decimal_to_degree>"
        __def_deg_pefix__ = "degrees"

        def dec2deg_wrapper(deg, col_type='lat'):
            """
            """
#             try:
            decimals, number = math.modf(deg)
            d = int(number)
            m = int(decimals * 60)
            s = (deg - d - m / 60) * 3600.00
            compass = {
                'lat': ('N','S'),
                'lon': ('E','W')
            }
            compass_str = compass[col_type][0 if d >= 0 else 1]

#             except Exception as err:
#                 logger.error("%s %s \n",__s_fn_id__, err)
#                 logger.debug(traceback.format_exc())
#                 print("[Error]"+__s_fn_id__, err)
                
            return '{}º{}\'{:.2f}"{}'.format(abs(d), abs(m), abs(s), compass_str)

        try:
            if "DEGLONATTR" not in kwargs.keys() \
                or "".join(kwargs['DEGLONATTR'].split())=="":
                kwargs['DEGLONATTR']="_".join([__def_deg_pefix__,dec_lon_attr]).replace(" ","_")
            if kwargs['DEGLONATTR'] in data.columns:
                err_txt = "default column name {} ".format(kwargs['DEGLONATTR'])
                err_txt+= "already in data columns. Define another column name"
                raise AttributeError(err_txt)
            logger.debug("%s New degree longitude column name is %s", 
                         __s_fn_id__, kwargs['DEGLONATTR'].upper())
            if "DEGLATATTR" not in kwargs.keys() \
                or "".join(kwargs['DEGLATATTR'].split())=="":
                kwargs['DEGLATATTR']="_".join([__def_deg_pefix__,dec_lat_attr]).replace(" ","_")
            if kwargs['DEGLATATTR'] in data.columns:
                err_txt = "default column name {} ".format(kwargs['DEGLATATTR'])
                err_txt+= "already a data column. Define another column name"
                raise AttributeError(err_txt)
            logger.debug("%s New degree lattitude column name is %s",
                         __s_fn_id__, kwargs['DEGLATATTR'].upper())
            ''' LONGITUDE '''
            convertUDF = F.udf(lambda z: dec2deg_wrapper(z,'lon'))
            data = data.withColumn(kwargs['DEGLONATTR'], \
                                    convertUDF(
                                        F.col(dec_lon_attr))\
                                               .alias(kwargs['DEGLONATTR']))
            if data is None and data.count()<=0:
                raise RuntimeError("Failed convert lon column %s to decimal; received empty %s" 
                                   % (dec_lon_attr.upper(), type(data)))
            logger.debug("%s augmented %s degree colum to data with %d rows", 
                         __s_fn_id__, kwargs['DEGLONATTR'].upper(), data.count())
            ''' LATTITUDE '''
            convertUDF = F.udf(lambda z: dec2deg_wrapper(z,'lat'))
            data = data.withColumn(kwargs['DEGLATATTR'], \
                                    convertUDF(
                                        F.col(dec_lat_attr))\
                                               .alias(kwargs['DEGLATATTR']))
            if data is None and data.count()<=0:
                raise RuntimeError("Failed convert lat column %s to decimal; received empty %s" 
                                   % (dec_lat_attr.upper(), type(data)))
            logger.debug("%s augmented %s degree colum to data with %d rows", 
                         __s_fn_id__, kwargs['DEGLATATTR'].upper(), data.count())

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return data

    def convert_coordinates(
        self,
        data : DataFrame= None,  # non empty dataframe with coordinates columns
        from_to : tuple = None,  # ordered pair defines convert from to method
        coord_cols:dict = None,  # x, y, z, or geometry column names to apply convertion
        **kwargs,  # additional behaviour changing key value pairs
    ) -> DataFrame:
        """
        Description:
            For the given columns arguments, the method will apply the coordinte convertions
            on the defined columns from standard to the required to standard
        Attributes :
            * data (DataFrame) non empty dataframe with coordinates columns
            * from_to (tuple)  ordered pair defines convert from to method
            * columns (str)  column names to apply convertion
            * kwargs (dict) additional behaviour changing key value pairs
        Returns :
            data (DataFrame) augmented with the new coordinate columns
        Exceptions:
            * data is None or empty raise AttributeError; abort
            * from_to tuple values do not agree with predefined values raise ValueError
                try to detect and convert to default geometry format
            * columns arguments are undefined raise AttributeError; try to detect columns
                and apply defined or default convertion
        """

        __s_fn_id__ = f"{self.__name__} function <convert_coordinates>"

        try:
            ''' validate data and attributes '''
            self.data = data
            if self._data.count()<=0:
                raise AttributeError("Invalid empty %s" % type(self._data))
            ''' validate convertion tuple '''
            if from_to[0].lower() in ['decimal','degrees'] and 2>len(coord_cols)>2:
                raise AttributeError("Expecting 2 column names but got %d" % len(coord_cols))
            if from_to[0].lower() in ['geometry'] and 1>len(coord_cols)>1:
                raise AttributeError("Expecting 1 column names but got %d" % len(coord_cols))
                
            ''' --- from DECIMAL --- '''
            if from_to[0].lower() == 'decimal':
                ''' validate columns '''
                if "LONGITUDE" not in coord_cols.keys() \
                    or "LATITUDE" not in coord_cols.keys():
                    raise AttributeError("The dictionary with LONGITUDE and LATITUDE must be defined")
                if "COLATTR" not in coord_cols['LONGITUDE'].keys() \
                    or "".join(coord_cols['LONGITUDE']['COLATTR'].split())=="":
                    raise AttributeError("Could not find a proper COLATTR for LONGITUDE")
                if "COLATTR" not in coord_cols['LATITUDE'].keys() \
                    or "".join(coord_cols['LATITUDE']['COLATTR'].split())=="":
                    raise AttributeError("Could not find a proper COLATTR for LATITUDE")
                logger.debug("%s Attribute validated for %s input columns", 
                             __s_fn_id__, from_to[0].upper())

                if from_to[1].lower() == 'degrees':
                    ''' convert to DEGREES '''
                    self._data = dataWorkLoads.decimal_to_degree(
                        data = self._data,
                        dec_lon_attr = coord_cols['LONGITUDE']['COLATTR'],
                        dec_lat_attr = coord_cols['LATITUDE']['COLATTR'],
                        **kwargs,
                    )
                    if self._data is None or self._data.count()<=0:
                        raise RuntimeError("Failed %s to %s convertion; returned empty %s" 
                                           % (from_to[0].upper(), from_to[1].upper(), 
                                              type(self._data)))
                    logger.debug("%s Successfully converted %d rows from %s to %s",
                                 __s_fn_id__, self._data.count(), 
                                 from_to[0].upper(), from_to[1].upper())

                elif from_to[1].lower() == 'geometry':
                    ''' convert to GEOMETRY '''
                    self._data = dataWorkLoads.decimal_to_geometry(
                        data = self._data,
                        dec_lon_attr = coord_cols['LONGITUDE']['COLATTR'],
                        dec_lat_attr = coord_cols['LATITUDE']['COLATTR'],
                        geom_type = 'POINT',
                        **kwargs,
                    )
                    if self._data is None or self._data.count()<=0:
                        raise RuntimeError("Failed %s to %s convertion; returned empty %s" 
                                           % (from_to[0].upper(), from_to[1].upper(), 
                                              type(self._data)))
                    logger.debug("%s Successfully converted %d rows from %s to %s",
                                 __s_fn_id__, self._data.count(), 
                                 from_to[0].upper(), from_to[1].upper())

                else:
                    raise ValueError("Invalid GIS coodinate type %s" % (from_to[1].upper()))

                logger.debug("%s completed without errors", __s_fn_id__)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error] "+__s_fn_id__, err)

        return self._data
