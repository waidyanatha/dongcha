#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "isoAlphaCodes"
__package__ = "transform"
__module__ = "etl"
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
    import pycountry

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))

'''
    CLASS transform between ISO Aplha-2 and Alpha-3 country, city, area, and continent codes.

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

    ''' Function -- GET ISO 3166 --

            author: <nuwan.waidyanatha@rezgateway.com>
            
            TODO: complete subdivision when data is available
    '''
    @staticmethod
    def subdivision_to_3166_2(
        data : DataFrame = None,
        subdiv_dict :list= None,
        alpha2_attr :str = None,
        **kwargs,
    ) -> DataFrame:
        """
        """

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <subdivision_to_3166_2>"
        __def_geom_attr__ = "geom"


        def subdiv_wrapper(alpha2):
            return pycountry.subdivisions.get(country_code=alpha2)[0]['code']

        try:
            for _subdiv_key, _subdiv_val in subdiv_dict:
                if _subdiv_val is None or "".join(_subdiv_val.split())=="":
                    _subdiv_val = "_".join([alpha2_attr,_subdiv_key])
                    if _subdiv_val in data.coulmns:
                        raise ValueError("%s already a dataframe column; specify in t_iso dict", 
                                         _subdiv_val)

                convertUDF = F.udf(lambda alpha2: subdiv_wrapper(alpha2))

                data = data.withColumn(return_attr, \
                                       F.when(F.col(country_attr).isNotNull() |
                                              ~F.col(country_attr).isin('NaN',''),
                                              convertUDF(F.col(country_attr)))\
                                       .otherwise(F.lit(None))\
                                       .alias(return_attr))

            if data is None and data.count()<=0:
                raise RuntimeError("Failed convert column %s to %s; received empty %s" 
                                   % (country_attr.upper(), alpha_code, type(data)))
            logger.debug("%s augmented %s column with %s data on %d rows", 
                         __s_fn_id__, return_attr.upper(), alpha_code, data.count())

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return data


    @staticmethod
    def country_to_3166_1(
        data : DataFrame = None,
        get_code : str = None,
        country_attr:str = None,
        return_attr :str = None,
        **kwargs,
    ) -> DataFrame:
        """
        """

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <country_to_3166_1>"

        def iso_3166_wrapper(country, get_code):

            _country = None
            _country = pycountry.countries.lookup(country)
            if _country is None:
                _country = pycountry.countries.search_fuzzy(country)[0]

            if _country is not None:
                if get_code == 'ALPHA2' and 'alpha_2' in dir(_country):
                    ret_val = _country.alpha_2
                elif get_code == 'ALPHA3' and 'alpha_3' in dir(_country):
                    ret_val = _country.alpha_3
                elif get_code == 'UN' and 'numeric' in dir(_country):
                    ret_val = _country.numeric
                else:
                    ret_val = None
            else:
                ret_val = None

            return ret_val

        try:
            convertUDF = F.udf(lambda country: iso_3166_wrapper(country, get_code))
            data = data.withColumn(return_attr, \
                                   F.when(F.col(country_attr).isNotNull() |
                                          ~F.col(country_attr).isin('NaN',''),
                                          convertUDF(F.col(country_attr)))\
                                   .otherwise(F.lit(None))\
                                   .alias(return_attr))

            if data is None and data.count()<=0:
                raise RuntimeError("Failed convert column %s to %s; received empty %s" 
                                   % (country_attr.upper(), get_code, type(data)))
            logger.debug("%s augmented %s column with %s data on %d rows", 
                         __s_fn_id__, return_attr.upper(), get_code, data.count())

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return data


    def get_iso_3166(
        self,
        data : any = None, # any dtype that can be converted to pyspark dataframe
        t_iso: dict=None,# dict of convertions and supporting values
        **kwargs,
    ) -> DataFrame:
        """
        Description:
            Takes the transformation inputs from the iso_dict and applies them to the data.
            A typical iso_dict must tell the country and/or city column names and the type
            of iso codes with return column name. The following example, converts 
            * country - column THECOUNRY to alpha3 with result in country_iso_3 and 
            * city - column OURCITY to alpha2 and alpha3 with results in city_iso_2, city_iso_3
            alpha = {"COUNTRY": {
                        "ATTRNAME" : 'THECOUNTRY',
                        "TRANSFORM" : {"ALPHA3" : 'country_iso_3'}},
                    "CITY" : {
                        "ATTRNAME"  : 'OURCITY',
                        "TRANSFORM" : {"ALPHA2" : 'city_iso_2', 'ALPHA3' : 'city_iso_3'}}}
        Attributes : 
            * data (any) dtype that can be converted to pyspark dataframe
            * t_iso (dict) convertions and supporting values (see example above)
        Returns : 
            self._data (DataFrame) with the new columns
        """

        __s_fn_id__ = f"{self.__name__} function <get_iso_3166>"
        
        __def_alpha_keys__ = ['COUNTRY', 'SUBDIV', 'CITY', 'CONTINENT']
        __def_trans_keys__ = ['ALPHA2','ALPHA3','UN', 'TYPE', 'CODE']

        try:
            self.data = data
            if self._data.count()<=0:
                raise AttributeError("Invalid empty %s" % type(self._data))

            ''' check if ATTRNAME and TRANSFORM are properly defined '''
            _remove_dict = {}
            for _alpha_key in t_iso.keys():
                if _alpha_key.upper() not in __def_alpha_keys__:
                    _err_msg = "invalid key {0} must be one of {1}"\
                                .format(_alpha_key.upper(),str(__def_alpha_keys__))
                    _remove_dict.update({_alpha_key:_err_msg})
                elif "ATTRNAME" not in t_iso[_alpha_key].keys() \
                    or "".join(t_iso[_alpha_key]['ATTRNAME'].split())=="":
                    _err_msg = "missing key ATTRNAME for {0}".format(_alpha_key.upper())
                    _remove_dict.update({_alpha_key:_err_msg})
                elif t_iso[_alpha_key]['ATTRNAME'] not in data.columns:
                    _err_msg = "{0} is not a valid column; did you mean {1}"\
                                .format(t_iso[_alpha_key]['ATTRNAME'].upper(),
                                        str(data.columns).upper())
                    _remove_dict.update({_alpha_key:_err_msg})
                elif "TRANSFORM" not in t_iso[_alpha_key].keys():
                    _err_msg = "missing key TRANSFORM for {0}".format(_alpha_key.upper())
                    _remove_dict.update({_alpha_key:_err_msg})
                elif [x for x in t_iso[_alpha_key]['TRANSFORM'].keys() 
                      if x.upper() not in __def_trans_keys__]:
                    _err_msg = "one or more invalid TRANSFORM keys {0} must be one of {1}"\
                                .format(str(t_iso[_alpha_key]['TRANSFORM'].keys()),
                                        str(__def_trans_keys__))
                    _remove_dict.update({_alpha_key:_err_msg})
                else:
                    logger.debug("%s verified ATTRNAME and TRANSFORM key/val pairs for %s",
                                 __s_fn_id__, _alpha_key.upper())

            for _del_key, _del_msg in _remove_dict.items():
                logger.error("%s removing key %s because %s", __s_fn_id__, _del_key, _del_msg)
                t_iso.pop(_del_key)

            if len(t_iso)<=0:
                raise AttributeError("Unspecified empty alpha %s; nothing to process" 
                                     % type(t_iso))
            logger.debug("%s alph dict has %d valid transformations to process", 
                         __s_fn_id__, len(t_iso))

            for _alpha_key in t_iso.keys():
                for _tran_key, _tran_val in t_iso[_alpha_key]['TRANSFORM'].items():
                    ''' ensure return column name '''
                    if _tran_val is None or "".join(_tran_val.split())=="":
                        _tran_val = "_".join([_alpha_key.lower(),_tran_key.lower()])

                    if _alpha_key.upper() == 'COUNTRY':
                        self._data = dataWorkLoads.country_to_3166_1(
                            data = self._data,
                            get_code = _tran_key.upper(),
                            country_attr=t_iso[_alpha_key]['ATTRNAME'],
                            return_attr =_tran_val,
                            **kwargs,
                        )
                    if _alpha_key.upper() == 'SUBDIV':
                        self._data = dataWorkLoads.subdivision_to_3166_2(
                            data = self._data,
                            get_code = _tran_key.upper(),
                            country_attr=t_iso[_alpha_key]['ATTRNAME'],
                            return_attr =_tran_val,
                            **kwargs,
                        )

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data


    ''' Function -- GET ISO 3166 --

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    @staticmethod
    def iso_to_country(
        data : DataFrame = None,
        code_type: str = None,
        iso_attr : str = None,
        get_code : str = None,
        return_attr :str = None,
        **kwargs,
    ) -> DataFrame:
        """
        """

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <iso_to_country>"


        def country_wrapper(iso_code, code_type = 'ALPHA3', get_code='NAME'):
            if code_type == 'ALPHA2':
                _country = pycountry.countries.get(alpha_2=iso_code)
            elif code_type == 'ALPHA3':
                _country = pycountry.countries.get(alpha_3=iso_code)
            elif code_type == 'UN':
                _country = pycountry.countries.get(numeric=iso_code)
            else:
                _country = None

            if _country is not None:
                if get_code == 'NAME' and 'name'  in dir(_country):
                    ret_val = _country.name
                elif get_code == 'OFFICIAL' and 'official_name'  in dir(_country):
                    ret_val = _country.official_name
                elif get_code == 'NUMERIC' and 'numeric'  in dir(_country):
                    ret_val = _country.numeric
                else:
                    ret_val = None
            else:
                ret_val = None

            return ret_val

        try:
            reverseUDF = F.udf(lambda iso_code: country_wrapper(iso_code,code_type,get_code))
            data = data.withColumn(return_attr, \
                                   F.when(F.col(iso_attr).isNotNull() |
                                          ~F.col(iso_attr).isin('NaN','','null'),
                                          reverseUDF(F.col(iso_attr)))\
                                   .otherwise(F.lit(None))\
                                   .alias(return_attr))

            if data is None and data.count()<=0:
                raise RuntimeError("Failed convert column %s to %s; received empty %s" 
                                   % (country_attr.upper(), code_type, type(data)))
            logger.debug("%s augmented %s column with %s data on %d rows", 
                         __s_fn_id__, return_attr.upper(), code_type, data.count())

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return data


    def reverse_iso_3166(
        self,
        data : any = None, # any dtype that can be converted to pyspark dataframe
        t_iso: dict=None,# dict of convertions and supporting values
        **kwargs,
    ) -> DataFrame:
        """
        Description:
            For a given data set that can be transformed to a pyspark dataframe, the
            identified iso-3166 columns with alpha2, alpha3, un, or subdivision codes
            will fetch the respective values like name, type, and so on.
        Attributes :
            * data (any) dtype that can be converted to pyspark dataframe
            * t_iso (dict) of convertions and supporting values
        Returns :
            self._data (DataFrame)  with the new columns
        """

        __s_fn_id__ = f"{self.__name__} function <get_iso_3166>"
        
        __def_get_keys__ = ['NAME', 'OFFICIAL', 'NUMERIC']
        __def_attr_keys__ = ['ALPHA2', 'ALPHA3', 'UN']
        __def_3166_keys__ = ['COUNTRY', 'SUBDIVISIN']

        try:
            self.data = data
            if self._data.count()<=0:
                raise AttributeError("Invalid empty %s" % type(self._data))

            ''' check if ATTRNAME and TRANSFORM are properly defined '''
            _remove_dict = {}
            for _3166_key in t_iso.keys():
                if _3166_key.upper() not in __def_3166_keys__:
                    _err_msg = "invalid 3166 key {0} must be one of {1}"\
                                .format(_3166_key.upper(),str(__def_3166_keys__))
                    _remove_dict.update({_3166_key:_err_msg})

                elif len([x for x in t_iso[_3166_key].keys() if x in __def_attr_keys__])<=0:
                    _err_msg = "missing attribute key must be one of {0}"\
                                .format(str(__def_attr_keys__).upper())
                    _remove_dict.update({_3166_key:_err_msg})

                elif "TRANSFORM" not in t_iso[_3166_key].keys():
                    _err_msg = "missing key TRANSFORM for {0}".format(_3166_key.upper())
                    _remove_dict.update({_3166_key:_err_msg})

                elif len([x for x in t_iso[_3166_key]['TRANSFORM'].keys() 
                      if x.upper() not in __def_get_keys__]) > 0:
                    _err_msg = "one or more invalid TRANSFORM keys {0} did you mean one of {1}"\
                                .format(str(t_iso[_3166_key]['TRANSFORM'].keys()),
                                        str(__def_get_keys__))
                    _remove_dict.update({_3166_key:_err_msg})
                else:
                    logger.debug("%s verified ATTRNAME and TRANSFORM key/val pairs for %s",
                                 __s_fn_id__, _3166_key.upper())

            for _del_key, _del_msg in _remove_dict.items():
                logger.error("%s removing key %s because %s", __s_fn_id__, _del_key, _del_msg)
                t_iso.pop(_del_key)

            if len(t_iso)<=0:
                raise AttributeError("Unspecified empty t_iso %s; nothing to process" 
                                     % type(t_iso))
            logger.debug("%s alph dict has %d valid transformations to process", 
                         __s_fn_id__, len(t_iso))

            ''' now process the valid t_iso '''
            for _3166_key in t_iso.keys():
                ''' validate column name if in dataset '''
                _col_name = [v for k,v in t_iso[_3166_key].items() if k in __def_attr_keys__][0]
                if _col_name not in data.columns:
                    raise ValueError("Column %s deos not match any dataset column names %s" % 
                                     (_col_name.upper(),str(data.columns).upper()))
                for _tran_key, _tran_val in t_iso[_3166_key]['TRANSFORM'].items():
                    ''' ensure return column name '''
                    if _tran_val is None or "".join(_tran_val.split())=="":
                        _tran_val = "_".join([_3166_key.lower(),_tran_key.lower()])
                        logger.warning("%s unspecified %s return colum set to default %s",
                                       __s_fn_id__, _tran_key.upper(), _tran_val.upper())
                    if _tran_val in data.columns:
                        raise ValueError("%s already a dataset column name; define a nother for %s" 
                                         % (_tran_val.upper(), _tran_key.upper()))

                    if _3166_key.upper() == 'COUNTRY':
                        _code_type = [k for k in t_iso[_3166_key].keys() if k in __def_attr_keys__][0]
                        self._data = dataWorkLoads.iso_to_country(
                            data = self._data,
                            code_type= _code_type,
                            iso_attr = _col_name,
                            get_code = _tran_key.upper(),
                            return_attr =_tran_val,
                            **kwargs,
                        )
#                     if _alpha_key.upper() == 'SUBDIV':
#                         self._data = dataWorkLoads.iso_subdiv(
#                             data = self._data,
#                             alpha_code = _tran_key.upper(),
#                             country_attr=t_iso[_alpha_key]['ATTRNAME'],
#                             return_attr =_tran_val,
#                             **kwargs,
#                         )
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data

