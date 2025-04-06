#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "__propAttr__"
__package__ = "mcmc"
__module__ = "ml"
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
    import re
    ''' SPORK PACKAGES '''
    import findspark
    findspark.init()
    from pyspark.sql import functions as F
    from pyspark.sql import DataFrame
    from pyspark.sql.types import *
    from pyspark.sql.window import Window
    from pymongo import MongoClient
    ''' DATATIME PACKAGES '''
    from datetime import datetime, date, timedelta

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))

'''
    CLASS configure the master property details, groups, reference, and geom entities

    Contributors:
        * farmraider@protonmail.com

    Resources:

'''

class properties():

    ''' Function --- INIT ---

            author: <farmraider@protonmail.com>
    '''
    def __init__(
        self,
        realm:str=None,
        desc :str=None,
        **kwargs):
        """
        Decription:
            Initializes the features: class property attributes, app configurations, 
                logger function, data store directory paths, and global classes 
        Attributes:
            desc (str) identify the specific instantiation and purpose
        Returns:
            None
        """

        self.__name__ = __name__
        self.__package__ = __package__
        self.__module__ = __module__
        self.__app__ = __app__
        self.__ini_fname__ = __ini_fname__
        self.__conf_fname__ = __conf_fname__
        if desc is None or "".join(desc.split())=="":
            self.__desc__ = " ".join([self.__app__,self.__module__,
                                      self.__package__,self.__name__])
        else:
            self.__desc__ = desc

        self._data = None
        self._checks=None
        self._realm=realm #None
        self._realmList = [
            'MCMC', # Markov Chain Monte Carlo
        ]
        self._checkList = [
            "all",
            "stationarity",
            "data volume",
            "representativeness",
            "dimensionality",
            "correlation structure",
            "multimodality",
            "boundary conditions",
            "mixing properties",
            "prior knowledge",
            "noise characteristics",
        ]

        global pkgConf  # this package configparser class instance
        global appConf  # configparser class instance
        global logger   # rezaware logger class instance
#         global clsSDB   # etl loader sparkRDB class instance

        __s_fn_id__ = f"{self.__name__} function <__init__>"
        
        try:
            self.cwd=os.path.dirname(__file__)
            pkgConf = configparser.ConfigParser()
            pkgConf.read(os.path.join(self.cwd,__ini_fname__))

            self._rezHome = pkgConf.get("CWDS","PROJECT")
            sys.path.insert(1,self._rezHome)
            
            ''' innitialize the logger '''
            from rezaware.utils import Logger as logs
            logger = logs.get_logger(
                cwd=self._rezHome,
                app=self.__app__, 
                module=self.__module__,
                package=self.__package__,
                ini_file=self.__ini_fname__)

            ''' set a new logger section '''
            logger.info('########################################################')
            logger.info("%s Class",self.__name__)

            ''' Set the wrangler root directory '''
#             self.pckgDir = pkgConf.get("CWDS",self.__package__)
            self._appDir = pkgConf.get("CWDS",self.__app__)
            ''' get the path to the input and output data '''
#             self.dataDir = pkgConf.get("CWDS","DATA")
            appConf = configparser.ConfigParser()
            appConf.read(os.path.join(self._appDir, self.__conf_fname__))

            logger.debug("%s initialization for %s module package %s %s done.\nStart workloads: %s."
                         %(self.__app__,
                           self.__module__,
                           self.__package__,
                           self.__name__,
                           self.__desc__))

            print("%s Class initialization complete" % self.__name__)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None


    ''' --- REALM --- '''
    @property
    def realm(self) -> DataFrame:
        """
        Description:
            realm @property and @setter functions. make sure it is a valid realm
        Attributes:
            realm in @setter will instantiate self._realm  
        Returns :
            self._realm (str) 
        """

        __s_fn_id__ = f"{self.__name__} function <@property realm>"

        try:
            if self._realm.upper() not in self._realmList:
                raise KeyError("Invalid realm; must be one of %s"
                                 % self._realmList)
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._realm.upper()

    @realm.setter
    def realm(self,realm) -> DataFrame:

        __s_fn_id__ = f"{self.__name__} function <realm.@setter>"

        try:
            if realm.upper() not in self._realmList:
                raise KeyError("Invalid %s realm; must be one of %s"
                                 % (type(realm), self._realmList))

            self._realm = realm.upper()

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._realm

    ''' Function --- DATA SETTER & GETTER ---

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    @property
    def data(self):

        __s_fn_id__ = f"{self.__name__} function <@property data>"

        try:
            ''' validate property value '''
            if not isinstance(self._data,DataFrame):
                self._data = self.session.createDataFrame(self._data)
            if self._data.count() <= 0:
                raise ValueError("No records found in data") 
                
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data

    @data.setter
    def data(self,data):

        __s_fn_id__ = f"{self.__name__} function <@data.setter>"

        try:
            ''' validate property value '''
            if data is None:
                raise AttributeError("Dataset cannot be empty")
            if not isinstance(data,DataFrame):
                self._data = self.session.createDataFrame(data)
                logger.debug("%s %s dtype convereted to %s with %d rows %d columns",
                         __s_fn_id__,type(data),type(self._data),
                         self._data.count(),len(self._data.columns))
            else:
                self._data = data

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data

    ''' --- CHECKS --- '''
    @property
    def checks(self) -> DataFrame:
        """
        Description:
            
        Attributes:
            
        Returns :
            self._checks (list) 
        """

        __s_fn_id__ = f"{self.__name__} function <@property checks>"

        try:
            if self._checks is None or not isinstance(self._checks, list)\
                or len(self._checks)<=0:
                raise ValueError("No checks list to share")

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._checks

    @checks.setter
    def checks(self,checks) -> DataFrame:

        __s_fn_id__ = f"{self.__name__} function <checks.@setter>"

        try:
            ''' validate checks '''
            checks = list(set([x.lower() for x in checks]).intersection(set(self._checkList)))
            if checks is None or len(checks)<=0:
                raise AttributeError("Invalid list of checks, must be: %s" 
                                     % ", ".join(self._checkList))
            else:
                self._checks = checks
            if "all" in checks:
                self._checks = self._checkList
            logger.debug("%s %d checks: %s qualify for processing", 
                         __s_fn_id__, len(self._checks), ", ".join(self._checks).upper())

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._checks


