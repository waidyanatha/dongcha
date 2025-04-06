#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "__propAttr__"
__package__= "timeseries"
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
    from dotenv import load_dotenv
    load_dotenv()
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
        self._realm= realm #None
        self._realmList = [
            'TIMESERIES', # 
            'STATISTICS',   # 
            'FORECAST', # 
        ]
        self._session = None   # spark session property
        self._data = None    # dataframe property
        self._startDT = None
        self._endDT = None
        self._dtAttr = None  # timestamp property
        self._partAttr=None  # partitionBy column name
        self._winSpecUnits = ['MINUTE','HOUR','DAY']
        self._winSpec = {
            'LENGTH':7,   # set default value to 7 days
            'UNIT':'DAY'  # rolling window
        }

        global pkgConf  # this package configparser class instance
        global appConf  # configparser class instance
        global logger   # rezaware logger class instance
#         global clsSDB   # etl loader sparkRDB class instance

        __s_fn_id__ = f"{self.__name__} function <__init__>"
        
        try:
            self.cwd=os.path.dirname(__file__)
            pkgConf = configparser.ConfigParser()
            pkgConf.read(os.path.join(self.cwd,__ini_fname__))

            self._projHome = pkgConf.get("CWDS","PROJECT")
            sys.path.insert(1,self._projHome)
            
            ''' initialize util class to use common functions '''
            from rezaware.modules.lib.spark import execSession as session
            clsSpark = session.Spawn(desc=self.__desc__)

            ''' innitialize the logger '''
            from rezaware.utils import Logger as logs
            logger = logs.get_logger(
                cwd=self._projHome,
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

    ''' Function
            name: data @property and @setter functions
            parameters:

            procedure: uses the clsSparkWL data@setter method to convert the data
                        to a pyspark DataFrame
    The properties implement
    * data [rdd] - in the form of a pyspark dataframe and any other dataframe will be converted
            into a resilient distributed dataset (rdd). However, the processed data is returned
            in the original dtype
    * datetimeAttr [str/int] - tells the class which of the columns is to be considered as the
            datetime of the timeseries. If unspecified, then the first detected datetime column
            is considered

            return self._data

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    @property
    def session(self):
        """
        Description:

        Atributes:

        Returns:
            self._session (SparkSession)
        """

        __s_fn_id__ = f"{self.__name__} function <@property session>"

        try:
            if self._session is None or self._session=={}:
                clsSpark.session={}
                
#                 from utils.modules.etl.load import sparkDBwls as spark
#                 clsSparkWL = spark.SparkWorkLoads(desc=self.__desc__)
#                 if clsSparkWL.session is None:
#                     clsSparkWL.session = {}
                self._session = clsSpark.session

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._session

    @session.setter
    def session(self,session):
        """
        Description:

        Atributes:

        Returns:
            self._session (SparkSession)
        """
        
        __s_fn_id__ = f"{self.__name__} function <@session.setter>"

        try:
            ''' TODO validate if active spark session '''
            if not session is None:
                self._session = session

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._session

    ''' --- DATA --- '''
    @property
    def data(self) -> DataFrame:
        """
        Description:

        Atributes:

        Returns:
            self._data (DataFrame)
        """
        
        __s_fn_id__ = f"{self.__name__} function <@propert data>"
        
        try:
            if self._data is None:
                raise ValueError("%s Data is of NoneType; cannot be used in any computations"
                                 % __s_fn_id__)
            if not isinstance(self._data,DataFrame):
                self._data = self.session.createDataFrame(self._data)
                logger.debug("%s Data converted to pyspark %s dtype",
                             __s_fn_id__,type(self._data))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data

    @data.setter
#     def data(self,data:DataFrame=clsSparkWL.spark_session.sparkContext.emptyRDD()):
    def data(self,data) -> DataFrame:
        """
        Description:

        Atributes:

        Returns:
            self._data (DataFrame)
        """

        __s_fn_id__ = f"{self.__name__} function <@data.setter>"
#         clsSparkWL.data=data
#         self._data = clsSparkWL.data

        try:
            if data is None:
                raise AttributeError("Dataset cannot be empty")

            if not isinstance(data,DataFrame):
                self._data = self.session.createDataFrame(data)
                logger.debug("%s Data of dtype %s converted to pyspark DataFrame",
                             __s_fn_id__,type(data))
            else:
                self._data = data
                logger.debug("%s Class property data is a pyspark DataFrame",__s_fn_id__)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data

    ''' --- START DATETIME --- '''
    @property
    def startDateTime(self):
        """
        Description:

        Atributes:

        Returns:
            self._startDateTime (timestamp)
        """

        __s_fn_id__ = f"{self.__name__} function <@endDateTime.setter>"

        try:
            if self._startDT is None and not self.data.isEmpty():
                self._startDT = self.data.select(F.min(self.datetimeAttr)).collect()

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._startDT

    @startDateTime.setter
    def startDateTime(self,start_date_time=None):
        """
        Description:
            Define the overall timeseries window boundaries. If undefined, the default
            will consider the Min datetime as the startDateTime and the Max datetime 
            as the endDateTime
        Atributes:
            start_date_time (timestamp)
        Returns:
            self._startDateTime (timestamp)
        """

        __s_fn_id__ = f"{self.__name__} function <@startDateTime.setter>"

        try:
            if start_date_time is None or not isinstance(start_date_time,datetime):
                raise AttributeError("Invalid datetime input parameter")
            self._startDT = start_date_time

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._startDT

    ''' --- END DATETIME --- '''
    @property
    def endDateTime(self):
        """
        Description:

        Atributes:

        Returns:
            self._endDateTime (timestamp)
        """

        __s_fn_id__ = f"{self.__name__} function <@endDateTime.setter>"

        try:
            if self._endDT is None and not self.data.isEmpty():
                self._endDT = self.data.select(F.max(self.datetimeAttr)).collect()

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._endDT

    @endDateTime.setter
    def endDateTime(self,end_date_time:datetime=datetime.now()):
        """
        Description:

        Atributes:

        Returns:
            self._endDateTime (timestamp)
        """

        __s_fn_id__ = f"{self.__name__} function <@endDateTime.setter>"

        try:
            if end_date_time is None or not isinstance(end_date_time,datetime):
                raise AttributeError("Invalid datetime input parameter")
            self._endDT = start_date_time

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._endDT

    ''' --- DATETIME ATTRIBUTE --- '''
    @property
    def datetimeAttr(self):
        """
        Description:

        Atributes:

        Returns:
            self._datetimeAttr (timestamp)
        """

        __s_fn_id__ = f"{self.__name__} function <@property datetimeAttr>"

        try:
            if self._dtAttr is None: # or \
                ''' TODO validate attribute; if invalid attempt 
                to find a datetime column in dataframe'''
#                 self.data.schema[self._dtAttr].dataType !=DateType or \
#                 self.data.schema[self._dtAttr].dataType !=TimestampType:# or \
#                 self.data.schema[self._dtAttr].dataType !=date or \
#                 self.data.schema[self._dtAttr].dataType !=datetime:

                logger.debug("%s The datetimeAttr was not explicitly set as a valid "+ \
                            "DateType or TimestampType and will try to set the first"+ \
                            "found valid column %s",__s_fn_id__,self.data.dtypes)
#                 print(self.data.dtypes)
                _dt_attr_list = next(
                    (x for x, y in self.data.dtypes 
                     if y==DateType or y==TimestampType),# or y==date or y==datetime),
                    None)
                print(_dt_attr_list)
                if _dt_attr_list is None:
                    raise AttributeError("Could not locate a valid datetime attribute "+ \
                                         "in the dataset with columns %s" \
                                         % str(self.data.dtypes))
                else:
                    self._dtAttr = _dt_attr_list[0]

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._dtAttr

    @datetimeAttr.setter
    def datetimeAttr(self,date_time_attr:str=''):
        """
        Description:

        Atributes:

        Returns:
            self._datetimeAttr (timestamp)
        """

        __s_fn_id__ = f"{self.__name__} function <@datetimeAttr.setter>"

        try:
            if self.data is None or self.data.count()<=0:
                raise ValueError("The dataset property must be defined before setting the datetimeAttr")
            if "".join(date_time_attr.split())!="" or \
                    self.data.filter(F.col(date_time_attr).cast("Timestamp").isNotNull()).count()>0:
                self._dtAttr = date_time_attr
            else:
                raise AttributeError("The datetimeAttribute cannot be an empty string")
            ''' cast the datetime attr to a timestamp '''
            self.data = self.data.withColumn(self._dtAttr,F.to_timestamp(self._dtAttr))
            logger.debug("%s Cast column %s to timestamp",__s_fn_id__,self._dtAttr)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._dtAttr

    @property
    def partitionAttr(self) -> str:
        """
        Description:
            Gets and returns the partition column name
        Attribute:

        Returns:
            self._partAttr
        """
        
        __s_fn_id__ = f"{self.__name__} function <@property partitionAttr>"
        
        try:
            if self._partAttr is None:
                logger.warning("%s No partition column set",__s_fn_id__)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._partAttr

    @partitionAttr.setter
    def partitionAttr(self,partition_attr) -> str:
        """
        Description:
            Gets and returns the partition column name
        Attribute:

        Returns:
            self._partAttr
        """

        __s_fn_id__ = f"{self.__name__} function <@partitionAttr.setter>"

        try:
            if partition_attr is not None or "".join(partition_attr.split())!="":
                self._partAttr=partition_attr
                logger.debug("%s Set partition column attribute name as %s",
                             __s_fn_id__,self._partAttr)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._partAttr


    ''' --- WINDOW SPECIFICATION --- '''
    # @property
    # def windowSpec(self) -> int:
    @property
    def winSpec(self) -> int:
        """
        Description:
            If the property is None, will set to a default 7 DAY window specification.
            Requires a valid datetimeAttr.
        Atributes:
            None
        Returns:
            self._winSpec (int)
        """

        __s_fn_id__ = f"{self.__name__} function <@property windowSpec>"
        __win_len__ = 7

        try:
            ''' function to calculate number of seconds from number of days '''
            if self._winSpec is None:
                days = lambda i: i * 86400
                self._winSpec = Window \
                    .orderBy(F.col(self.datetimeAttr).cast('long')) \
                    .rangeBetween(-days(__win_len__),0)
                logger.debug("%s Class property winSpec was not explicitly set "+\
                             "Setting to $d DAY",__s_fn_id__,__win_len__)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._winSpec

    # @windowSpec.setter
    # def windowSpec(self,window_spec:dict={}) -> int:
    @winSpec.setter
    def winSpec(self,window_spec:dict={}) -> int:
        """
        Description:
            With two key value pairs that defines an integer value of the window
            lengthe (LENGTH) and the unit of measure (UNIT) that can only be 
            [MINUTE, HOUR, DAY]. The datetimeAttr is required to 
        Atributes:
            window_spec(dict) - LENGTH (int) and UNIT (str) MINUTE, HOUR, DAY 
        Returns:
            self._winSpec (int)
        """

        __s_fn_id__ = f"{self.__name__} function <@windowSpec.setter>"

        try:
            if not len(window_spec)>0 or \
                not 'LENGTH' in window_spec.keys() or \
                not 'UNIT' in window_spec.keys():
                raise AttributeError("Missing one or more keys LENGTH and UNIT input parameters")
            if not (isinstance(window_spec['LENGTH'],int) or window_spec['LENGTH'] > 0):
                raise AttributeError("The window_spec LENGTH must be of dtype int > 0")
            if window_spec['UNIT'] not in self._winSpecUnits:
                raise AttributeError("Invalid rolling window UNIT %s must be %s" 
                                     %(window_spec['UNIT'],
                                       self._winSpecUnits))
            if self.datetimeAttr is None:
                raise AttributeError("A valid datetimeAttr property must be specified to proceed")
            
            ''' function to calculate number of seconds from number of days '''
            if window_spec['UNIT'] == "DAY":
                _time_attr = lambda i: i * 86400
            elif window_spec['UNIT'] == "HOUR":
                _time_attr = lambda i: i * 3600
            elif window_spec['UNIT'] == "MINUTE":
                _time_attr = lambda i: i * 60
            else:
                raise RuntimeError("Something was wrong")

            if self.partitionAttr:
                self._winSpec = Window \
                    .partitionBy(self.partitionAttr) \
                    .orderBy(F.col(self.datetimeAttr).cast('long')) \
                    .rangeBetween(-_time_attr(window_spec['LENGTH']),0)
            else:
                self._winSpec = Window \
                    .orderBy(F.col(self.datetimeAttr).cast('long')) \
                    .rangeBetween(-_time_attr(window_spec['LENGTH']),0)

            logger.debug("%s WinSpec set to %s ",__s_fn_id__,str(self._winSpec))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._winSpec

