#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "sparkFile"
__package__ = "loader"
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
    from pyspark.sql.functions import lit, current_timestamp
    from pyspark.sql import DataFrame
    from google.cloud import storage   # handles GCS reads and writes
    import pandas as pd
    import numpy as np
    import json
    import boto3   # handling AWS S3
    from botocore.client import ClientError

    from rezaware.modules.etl.loader import __propAttr__ as attr

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))

'''
    CLASS create, update, and migrate data from and to files using pyspark. Specifically,
        handles file read/write with:
        * Local File Systems folders
        * Amazon cloud AWS S3 buckets
        * Google Cloud Storate (GCS) objects

    Contributors:
        * nuwan.waidyanatha@rezgateway.com
        * samana.thetha@gmail.com

    Resources:
        https://computingforgeeks.com/how-to-install-apache-spark-on-ubuntu-debian/
'''
class dataWorkLoads(attr.properties):
    ''' Function --- INIT ---
    
        author: <nuwan.waidyanatha@rezgateway.com>

    '''
    def __init__(
        self, 
        desc : str="spark workloads",   # identifier for the instances
        store_mode:str=None,
        store_root:str=None,
        jar_dir : str =None,
        **kwargs,
    ):
        """
        Description:
        Attributes:
        Returns:
            None
        Exceptions:
        """

        ''' instantiate property attributes '''
        super().__init__(
#             desc=self.__desc__,
            realm="FILES"
        )
        self.__name__ = __name__
        self.__package__ = __package__
        self.__module__ = __module__
        self.__app__ = __app__
        self.__ini_fname__ = __ini_fname__
        self.__conf_fname__ = __conf_fname__
        self.__desc__ = desc

#         ''' instantiate property attributes '''
#         super().__init__(
#             desc=self.__desc__,
#             realm="FILES"
#         )

#         self._dType = None
#         self._dTypeList = [
#             'RDD',     # spark resilient distributed dataset
#             'SDF',     # spark DataFrame
#             'PANDAS',  # pandas dataframe
#             'ARRAY',   # numpy array
#             'DICT',    # data dictionary
#         ]

        ''' default values '''
        self._storeModeList = [
            'local-fs',   # local hard drive on personal computer
            'aws-s3-bucket',   # cloud amazon AWS S3 Bucket storage
            'google-storage',  # google cloud storage buckets
        ]
        self._asTypeList = [
            'str',   # text string ""
            'list',  # list of values []
            'dict',  # dictionary {}
            'array', # numpy array ()
            'set',   # set of values ()
            'pandas', # pandas dataframe
            'spark',  # spark dataframe
        ]   # list of data types to convert content to
        self._rwFormatTypes = [
            'csv',   # comma separated value
            'json',  # Javascript object notation
            # 'text',  # text file
            'txt',  # text file
        ]

        ''' Initialize property var to hold the data '''
#         self._data = None
        self._storeMode =store_mode
        self._storeRoot =store_root  # holds the data root path or bucket name
        self._folderPath=None  # property attr for the set/get folder path
        self._asType = None

        ''' Initialize spark session parameters '''
#         self._homeDir= None   # spark $SPARK_HOME dir path property required for sessions
#         self._binDir = None   # spark $SPARK_BIN dir path property required for sessions
#         self._config = None   # spark .conf option property
        self._jarDir = jar_dir   # spark JAR files dir path property
#         self._appName= None   # spark appName property with a valid string
#         self._master = None   # spark local[*], meso, or yarn property 
        self._rwFormat=None   # spark read/write formats (jdbc, csv,json, text) property
#         self._session =None   # spark session is set based on the storeMode property
        self._context =None   # spark context is set to support Hadoop & authentication
#         self._saveMode=None   # spark write append/overwrite save mode property

        ''' initiate to load app.cfg data '''
        global logger
        global pkgConf
        global appConf

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

            ''' Set the wrangler root directory '''
            self.pckgDir = pkgConf.get("CWDS",self.__package__)
            self.appDir = pkgConf.get("CWDS",self.__app__)
            ''' get the path to the input and output data '''
            self.dataDir = pkgConf.get("CWDS","DATA")

            appConf = configparser.ConfigParser()
            appConf.read(os.path.join(self.appDir, self.__conf_fname__))

            ''' get tmp storage location '''
            self.tmpDIR = None
            if "WRITE_TO_FILE" in kwargs.keys():
                self.tmpDIR = os.path.join(self.dataDir,"tmp/")
                if not os.path.exists(self.tmpDIR):
                    os.makedirs(self.tmpDIR)

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


#     ''' Function -- DATA --
#             TODO: 
#             author: <nuwan.waidyanatha@rezgateway.com>
#     '''
#     @property
#     def data(self):
#         """ @propert data function

#             supports a class decorator @property that is used for getting the
#             instance specific datafame. The data must be a pyspark dataframe
#             and if it is not one, the function will try to convert the to a 
#             pyspark dataframe.

#             return self._data (pyspark dataframe)
#         """

#         __s_fn_id__ = f"{self.__name__} function <@property data>"

#         try:
#             if not isinstance(self._data,DataFrame):
#                 self._data = self.session.createDataFrame(self._data)
#             if self._data.count() <= 0:
#                 raise ValueError("No records found in data") 

#         except Exception as err:
#             logger.error("%s %s \n",__s_fn_id__, err)
#             logger.debug(traceback.format_exc())
#             print("[Error]"+__s_fn_id__, err)

#         return self._data

#     @data.setter
#     def data(self,data):
#         """ @data.setter function

#             supports the class propert for setting the instance specific data. 
#             The data must not be None-Type and must be a pyspark dataframe.

#             return self._data (pyspark dataframe)
#         """

#         __s_fn_id__ = f"{self.__name__} function <@data.setter>"

#         try:
#             if data is None:
#                 raise AttributeError("Dataset cannot be empty")
#             self._data = data
#             logger.debug("%s data property %s set",__s_fn_id__,type(self._data))
                
#         except Exception as err:
#             logger.error("%s %s \n",__s_fn_id__, err)
#             logger.debug(traceback.format_exc())
#             print("[Error]"+__s_fn_id__, err)

#         return self._data

    ''' Function - @property mode and @mode.setter

            parameters:
                store_mode - local-fs sets to read and write on your local machine file system
                           aws-s3-bucket sets to read and write with an AWS S3 bucket 
            procedure: checks if it is a valid value and sets the mode
            return (str) self._storeMode

            author: <nuwan.waidyanatha@rezgateway.com>

    '''
    @property
    def storeMode(self) -> str:

        __s_fn_id__= f"{self.__name__} function <@property.storeMode>"
        try:
#                 self._storeMode.lower() not in self._storeModeList and \
            if self._storeMode is None and appConf.has_option("DATASTORE","MODE"):
                self._storeMode = appConf.get("DATASTORE","MODE")
                logger.warning("%s Reseting non-type storeMode to default %s from %s",
                              __s_fn_id__,self._storeMode.upper(), self.__conf_fname__.upper())

#             if not self._storeMode.lower() in self._storeModeList:
#                 raise ValueError("Parameter storeMode is not and must be set")

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._storeMode

    @storeMode.setter
    def storeMode(self, store_mode:str) -> str:

        __s_fn_id__ = f"{self.__name__} function @mode.setter"
        try:
            if not store_mode.lower() in self._storeModeList:
                raise ValueError("Parameter storeMode is not and must be set")
#             if not store_mode.lower() in self._storeModeList and \
#                 appConf.has_option["DATASTORE","MODE"]:
#                 self._storeMode = appConf.get["DATASTORE","MODE"]
#                 logger.warning("%s is invalid MODE and reseting to default % mode from %s",
#                               store_mode,self._storeMode, self.__conf_fname__)
#             else:
            self._storeMode = store_mode.lower()
            logger.debug("%s storeMode set to %s",__s_fn_id__,self._storeMode.upper())

#                 raise ValueError("Invalid mode = %s. Must be in %s" % (store_mode,self._storeModeList))

            if self._storeMode == 'google-storage' and os.environ.get('GCLOUD_PROJECT') is None:
                if appConf.has_option("GOOGLE","PROJECTID"):
                    os.environ["GCLOUD_PROJECT"] = appConf.get("GOOGLE","PROJECTID")
                    logger.info("%s GCLOUD_PROJECT os.environment set to %s",
                                __s_fn_id__, os.environ.get('GCLOUD_PROJECT'))
                else:
                    raise RuntimeError("PROJECTID of the google project, required to"+\
                                       "set the environment variable is undefined in %s"
                                       % (self.__conf_fname__))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._storeMode


    ''' Function - @property store_root and @store_root.setter

            parameters:
                store_root - local file system root directory or (e.g. wrangler/data/ota/scraper)
                            S3 bucket name (e.g. rezaware-wrangler-source-code)
            procedure: Check it the directory exists and then set the store_root property
            return (str) self._storeRoot

            author: <nuwan.waidyanatha@rezgateway.com>

    '''
    @property
    def storeRoot(self) -> str:

        __s_fn_id__ = f"{self.__name__} function @property storeRoot"

        try:
            if self._storeRoot is None and appConf.has_option("DATASTORE","ROOT"):
                self._storeRoot = appConf.get("DATASTORE","ROOT")
                logger.warning("%s Non-type storeRoot set to default %s from %s",
                               __s_fn_id__,self._storeRoot.upper(),self.__conf_fname__)
#             else:
#                 raise ValueError("Invalid Non-Type %s. Set as a property of define in %s"
#                                  % (self._storeRoot,self.__conf_fname__))


        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._storeRoot

    @storeRoot.setter
    def storeRoot(self, store_root:str="") -> str:

        __s_fn_id__ = f"{self.__name__} function @storeRoot.setter"

        try:
            if store_root is None and "".join(store_root.split())=="":
                raise AttributeError("Invalid store_root parameter %s" % store_root)
#                 appConf.has_option["DATASTORE","ROOT"]:
#                 self._storeRoot = appConf.get["DATASTORE","ROOT"]
#                 logger.warning("%s is invalid ROOT and reseting to default %",
#                               store_root,self._storeRoot)
#             else:
#                 self._storeRoot = store_root

            ''' validate storeRoot '''
            if self._storeMode == "aws-s3-bucket":
                ''' check if bucket exists '''
#                 logger.debug("%s %s",__s_fn_id__,self._storeMode)
                s3_resource = boto3.resource('s3')
                s3_bucket = s3_resource.Bucket(name=store_root)
#                 count = len([obj for obj in s3_bucket.objects.all()])
                try:
                    s3_resource.meta.client.head_bucket(Bucket=s3_bucket.name)
                except ClientError as aws_err:
                    logger.warning("%s Could not find %s bucket but will be created",
                                   __s_fn_id__,s3_bucket.name)
#                 logger.debug("%s bucket with %d objects exists",
#                              self.storeMode,count)
#                 if count <=0:
#                     raise ValueError("Invalid S3 Bucket = %s.\nAccessible Buckets are %s"
#                                      % (str(_bucket.name),
#                                         str([x for x in s3_resource.buckets.all()])))

            elif self._storeMode == "local-fs":
                ''' check if folder path exists '''
                if not os.path.exists(store_root):
                    raise ValueError("Invalid local folder path = %s does not exists." 
                                     % (store_root))

            elif self._storeMode == "google-storage":
                ''' check if bucket exists '''
                logger.debug("%s %s",__s_fn_id__,self._storeMode)
#                 client = storage.Client()
#                 if not client.bucket(store_root).exists():
#                     raise ValueError("Invalid GCS bucket %s, does not exist." % (store_root))

            else:
                raise ValueError("storeRoot %s does not exist for storeMode %s" 
                                 % (store_root,self._storeMode))

            self._storeRoot = store_root
            logger.debug("%s storeRoot set to %s for %s",
                         __s_fn_id__,self._storeRoot.upper(),self._storeMode.upper())

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._storeRoot


    @property
    def folderPath(self) -> str:
        """
        Description:
            property gets the vaildated folder path; if not exists raises an exception
        Attributes:
            NA
        Returns:
            self._folderPath (str)
        """

        __s_fn_id__ = f"{self.__name__} function <@property folderPath>"

        try:
            if self._folderPath is None:
                raise AttributeError("%s None-type folder cannot be used in this class" % __s_fn_id__)
            elif self._storeMode == "aws-s3-bucket":
                ''' check if bucket exists '''
                s3_resource = boto3.resource('s3')
                path = self._folderPath.rstrip('/') 
                s3_bucket = s3_resource.Bucket(name=self._storeRoot)
                resp = s3_resource.list_objects(
                    Bucket=s3_resource.Bucket(name=self._storeRoot),
                    Prefix=path,
                    Delimiter='/',
                    MaxKeys=1)
                if "CommonPrefixes" not in resp:
                    raise ValueError("%s Bucket %s does not contain a folder with path %s"
                                     % (__s_fn_id__,str(_bucket.name),path))

            elif self._storeMode == "local-fs":
                ''' check if folder path exists '''
                if not os.path.exists(os.path.join(self._storeRoot,self._folderPath)):
                    raise ValueError("Invalid local folder path %s does not exists in root %s." 
                                     % (self._folderPath),self._storeRoot)

            elif self._storeMode == "google-storage":
                ''' check if bucket exists '''
                logger.debug("%s %s",__s_fn_id__,self.storeMode)
            else:
                raise ValueError("%s something went wrong with getting %s from root %s in %s" 
                                 % (__s_fn_id__,self._folderPath,self._storeRoot,self._storeMode))
        
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._folderPath

    @folderPath.setter
    def folderPath(self,folder_path:str=None) -> str:
        """
        Description:
            property sets folder with path in the storeRoot
        Attributes:
            folder_path (str) - folder path relative to the root
        Returns:
            self._folderPath (str)
        """

        __s_fn_id__ = f"{self.__name__} function <@folderPath.setter>"

        try:
            if folder_path is None:
                raise AttributeError("Non-type folder_path value is unacceptable")

                ''' S3 creates the folder when saving a file; there's no concept of folders '''
            elif self._storeMode == "aws-s3-bucket":
                pass
#                 ''' check if bucket exists '''
#                 s3_resource = boto3.resource('s3')
#                 path = self._folderPath.rstrip('/') 
#                 s3_bucket = s3_resource.Bucket(name=self._storeRoot)
#                 resp = s3_resource.list_objects(
#                     Bucket=s3_resource.Bucket(name=self._storeRoot),
#                     Prefix=path,
#                     Delimiter='/',
#                     MaxKeys=1)
#                 ''' create folder if not exists '''
#                 if "CommonPrefixes" not in resp:
#                     k = s3_bucket.new_key(folder_path)
#                     k.set_contents_from_string('')
#                     logger.debug("%s Created the nonexistant folder %s in bucket %s %s",
#                                  __s_fn_id__,folder_path,self._storeRoot)
# #                     raise ValueError("%s Bucket %s does not contain a folder with path %s"
# #                                      % (__s_fn_id__,str(_bucket.name),path))

            elif self._storeMode == "local-fs":
                ''' check if folder path exists '''
                if not os.path.exists(os.path.join(self._storeRoot,folder_path)):
                    os.makedirs(os.path.join(self._storeRoot,folder_path))
#                     raise ValueError("Invalid local folder path %s does not exists in root %s." 
#                                      % (self._folderPath),self._storeRoot)

            elif self._storeMode == "google-storage":
                ''' check if bucket exists '''
                logger.debug("%s %s",__s_fn_id__,self.storeMode)
            else:
                raise ValueError("%s something went wrong with setting folder %s in root %s of %s" 
                                 % (__s_fn_id__,folder_path,self._storeRoot,self._storeMode))

            self._folderPath=folder_path
            logger.debug("%s Folder %s successfully set in root %s of %s",
                         __s_fn_id__,folder_path,self._storeRoot,self._storeMode)
    
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._folderPath

    @property
    def asType(self) -> str:
        """
        Description: 
            The asType property sets the dtype for returning the data. See self._asTypeList
            for valid types. If the asType is not set then it will default to SPARK dataframe
        
        Returns:
            self._asType (str) if unspecified default is SPARK
        """
        __s_fn_id__ = f"{self.__name__} function @property asType"

        try:
            if self._asType is None:
                self._asType = 'spark'
                logger.warning("%s Non-type asType value, set to default value SPARK",__s_fn_id__)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._asType.lower()

    @asType.setter
    def asType(self, as_type:str="") -> str:
        """
        Description: 
            The asType property is set to the value defined in the as_type input parameter.
            If the value is not in self._asTypeList, the function will throw an exception.
        
        Returns:
            self._asType (str)
        """
        __s_fn_id__ = f"{self.__name__} function @asType.setter"

        try:
            if as_type.lower() not in self._asTypeList:
                raise AttributeError("Invaid attribute as_type; must be %s" % str(self._asTypeList))
            self._asType = as_type.lower()
            logger.debug("%s The asType property set to %s",__s_fn_id__,self._asType)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._asType


    ''' RWFORMAT '''
    @property
    def rwFormat(self) -> str:

        __s_fn_id__ = f"{self.__name__} function <@property rwFormat>"

        try:
            if self._rwFormat is None and appConf.has_option('SPARK','FORMAT'):
                self._rwFormat = appConf.get('SPARK','FORMAT')
                logger.debug("Non-type Spark rwFormat set to default %s from %s"
                             ,self._rwFormat,self.__conf_fname__)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._rwFormat

    @rwFormat.setter
    def rwFormat(self,rw_format:str='csv') -> str:

        __s_fn_id__ = f"{self.__name__} function <@rwFormat.setter>"

        try:
            if rw_format.lower() not in self._rwFormatTypes:
                raise AttributeError("Invalid spark read/write FORMAT %s must %s" 
                                     % (rw_format,str(self._rwFormatTypes)))

            self._rwFormat = rw_format
            logger.debug("@setter Spark rwFormat set to: %s",self._rwFormat)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._rwFormat


    ''' Function --- SPARK CONTEXT ---

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    @property
    def context(self):
        """
        Description:
            Sets the spark contect to work with Hadoop necessary for AWS-S3, GCS, etc
            If the context is None, @property will set the default values
        Attributes:
        Returns:
            self._context (sparkConf object)
        """

        __s_fn_id__ = f"{self.__name__} function <@property context>"
        _access_key=None
        _secret_key=None

        try:
            if self._context is None:
                logger.debug("%s %s %s",__s_fn_id__, self._storeMode)
                conf = self.session.sparkContext._jsc.hadoopConfiguration()
                if self._storeMode == 'aws-s3-bucket':
                    print('getting credentials')
                    _access_key, _secret_key = credentials.aws()
                    logger.debug("%s %s %s",__s_fn_id__, _access_key, _secret_key)
                    conf.set("fs.s3a.access.key", _access_key)
                    conf.set("fs.s3a.secret.key", _secret_key)
                    conf.set("fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem")

                elif self._storeMode == 'google-storage':
                    conf.set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
                    conf.set("fs.AbstractFileSystem.gs.impl",
                             "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
                    # This is required if you are using service account and set true, 
                    conf.set('fs.gs.auth.service.account.enable', 'true')
                    ##conf.set('google.cloud.auth.service.account.json.keyfile', "/path/to/keyfile")
                    # Following are required if you are using oAuth
                    ##conf.set('fs.gs.auth.client.id', 'YOUR_OAUTH_CLIENT_ID')
                    ##conf.set('fs.gs.auth.client.secret', 'OAUTH_SECRET')
                else:
                    pass

                self._context = conf
                logger.info("Non-type context configurered for Spark %s authentication with %s"
                            ,self.storeMode,self._context)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug("%s",traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._context

    @context.setter
    def context(self,context_args:dict={}):

        __s_fn_id__ = f"{self.__name__} function <@context.setter>"
        _access_key=None
        _secret_key=None

        try:
            conf = self.session.sparkContext._jsc.hadoopConfiguration()
            ''' -- AWS S3 --- '''
            if self.storeMode == 'aws-s3-bucket':
                _access_key, _secret_key = credentials.aws()
                if "ACCESSKEY" in context_args.keys():
                    _access_key = context_args['ACCESSKEY']
                if "SECRETKEY" in context_args.keys():
                    _secret_key = context_args['SECRETKEY']

                conf.set("fs.s3a.access.key", _access_key)
                conf.set("fs.s3a.secret.key", _secret_key)
                conf.set("fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem")

            elif self.storeMode == 'google-storage':
                conf.set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
                conf.set("fs.AbstractFileSystem.gs.impl",
                         "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
                # This is required if you are using service account and set true, 
                conf.set('fs.gs.auth.service.account.enable', 'true')
#                 if "".join(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])!="":
#                     conf.set('google.cloud.auth.service.account.json.keyfile',
#                              os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
                # Following are required if you are using oAuth
                ##conf.set('fs.gs.auth.client.id', 'YOUR_OAUTH_CLIENT_ID')
                ##conf.set('fs.gs.auth.client.secret', 'OAUTH_SECRET')

            else:
                pass

            self._context = conf
            logger.info("%s Setting a new spark contex config: %s for %s",
                        __s_fn_id__,self._context, self.storeMode)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._context


    ''' Function --- IMPORT_FILES ---

        TODO : google-storage still not working
        author: <nuwan.waidyanatha@rezgateway.com>
    '''
    def converter(func):
        """
        Description:
            The retrieved data is converted to the dtype specificed in the as_type
            attribute. The self._data is returned in the prescribed format.
        Arguments:
            func inherits the import_files function
        Returns:
            self._data (pyspark dataframe)        
        """
        @functools.wraps(func)
        def wrapper_converter(
            self,
            as_type,
            folder_path,
            file_name,
            file_type,
            **options,
        ):
            """
                @functools.wraps receives the data, if not null and possible to convert
                will convert to the requested dtype.
            """
            __s_fn_id__ = f"{self.__name__} Function <wrapper_converter>"

            try:
#                 format_, as_type_ = func(self,as_type,folder_path,file_name,file_type,**options)
                data_ = func(self,as_type,folder_path,file_name,file_type,**options)
                if data_ is None:
                    raise AttributeError("Cannot initiate %s with Non-type dataset" % __s_fn_id__)

                ''' convert to dtype '''
                if self.asType == 'pandas':
                    self._data = data_.toPandas()
                    logger.debug("Converted %d rows to pandas dataframe",self._data.shape[0])
                elif self.asType == 'dict':
                    if isinstance(data_,dict):
                        self._data=data_
                    else:
                        self._data = data_.toPandas().T.to_dict('list')
                    logger.debug("Converted %d rows to dictionary",len(self._data))
                elif self.asType == 'list':
                    logger.warning("Returning data as pyspark dataframe. "+\
                                   "Unsupported convertion, to be implemented in the future")
                elif self.asType == 'str':
                    logger.warning("Returning data as pyspark dataframe. "+\
                                   "Unsupported convertion, to be implemented in the future")
                else:
                    self._data = data_
                    logger.info("Retuning %d rows pyspark dataframe, no convertion required."
                                ,self._data.count())

            except Exception as err:
                logger.error("%s %s",__s_fn_id__, err)
                logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)

            return self._data

        return wrapper_converter

    def importer(func):
        """
        Description:
            Wrapper function will apply the storeMode specific import protocols
            to read the data as a pyspark dataframe
        Arguments:
            func inherits the import_files function
        Returns:
            self._data (pyspark dataframe)        
        """
        @functools.wraps(func)
        def wrapper_importer(
            self,
            as_type,
            folder_path,
            file_name,
            file_type,
            **options,
        ):
            """
                @functools.wraps to read the data into a pyspark dataframe using the
                established self_rwFormat and the self.session
            """
            __s_fn_id__ = f"{self.__name__} function <wrapper_importer>"

            try:
                format_, as_type_ = func(self,as_type,folder_path,file_name,file_type,**options)
                if format_ is None or as_type_ is None:
                    raise AttributeError("The format_ or as_type_ can not be a Non-Types")

                '''set and validate the folder path '''
                self.folderPath=folder_path

                if self.storeMode == 'local-fs':
                    ''' read content from local file system '''
                    if file_name:
                        file_path = str(os.path.join(self.storeRoot,self._folderPath,file_name))
                    else:
                        file_path = str(os.path.join(self.storeRoot,self._folderPath))
#                     print(file_path)
                elif self.storeMode == 'aws-s3-bucket':
                    ''' read content from s3 bucket '''
                    if file_name:
                        file_path = str(os.path.join(self.storeRoot,self._folderPath,file_name))
                    else:
                        file_path = str(os.path.join(self.storeRoot,self._folderPath))
                    file_path = "s3a://"+file_path
                    self.context = {}

                elif self.storeMode == 'google-storage':
                    ''' read content from google cloud storage '''
                    if file_name:
                        file_path = str(os.path.join(self.storeRoot,self._folderPath,file_name))
#                         client = storage.Client()
#                         bucket = client.bucket(self.storeRoot)
#                         blob = bucket.blob(file_path)
#     #                     blob.upload_from_filename(_tmp_file_path)
#                         with blob.open("r") as f:
#                             file_content = f.read()
                    elif file_type:
                        ''' multiple files of same file type '''
                        file_path = str(os.path.join(self.storeRoot,self._folderPath))
                    file_path = "gs://"+file_path
                    self.context = {}
                else:
                    raise typeError("Invalid storage mode %s" % self.storeMode)
                logger.debug("%s file path set to %s", __s_fn_id__, file_path)

                if as_type_ in ['spark','pandas','array','list']:
                    sdf = self.session.read\
                                    .format(self.rwFormat)\
                                    .options(**options)\
                                    .load(file_path)
                    if sdf.count() > 0:
                        self._data = sdf
                        logger.info("Read %d rows of data into dataframe",self._data.count())
                elif as_type_ in ['dict']:
                    ''' TOD MOVE TO @staticmethod and add S3 & GS '''
                    with open(file_path, 'r') as _dict_file:
                        self._data = json.load(_dict_file)
                        logger.info("Read %d rows of data into dictionary",len(self._data))

                elif as_type_ in ['str']:
                    ''' TOD MOVE TO @staticmethod and add S3 & GS '''
                    try:
                        with open(file_path, 'r') as _str_file:
                            self._data = _str_file.read()
                            logger.info("Read %d char length str",len(self._data))
                    except FileNotFoundError as ferr:
                        logger.error("%s File not found. %s",__s_fn_id__, ferr)

            except Exception as err:
                logger.error("%s %s",__s_fn_id__, err)
                logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)

            return self._data
        return wrapper_importer


    @converter
    @importer
    def read_files_to_dtype(
        self,
        as_type:str="",      # optional - define the data type to return
        folder_path:str="",  # optional - relative path, w.r.t. self.storeRoot
        file_name:str=None,  # optional - name of the file to read
        file_type:str=None,  # optional - read all the files of same type
        **options,
    ):
        """
        Description:
            When the direct file path or path to a folder is given, the function
            will read the files and convert them to a dtype specified as_type.
            If a file type is given, instead of the direct file path with name, 
            the function will read all the files in the folder of the requested file
            type. If both a file_name and file_type are specified, then only data
            from the file_name is imported.
            If a folder is not specified, then the function will import data from the
            root folder (e.g. AWS S3 bucket)
        Arguments:
            as_type (str) the dtype to return data as; e.g. dataframe, dict, list
                see self._asTypeList. If unspecified will return the default pyspark
                dataframe
            folder_path (str) a path to the folder relative to the storeRoot; if
                unspecified, will use the self._storeRoot as the folder
            file_name (str) specific a particular file of type
            file_type (str) speficies a file type to read a collection of files; 
                see self._rwFormatTypes for supported file types
            kwargs
        returns:
            _read_mode (str) indicating whether a file name or file type read process
        """

        __s_fn_id__ = f"{self.__name__} function <read_files_to_dtype>"
        _read_mode = None
        _read_format = None

        try:
            self.asType = as_type  # validate and set the property
            logger.info("Reading data from %s at root %s",self.storeMode,self.storeRoot)
            if file_name is not None and \
                "".join(file_name.split())!="":
                ''' check if supported file type and set the reFormat propert '''
                _fname, _fext = os.path.splitext(file_name)  
                self.rwFormat = _fext.replace(".", "").lower()
                logger.info("Reading format %s data from file %s from folder %s",
                           self._rwFormat,_fname,folder_path)
            elif file_type.replace(".", "").lower() in self._rwFormatTypes:
                self.rwFormat = file_type.lower()
                logger.info("Reading all files with read format %s from folder %s",
                           self._rwFormat,folder_path)
            else:
                raise AttributeError("Either a file_name to read a single file "+\
                                     "or a file_type to read all files of type "+\
                                     "must be specified.")

        except Exception as err:
            logger.error("%s %s",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._rwFormat, self._asType
        


    ''' Function --- READ_CSV_TO_SDF ---
            TODO:
            author: <nuwan.waidyanatha@rezgateway.com>
            
    '''
#     @classmethod
    def read_csv_to_sdf(
        self,
        files_path:str="", 
        **kwargs):
        """ 
            When the direct file path or path to a folder is given, the function
            will read the files and convert them to a pyspark dataframe. If a folder
            path is given, instead of the direct file path with name, the function
            will read all the CSV files in that folder
            
            Arguments:
                filesPath (str) a path relative to the storeRoot

            return self._data (pyspark dataframe)
        """

        __s_fn_id__ = f"{self.__name__} Class <SparkWorkLoads> Function <read_folder_csv_to_sdf>"

        _csv_to_sdf = self.session.sparkContext.emptyRDD()     # initialize the return var
#         _tmp_df = self.session.sparkContext.emptyRDD()
        _start_dt = None
        _end_dt = None
        _sdf_cols = []
        _l_cols = []
        _traceback = None

        try:
            ''' check if the folder and files exists '''
            if not filesPath or "".join(files_path.split())=="":
                raise ValueError("Invalid file or folder path %s" % filesPath)
            if "IS_FOLDER" in kwargs.keys() and kwargs['IS_FOLDER']:
                filelist = os.listdir(filesPath)
                if not (len(filelist) > 0):
                    raise ValueError("No data files found in director: %s" % (filesPath))

            ''' set options '''
            _recLookup="true"
            if "RECURSIVELOOKUP" in kwargs.keys() and \
                kwargs['RECURSIVELOOKUP'].lower() in ['true','false']:
                _recLookup = kwargs['RECURSIVELOOKUP'].lower()
            _header="true"
            if "HEADER" in kwargs.keys() and kwargs['HEADER'].lower() in ['true','false']:
                _header = kwargs['HEADER'].lower()
            _inferSchema = "true"
            if "INFERSCHEMA" in kwargs.keys() and kwargs['INFERSCHEMA'].lower() in ['true','false']:
                _inferSchema = kwargs['INFERSCHEMA'].lower()
            _delimeter = ','
            if "DELIMETER" in kwargs.keys():
                _delimeter = kwargs['DELIMETER']
            ''' extract data from **kwargs if exists '''
            if 'SCHEMA' in kwargs.keys():
                _sdf_cols = kwargs['SCHEMA']
            if 'FROMDATETIME' in kwargs.keys():
                _from_dt = kwargs['FROMDATETIME']
            if 'TODATETIME' in kwargs.keys():
                _to_dt = kwargs['TODATETIME']

            _csv_to_sdf = self.session.read\
                            .options( \
                                     recursiveFileLookup=_recLookup, \
                                     header=_header, \
                                     inferSchema=_inferSchema, \
                                     delimiter=_delimeter \
                                    ) \
                            .csv(filesPath)

#            _csv_to_sdf.select(split(_csv_to_sdf.room_rate, '[US$]',2).alias('rate_curr')).show()
            if 'TOPANDAS' in kwargs.keys() and kwargs['TOPANDAS']:
                _csv_to_sdf = _csv_to_sdf.toPandas()
                logger.debug("Converted pyspark dataframe to pandas dataframe with %d rows"
                             % _csv_to_sdf.shape[0])

        except Exception as err:
            logger.error("%s %s",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return _csv_to_sdf


    ''' Function - write_data with wrapper_converter

            author(s): <nuwan.waidyanatha@rezgateway.com>
            
            TODO: cleanup this code to modify the tmp file save and read
                before saving to storeMode. For example, pyspard dataframe
                is converted to pandas to the tmp file save.
    '''
    def writer(func):

        @functools.wraps(func)
        def wrapper_writer(
            self,
            file_name:str,   # optional - name of the file to read
            folder_path:str, # mandatory - relative path, w.r.t. self.storeRoot
            data,   # data to be stored
        ):

            __s_fn_id__ = f"{self.__name__} function <wrapper_writer>"

            try:
                _file_path=func(self,file_name,folder_path,data)

                ''' create a tmp folder to stage the file before writing to file path '''
                _tmp_folder_path = os.path.join(self.dataDir,"tmp")
                if not os.path.exists(_tmp_folder_path):
                    os.makedirs(_tmp_folder_path)
                    logger.debug("Created a new tmp folder %s",_tmp_folder_path)
                else:
                    logger.debug("%s tmp folder exists",_tmp_folder_path)
                ''' make the tmp file path to save '''    
                _tmp_file_path = os.path.join(_tmp_folder_path,file_name)
                logger.debug("tmp_file_path set to %s",_tmp_file_path)

                ''' check data type and save to tmp '''
                try:
                    _file_type = file_name.rsplit('.',1)[1]
                except IndexError:
                    _file_type = None
                if _file_type is None:
                    raise ValueError("Failed to strip file type extension from %s" 
                                     % file_name.upper())
                logger.debug("%s File name %s is of file type %s", 
                             __s_fn_id__, file_name.upper(), _file_type.upper())
    #             _file_type = file_name.rsplit('.',1)[1]

                if isinstance(data,str):
                    ''' Strings to txt, csv, json files '''
                    if _file_type.lower() == 'txt':
                        print(_file_type)
                    elif _file_type.lower() == 'csv':
                        print(_file_type)

                elif isinstance(data,dict):
                    ''' Dictionary to txt, csv, json files '''
                    if _file_type.lower() == 'csv':
                        with open(_tmp_file_path, 'w') as f:  
                            writer = csv.writer(f)
                            for key, value in data.items():
                                writer.writerow([key, value])
                    elif _file_type.lower() == 'json':
                        with open(_tmp_file_path, "w") as f:
                            json.dump(data, f)
                    else:
                        raise TypeError("Unsupported file type for dictionary data type")

                elif isinstance(data,list):
                    ''' List to txt, csv, json files '''
                    if _file_type.lower() == 'txt':
                        print("TBD",_file_type)
                    elif _file_type.lower() == 'json':
                        with open(_tmp_file_path, "w") as f:
                            json.dump(data, f)
                    elif _file_type.lower() == 'csv':
                        print(_file_type)
                    else:
                        raise TypeError("Unsupported file type for List data type")

                elif isinstance(data,np.ndarray):
                    ''' Array to txt, csv files '''
                    if _file_type.lower() == 'txt':
                        np.savetxt(_tmp_file_path, data)
                    elif _file_type.lower() == 'csv':
                        np.savetxt(_tmp_file_path, data, delimiter=",")
                    else:
                        raise TypeError("Unsupported file type for Array data type")

                elif isinstance(data,pd.DataFrame):
                    ''' Pandas DataFrame to txt, csv, json files '''
#                     if file_name.rsplit('.',1)[1] == "csv":
                    if _file_type.lower() == "csv":
                        data.to_csv(_tmp_file_path,index=False)
                    elif _file_type.lower() == "json":
                        data.to_json(_tmp_file_path)
                    else:
                        raise TypeError("Unsupported file type for Pandas DataFrame data type")

                elif isinstance(data,DataFrame):
                    ''' Spark dataframe to txt, csv, json files '''
    #                 _data_type = "SPARK"
    #                 options={
    #                     "header":True,
    #                 }
    #                 self.saveMode="Overwrite"
    #                 self._data.write.mode(self._saveMode)\
    #                         .option("header",True)\
    #                         .format(self.rwFormat)\
    #                         .save(_tmp_file_path)
                    self._data = data.toPandas()
                    if _file_type.lower() == "csv":
                        self._data.to_csv(_tmp_file_path,index=False)
                    elif _file_type.lower() == "json":
                        self._data.to_json(_tmp_file_path)
                    else:
                        raise TypeError("Unsupported file type for Pandas DataFrame data type")
                else:
                    raise TypeError("Unrecognized data type %s must be either of\n%s"
                                    % (type(self._data),str(self._asTypeList)))

                ''' transfer the tmp file to storage '''
                with open(_tmp_file_path,'r') as infile:
    #                 object_data = infile.read()
                    self._data = infile.read()

                    if self.storeMode == 'aws-s3-bucket':
                        ''' write file to AWS S3 Bucket '''
                        s3 = boto3.client('s3')
                        s3.put_object(Body=self._data, 
                                      Bucket=self.storeRoot,
                                      Key=_file_path)

                    elif self.storeMode == 'local-fs':
                        ''' write file to Local File System '''
                        with open(_file_path,'w') as savefile:
                            savefile.write(self._data)

                    elif self.storeMode == 'google-storage':
                        ''' write file to google-cloud-storage '''
                        client = storage.Client()
                        bucket = client.bucket(self.storeRoot)
                        blob = bucket.blob(_file_path)
    #                     blob.upload_from_filename(_tmp_file_path)
                        with blob.open("w") as f:
                            f.write(self._data)

                    else:
                        raise RuntimeError("Something went wrong writing %s file to %s"
                                           % (_tmp_file_path,self.storeMode))

                ''' remove the tmp file throw exception if something other than does not exists '''
                try:
                    os.remove(_tmp_file_path)
                except OSError as e:
                    if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
                        raise # re-raise exception if a different error occurred

            except Exception as err:
                logger.error("%s %s \n",__s_fn_id__, err)
                logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)

#             return self._data
            return _file_path

        return wrapper_writer

    @writer
    def write_data(
        self,
        file_name:str,   # optional - name of the file to read
        folder_path:str, # mandatory - relative path, w.r.t. self.storeRoot
        data,   # data to be stored
    ):
        """
        Description:

        Attributes:
            as_type (str) - mandatory - define the data type to return
            folder_path(str) - madandatory to give the relative path w.r.t. store_root
            file_name (str) - is mandatory and can be any defined in self._asTypeList
            file_type:str=None    # optional - read all the files of same type
        Returns (str) _file_path

        Resources:                     
          * https://www.sqlservercentral.com/articles/reading-a-specific-file-from-an-s3-bucket-using-python
          * https://realpython.com/python-boto3-aws-s3/
        """

        _s_fn_id = f"{self.__name__} function <write_data>"
        file_content=None

        try:
#             self.data=data
#             logger.debug("Writing %d rows of data to %s",self.data.count(),self.storeMode)
            if data is None:
                raise AttributeError("None-type data set cannot be processed")

            if self.storeMode == 'aws-s3-bucket':
                ''' check if s3 bucket key not exists, then create '''
                s3 = boto3.client('s3')
                try:
                    _header = s3.head_object(Bucket=self.storeRoot,
                                   Key=folder_path)
                    logger.debug("Key folder %s in %s s3 bucket exists "+\
                                 "with content length %s last modified %s",
                                 folder_path,
                                 self.storeRoot,
                                 _header['ContentLength'],
                                 _header['LastModified'])
                except ClientError as s3e:
                    if s3e.response['ResponseMetadata']['HTTPStatusCode'] == 404:
                        ''' Not found then create '''
                        s3.put_object(Bucket=self.storeRoot,Body='', Key=folder_path)
                        logger.debug("Created a new Key folder %s in %s s3 bucket",
                                     folder_path,self.storeRoot)
                    else:
                        raise RuntimeError("Something was wrong when checking S3 bucket key header\n%s",
                                    s3e)
                ''' set the full key path with file '''
                _file_path = str(os.path.join(folder_path,file_name))

            elif self.storeMode == 'local-fs':
                ''' check if folder existis '''
                if not os.path.exists(os.path.join(self.storeRoot,folder_path)):
                    os.makedirs(os.path.join(self.storeRoot,folder_path))
                    logger.debug("Created a new folder %s in %s root path ",
                                     folder_path,self.storeRoot)
                else:
                    logger.debug("Folder %s in %s root path Exists",
                                     folder_path,self.storeRoot)
                ''' give absolute path to write '''
                _file_path = str(os.path.join(self.storeRoot,folder_path,file_name))

            elif self.storeMode == 'google-storage':
                ''' check if google bucket key not exists, then create '''
                _file_path = str(os.path.join(folder_path,file_name))

            else:
                raise typeError("Invalid storage mode %s" % self.storeMode)


        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return _file_path


    ''' Function --- SAVE SDF TO CSV

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    def save_sdf_to_csv(
        self, 
        data : any = None,  # can be any pandas or spark dataframe 
        file_path:str=None, # file path relative to the storeRoot
        **kwargs):
        """
        """
        
        _csv_file_path = None

        __s_fn_id__ = f"{self.__name__} function <save_sdf_to_csv>"
        logger.info("Executing %s in %s",__s_fn_id__, __name__)

        try:
            self._data = data
            ''' determine where to save '''
            if file_path:
                _csv_file_path = file_path
                logger.info("File ready to save to %s", _csv_file_path)
            else:
                fname = __package__+"_"+"save_sdf_to.csv"
                _csv_file_path = os.path.join(self.tmpDIR, fname)
                logger.info("No file path defined, saving to default %s", _csv_file_path)

            self.rwFormat = 'csv'

            ''' save sdf to csv '''
            self._data.write.mode(self._saveMode)\
                    .option("header",True)\
                    .format(self.rwFormat)\
                    .save(_csv_file_path)

            logger.info("%d rows of data written to %s",
                        self._data.count(), _csv_file_path)

        except Exception as err:
            logger.error("%s %s",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return _csv_file_path


class credentials():
    """
    Description:
        The class supports retrieving authentication credentials for establishing the
        necessary cloud storage connections.
    """

    ''' Function --- GET_APP_CONF ---

            author: <nuwan.waidyanatha@rezgateway.com>
                    <samana.thetha@gmail.com>
                    <farmraider@protonmail.com>
    '''
    @staticmethod
    def get_app_conf():
        """
        Description:
            Uses the dataWorkLoads current working directory (cwd) and config file name
        Attributes:
        Returns: 
            appCFG (configparser object)
        """
        __s_fn_id__ = f"{dataWorkLoads.__name__} function <get_app_conf>"

        try:
#         self.cwd=os.path.dirname(__file__)
            clsSpark = dataWorkLoads(desc=__s_fn_id__)
            appCFG = configparser.ConfigParser()
            appCFG.read(os.path.join(clsSpark.appDir,clsSpark.__conf_fname__))
            
        except Exception as err:
            logger.error("%s %s",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return appCFG

    ''' Function --- GET_APP_CONF ---
    
        TODO: retrieve credentials from the name profile. Now it is reading
              the default profile.

        authors: <nuwan.waidyanatha@rezgateway.com>
    '''
    @staticmethod
    def aws(**kwargs):
        """ 
        Description:
            returns the aws credentials from the aws CLI established credentials
            file stored in the local folder defined in the app config file (__conf_fname__)
            to instatiate the configparser to read the key value pairs. 
        Attributes:
        Returns: 
            aws_access_key_id (str) 
            aws_secret_access_key (str)
        """
        import os
        from configparser import ConfigParser
        
        __s_fn_id__ = f"{dataWorkLoads.__name__} function <aws> credentials"
        aws_access_key_id=None
        aws_secret_access_key=None

        try:
            _appCFG = credentials.get_app_conf()
            ''' --- profile --- '''
            if "CREDPROFILE" in kwargs.keys():
                _profile = kwargs['CREDPROFILE']
            elif _appCFG.has_option('AWSAUTH','CREDPROFILE'):
                _profile = _appCFG.get('AWSAUTH','CREDPROFILE')
            else:
                _profile = 'DEFAULT'

            ''' --- file --- '''
            if "CREDFILEPATH" in kwargs.keys():
                _fpath = kwargs['CREDFILEPATH']
            elif _appCFG.has_option('AWSAUTH','CREDFILEPATH'):
                _fpath = _appCFG.get('AWSAUTH','CREDFILEPATH')
            else:
                _fpath = '~/.aws/credentials'

            ''' initialize configparser to read credentials '''
            credConf = configparser.ConfigParser()
            credConf.read(_fpath)
            ''' get profile specific keys '''
            aws_access_key_id = credConf.get(_profile.lower(),"aws_access_key_id")
            aws_secret_access_key=credConf.get(_profile.lower(),"aws_secret_access_key")
            ''' create environment variables '''
            if os.environ.get["AWS_ACCESS_KEY"] is None or os.environ.get["AWS_ACCESS_KEY"]=="":
                os.environ["AWS_ACCESS_KEY"] = aws_access_key_id
            if os.environ.get["AWS_SECRET_KEY"] is None or os.environ.get["AWS_SECRET_KEY"]=="":
                os.environ["AWS_SECRET_KEY"] = aws_secret_access_key

#             with open(os.path.expanduser(_fpath)) as f:
#                 for line in f:
#                     try:
#                         if line.lower()=="["+_profile.lower()+"]":
#                             print(line.lower(),"["+_profile.lower()+"]")
#     #                     print(line.strip().split(' = '))
#                             key, val = line.strip().split(' = ')
#                             if key == 'aws_access_key_id':
#                                 aws_access_key_id = val
#                             elif key == 'aws_secret_access_key':
#                                 aws_secret_access_key = val
#                     except ValueError:
#                         pass

        except Exception as err:
            logger.error("%s %s",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return aws_access_key_id, aws_secret_access_key


#     ''' Function
#             name: read_s3obj_to_sdf
#             parameters:
#                 bucketName (str) - s3 bucket name
#                 objPath (str) - s3 key that points to the objecy
#             procedure: 
#             return DataFrame

#             author: <nuwan.waidyanatha@rezgateway.com>

#     '''
#     def read_s3csv_to_sdf(self,bucketName:str,keyFPath: str, **kwargs):

#         import boto3
        
#         _csv_to_sdf = self.session.sparkContext.emptyRDD()     # initialize the return var
# #         _tmp_df = self.session.sparkContext.emptyRDD()
#         _start_dt = None
#         _end_dt = None
#         _sdf_cols = []
#         _l_cols = []
#         _traceback = None
        
#         __s_fn_id__ = "function <read_s3csv_to_sdf>"
#         logger.info("Executing %s in %s",__s_fn_id__, __name__)

#         try:

#             if not 'AWSAUTH' in pkgConf.sections():
#                 raise ValueError('Unable to find AWSAUTH keys and values to continue')
            
#             AWS_ACCESS_KEY_ID = pkgConf.get('AWSAUTH','ACCESSKEY')
#             AWS_SECRET_ACCESS_KEY = pkgConf.get('AWSAUTH','SECURITYKEY')
#             AWS_REGION_NAME = pkgConf.get('AWSAUTH','REGION')

#             s3 = boto3.resource(
#                 's3',
#                 aws_access_key_id=AWS_ACCESS_KEY_ID,
#                 aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#                 region_name=AWS_REGION_NAME,
#             )
# #             response = s3.get_object(Bucket=bucketName, Key=str(key))
# #             print(self.session.__dict__)
# #             self.session.sparkContext.hadoopConfiguration.set("fs.s3a.access.key", AWS_ACCESS_KEY_ID)
# #             self.session.sparkContext\
# #                     .hadoopConfiguration.set("fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
# #             self.session.sparkContext\
# #                   .hadoopConfiguration.set("fs.s3a.endpoint", "s3.amazonaws.com")

# #             os.environ['PYSPARK_SUBMIT_ARGS'] = '-- packages com.amazonaws:aws-java-sdk:1.7.4,org.apache.hadoop:hadoop-aws:2.7.3 pyspark-shell'
            
# #             conf = SparkConf().set('spark.executor.extraJavaOptions', \
# #                                    '-Dcom.amazonaws.services.s3.enableV4=true')\
# #                             .set('spark.driver.extraJavaOptions', \
# #                                  '-Dcom.amazonaws.services.s3.enableV4=true')\
# #                             .setAppName('pyspark_aws')\
# #                             .setMaster('local[*]')
            
# #             sc=SparkContext(conf=conf)
# # #             sc=self.session.sparkContext(conf=conf)
# #             sc.setSystemProperty('com.amazonaws.services.s3.enableV4', 'true')
            
# #             hadoopConf = sc._jsc.hadoopConfiguration()
# #             hadoopConf.set('fs.s3a.access.key', AWS_ACCESS_KEY_ID)
# #             hadoopConf.set('fs.s3a.secret.key', AWS_SECRET_ACCESS_KEY)
# #             hadoopConf.set('fs.s3a.endpoint', AWS_REGION_NAME)
# #             hadoopConf.set('fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem')
            
# #             spark=SparkSession(sc)

# #             Bucket=bucketName,
# #             Key=keyFPath

# #             s3 = boto3.resource('s3')
#             bucket = s3.Bucket(str(bucketName))
#             obj = bucket.objects.filter(Prefix=str(keyFPath))
# #             response = s3.get_object(Bucket=bucketName, Key=str(keyFPath))
# #             _s3_obj = "s3a://"+bucketName+"/"+objPath
# #             _csv_to_sdf=spark.read.csv(
# #             _csv_to_sdf=self.session.read.csv(
#             _csv=self.session.read.csv(
#                 obj,
# #                 _s3_obj,
#                 header=True,
#                 inferSchema=True)
# #             _csv_to_sdf = self.session.read.csv(_s3_obj)

#         except Exception as err:
#             logger.error("%s %s \n",__s_fn_id__, err)
#             print("[Error]"+__s_fn_id__, err)
#             print(traceback.format_exc())

#         return _csv_to_sdf