#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "csvToRDBM"
__module__ = "io"
__package__ = "importer"
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
    from pyspark.sql.types import *
    from google.cloud import storage   # handles GCS reads and writes
    import pandas as pd
    import numpy as np
    import json

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))

'''
    CLASS process read and write postGIS geometry data for specific database, schema, and tables.

        Makes uses of
        * sparkRDBM for reading and writing date from and to database tables
        * each database schema will have a deafult table location_geom with
            table attributes for holding any of the database entity gis data

    Contributors:
        * nuwan.waidyanatha@rezgateway.com
        * samana.thetha@gmail.com

    Resources:
        https://postgis.net/docs/en/using_postgis_dbmanagement.html
'''
class dataWorkLoads():
    ''' Function --- INIT ---
    
        author: <nuwan.waidyanatha@rezgateway.com>

    '''
    def __init__(
        self, 
        desc : str = None,   # identifier for the instances
        store_mode:str=None,
        store_root:str=None,
        db_type : str=None,
        db_name : str=None,
        db_schema:str=None,
        db_user : str=None,
        db_pswd : str=None,
        **kwargs,
    ):
        """
        Description:
            Initializes the postGIS dataWorkloads: class property attributes, app 
            configurations, logger function, data store directory paths, and global classes
        Attributes:
            desc (str) to change the instance description for identification
            db_name (str),
            db_schema (str),
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
        self._storeMode = store_mode
        self._storeRoot = store_root
        self._dbType = db_type
        self._dbName = db_name     # sets the database to interact with
        self._dbSchema=db_schema # specifies the schema
        self._dbUser = db_user
        self._dbPswd = db_pswd

        ''' initiate to load app.cfg data '''
        global logger  # inherits the utils logger class
        global pkgConf # inherits package app.ini config data
        global clsSDB  # inherits the loader sparkRDBM dataWorkLoads
        global clsFile # inherits the loader sparkFile dataWorkLoads

        __s_fn_id__ = f"{self.__name__} function <__init__>"

        try:
            self.cwd=os.path.dirname(__file__)
            pkgConf = configparser.ConfigParser()
            pkgConf.read(os.path.join(self.cwd,__ini_fname__))

            self.rezHome = pkgConf.get("CWDS","PROJECT")
            sys.path.insert(1,self.rezHome)

            ''' initialize the logger '''
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
            from rezaware.modules.etl.loader import sparkRDBM as db
            clsSDB = db.dataWorkLoads(desc=self.__desc__)
            ''' instantiate sparkFile dataWorkLoads '''
            from rezaware.modules.etl.loader import sparkFile as file
            clsFile = file.dataWorkLoads(desc=self.__desc__)

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
                self._data = clsSDB.session.createDataFrame(self._data)
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
                self._data = clsSDB.session.createDataFrame(data)
                logger.warning("%s %s converted to %s",
                               __s_fn_id__, type(data), type(self._data))
            else:
                self._data = data

            logger.debug("%s data propert set with %d rows and %d columns ",
                         __s_fn_id__, self._data.count(), len(self._data.columns))
                
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data


    ''' Function --  --
            TODO: 
            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    def write_data(func):

        @functools.wraps(func)
        def wrapper_writer(self, folder_path, file_name, db_tbl, **kwargs):
            """
            """
            __s_fn_id__ = f"{self.__name__} function <wrapper_writer>"
            
            _audit_proc_str = "_".join(
                [self.__app__ , self.__module__, self.__package__, __s_fn_id__])

            try:
                _read_data = func(self, folder_path, file_name, db_tbl, **kwargs)
                if not isinstance(_read_data,DataFrame) or _read_data.count()<=0:
                    raise AttributeError("Invalid data empty %s" % type(_read_data).upper())

                ''' get the StructType of table '''
                _tbl_cols = ",".join([*_read_data.columns])
                _query = f"SELECT {_tbl_cols} FROM {self._dbSchema}.{db_tbl} LIMIT 1"
                _tbl_schema = clsSDB.read_data_from_table(select=_query).schema
                ''' recreate dataframe with table structure '''
                self._data = clsSDB.session.createDataFrame(
                    data=_read_data.collect(),
                    schema=_tbl_schema)

                ''' set primary key column '''
                if "PKATTR" in kwargs.keys() and "".join(kwargs['PKATTR'].split())!="":
                    _pk_col = kwargs['PKATTR']
                else:
                    ''' try first column as pk '''
                    _pk_col = self._data.columns[0]

                ''' INSERT select NEW rows without primary key '''
                _no_pk_data = self._data.filter(F.col(_pk_col).isNull() | \
                                                F.col(_pk_col).isin(['','NaN','null']))
                ''' UPDATE select OLD rows with primary key '''
                _with_pk_data = self._data.filter(F.col(_pk_col).isNotNull() | \
                                                ~F.col(_pk_col).isin(['','NaN','null']))
                logger.debug("%s partitioned %d update rows with pk and %d insert rows without pk", 
                             __s_fn_id__, _with_pk_data.count(), _no_pk_data.count())

                ''' if any, process INSERT rows '''
                _insert_count=0
                if _no_pk_data.count()>0:
                    ''' augment create process '''
                    if "created_proc" not in _no_pk_data.columns:
                            _no_pk_data = _no_pk_data.withColumn("created_proc", 
                                                                     F.lit(_audit_proc_str))
                    try:
                        _no_pk_data = _no_pk_data.drop(F.col(_pk_col))
                        _insert_count = clsSDB.insert_sdf_into_table(
                            save_sdf =_no_pk_data,
                            db_name = None,
                            db_table= db_tbl,
                        )
                        if not isinstance(_insert_count,int) or _insert_count<=0:
                            raise RuntimeError("Failed to insert %d rows in table %s in %s %s database"
                                               % (_no_pk_data.count(), db_tbl.upper(), 
                                                  self._dbName.upper(), self._dbType.upper()))
                        logger.debug("%s Successfully inserted %d of %d rows in %s %s %s database",
                                     __s_fn_id__, _insert_count, _no_pk_data.count(), 
                                     db_tbl.upper(), self._dbName.upper(), self._dbType.upper())

                    except Exception as ins_err:
                        logger.error("%s %s \n",__s_fn_id__, ins_err)
                        logger.debug(traceback.format_exc())
                        print("[Error]"+__s_fn_id__, ins_err)

                ''' if any, process INSERT rows '''
                _upsert_count=0
                if _with_pk_data.count()>0:
                    ''' augment modify process '''
                    if "modified_proc" not in _with_pk_data.columns:
                            _with_pk_data = _with_pk_data.withColumn(
                                "modified_proc", F.lit(_audit_proc_str))
                    ''' set the options '''
                    _cols_not_for_update = ['created_dt','created_by','created_proc']
                    _options={
                        "BATCHSIZE":1000,   # batch size to partition the dtaframe
                        "PARTITIONS":1,    # number of parallel clusters to run
                        "OMITCOLS":_cols_not_for_update,    # columns to be excluded from update
                    }
                    try:
                        _upsert_count = clsSDB.upsert_sdf_to_table(
                            save_sdf=_with_pk_data,
                            db_name =None,
                            db_table=db_tbl,
                            unique_keys=[_pk_col],
                            **_options
                        )
                        if not isinstance(_upsert_count,int) or _upsert_count<=0:
                            raise RuntimeError("Failed to upsert %d rows in table %s in %s %s database"
                                               % (_no_pk_data.count(), db_tbl.upper(), 
                                                  self._dbName.upper(), self._dbType.upper()))
                        logger.debug("%s Successfully upserted %d of %d rows %s %s %s database",
                                     __s_fn_id__, _upsert_count, _with_pk_data.count(), 
                                     db_tbl.upper(), self._dbName.upper(), self._dbType.upper())
                
                    except Exception as upd_err:
                        logger.error("%s %s \n",__s_fn_id__, upd_err)
                        logger.debug(traceback.format_exc())
                        print("[Error]"+__s_fn_id__, upd_err)

            except Exception as err:
                logger.error("%s %s \n",__s_fn_id__, err)
                logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)

            return _insert_count, _upsert_count

        return wrapper_writer

    def read_csv(func):

        @functools.wraps(func)
        def wrapper_reader(self, folder_path, file_name, db_tbl, **kwargs):
            """
            """
            __s_fn_id__ = f"{self.__name__} function <wrapper_reader>"

            try:
                _read_status = func(self, folder_path, file_name, db_tbl, **kwargs)
                if not _read_status:
                    raise RuntimeError("Failed to initialize process")

                ''' read file into DataFrame '''
                options = {
                    "inferSchema":True,
                    "header":True,
                    "delimiter":",",
                    "pathGlobFilter":'*.csv',
                    "recursiveFileLookup":True,
                }
                self.data = clsFile.read_files_to_dtype(
                    as_type = "spark",
                    folder_path=folder_path,
                    file_name = file_name,
                    file_type = 'csv',
                    **options,
                )
                if not isinstance(self._data,DataFrame) and self._data.count()<=0:
                    raise RuntimeError("Failed to read CSV data from %s got empty %s" 
                                       % (folder_path.upper(), type(self._data)))
                logger.debug("%s read %d rows and %d columns from %s into %s", 
                             __s_fn_id__, self._data.count(), len(self._data.columns), 
                             folder_path, type(self._data))

            except Exception as err:
                logger.error("%s %s \n",__s_fn_id__, err)
                logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)

            return self._data

        return wrapper_reader

    @write_data
    @read_csv
    def save_to_db(
        self,
        folder_path:str = None,
        file_name : str = None,
        db_tbl : str=None,
        **kwargs
    ):
        """
        Desciption:
            File path is mandatory
            If only file_path is provided, then retrieve all csv files from the folder
                to write them to the given database and table name
            If the file name is given, then the specific file in the folder path is read
                and written to the table
        """

        __s_fn_id__ = f"{self.__name__} function <save_to_db>"
        init_complete_ = False

        try:
            ''' validate and set file related properties '''
            if not isinstance(folder_path,str) or "".join(folder_path.split())=="":
                raise AttributeError("Invalid folder_path empty %s" % type(folder_path))
            if "STOREMODE" in kwargs.keys() and "".join(kwargs['STOREMODE'].split())!="":
                self._storeMode = kwargs['STOREMODE']
            clsFile.storeMode = self._storeMode
            if "STOREROOT" in kwargs.keys() and "".join(kwargs['STOREROOT'].split())!="":
                self._storeRoot = kwargs['STOREROOT']
            clsFile.storeRoot = self._storeRoot

            ''' validate and set database related properties '''
            if "DBTYPE" in kwargs.keys() and "".join(kwargs['DBTYPE'].split())!="":
                self._dbType = kwargs['DBTYPE']
            clsSDB.dbType = self._dbType
            if "DBNAME" in kwargs.keys() and "".join(kwargs['DBNAME'].split())!="":
                self._dbName = kwargs['DBTYPE']
            clsSDB.dbName = self._dbName
            if "DBSCHEMA" in kwargs.keys() and "".join(kwargs['DBSCHEMA'].split())!="":
                self._dbSchema = kwargs['DBSCHEMA']
            clsSDB.dbSchema = self._dbSchema
            if "DBUSER" in kwargs.keys() and "".join(kwargs['DBUSER'].split())!="":
                self._dbUser = kwargs['DBUSER']
            clsSDB.dbUser = self._dbUser
            if "DBPSWD" in kwargs.keys() and "".join(kwargs['DBPSWD'].split())!="":
                self._dbPswd = kwargs['DBPSWD']
            clsSDB.dbPswd = self._dbPswd
            if self._dbType.lower() == 'postgresql':
                ''' check if table exists '''
                _db_tbl_lst = clsSDB.get_db_table_info(
                    db_type = self._dbType,
                    db_name = self._dbName,
                    db_schema=self._dbSchema,
                ).select(F.col('table_name')).collect()
                _db_tbl_lst = [x[0] for x in _db_tbl_lst]
                if db_tbl.lower() not in _db_tbl_lst:
                    raise AttributeError("Invalid table %s did you mean %s" 
                                         % (db_tbl.upper(), str(_db_tbl_lst)))
                             
            init_complete_ = True

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return init_complete_
