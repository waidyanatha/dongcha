#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "reference"
__package__= "utils"
__module__ = "lib"
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
    from datetime import datetime, date, timedelta

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))

'''
    CLASS configure the master hotel details, groups, lookups, and locations

    Contributors:
        * nuwan.waidyanatha@rezgateway.com

    Resources:

'''

class dataWorkLoads():

    ''' Function --- INIT ---

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    def __init__(
        self, 
        desc,
        realm:str= None,
        category:str=None,
        **kwargs):
        """
        Decription:
            Initializes the ExtractFeatures: class property attributes, app configurations, 
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

        self._tbl_name = "util_refer"
        self._data = None
        self._realm= realm
        self._realm_list=None
        self._category = category
        self._category_list=None

        global pkgConf
        global appConf
        global logger
        global clsSDB
        global clsFile

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
            logger.info("%s Class",self.__name__)

            ''' import sparkRDBM dataworkload to read/write data from/to rdbms'''
            from rezaware.modules.etl.loader import sparkRDBM as db
            clsSDB = db.dataWorkLoads(
                desc=self.__desc__,
            )
            if ("DBTYPE" in kwargs.keys() and "".join(kwargs['DBTYPE'].split())!="")\
                or ("DBDRIVER" in kwargs.keys() and "".join(kwargs['DBDRIVER'].split())!="")\
                or ("DBHOSTIP" in kwargs.keys() and "".join(kwargs['DBHOSTIP'].split())!="")\
                or ("DBPORT" in kwargs.keys() and "".join(kwargs['DBPORT'].split())!="")\
                or ("DBNAME" in kwargs.keys() and "".join(kwargs['DBNAME'].split())!="")\
                or ("DBSCHEMA" in kwargs.keys() and "".join(kwargs['DBSCHEMA'].split())!="")\
                or ("DBUSER" in kwargs.keys() and "".join(kwargs['DBUSER'].split())!="")\
                or ("DBPSWD" in kwargs.keys() and "".join(kwargs['DBPSWD'].split())!=""):
                clsSDB.dbConnURL = kwargs
            if ("CONFIG" in kwargs.keys() and "".join(kwargs['CONFIG'].split())!="")\
                or ("MASTER" in kwargs.keys() and "".join(kwargs['MASTER'].split())!="")\
                or ("APPNAME" in kwargs.keys() and "".join(kwargs['APPNAME'].split())!="")\
                or ("JARDIR" in kwargs.keys() and "".join(kwargs['JARDIR'].split())!=""):
                clsSDB.session=kwargs
            ''' import sparkFILE dataworkload to read/write data from/to file storage'''
            from rezaware.modules.etl.loader import sparkFile as file
            _store_mode=None
            if "STOREMODE" in kwargs.keys() and "".join(kwargs['STOREMODE'].split())!="":
                _store_mode = kwargs['STOREMODE']
            _store_root=None
            if "STOREROOT" in kwargs.keys() and "".join(kwargs['STOREROOT'].split())!="":
                _store_root = kwargs['STOREROOT']
            clsFile = file.dataWorkLoads(
                desc=self.__desc__,
                f_store_mode= _store_mode,
                f_store_root= _store_root,
            )

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


    ''' Function --- CLASS PROPERTY SETTER & GETTER ---

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    ''' --- DATA --- '''
    @property
    def data(self):

        __s_fn_id__ = f"{self.__name__} function <@property data>"

#         __def_tbl_name__ = "util_refer"

        try:
            ''' validate property value '''
            if self._data is  None:
                _query = f"SELECT * FROM {clsSDB.dbSchema}.{self._tbl_name} "
                _query +=f"WHERE deactivate_dt IS NULL "
                self._data = clsSDB.read_data_from_table(select=_query)
                if self._data is None or self._data.count()<=0:
                    raise RuntimeError("Failed to retrieve %s realm data" % self._realm)
                logger.warning("%s read %d rows of data for %s realm", 
                               __s_fn_id__, self._data.count(), self._realm)
            if not isinstance(self._data,DataFrame):
                self._data = self.session.createDataFrame(self._data)
                logger.warning("%s non-pyspark dataset converted to %s with %d rows and %d columns", 
                               __s_fn_id__, type(self._data), 
                               self_.data.count(), len(self_.data.columns))
            if self._data.count() <= 0:
                raise AttributeError("No records in property data; empty %s" % type(self._data)) 
                
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
                raise AttributeError("Cannot set class property data with empty %s" % type(data))
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


    ''' --- REALM LIST --- '''
    @property
    def realm_list(self):

        __s_fn_id__ = f"{self.__name__} function <@property realm_list>"

        try:
            ''' validate realm_list value '''
            if not isinstance(self._realm_list, list) or len(self._realm_list)<=0:
                self._realm_list = [x[0] for x in 
                                     clsSDB.get_db_table_info(
                                         db_type=clsSDB.dbType,
                                         db_name=clsSDB.dbName,
                                         db_schema=clsSDB.dbSchema,
                                     )\
                                     .select("table_name").collect()]
                
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._realm_list


    ''' --- REALM --- '''
    @property
    def realm(self):

        __s_fn_id__ = f"{self.__name__} function <@property realm>"

        try:
            ''' validate realm value '''
            if not isinstance(self._realm, str) or "".join(self._realm.split())=="":
                raise AttributeError("Unspecified class property realm, %s" % type(self._realm))
                
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._realm

    @realm.setter
    def realm(self,realm):

        __s_fn_id__ = f"{self.__name__} function <@realm.setter>"

        try:
            ''' validate property value '''
            if realm not in self.realm_list:
                raise AttributeError("Cannot set class property with empty %s; specify one of %s" 
                                     % type(realm), self._realm_list)

            self._realm = realm

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._realm

    ''' --- CAEGORY --- '''
    @property
    def category(self):

        __s_fn_id__ = f"{self.__name__} function <@property category>"

        try:
            ''' validate category value '''
            if self._category not in self.category_list:
                raise AttributeError("Unspecified class property category, %s" % type(self._category))
                
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._category

    @category.setter
    def category(self,category):

        __s_fn_id__ = f"{self.__name__} function <@category.setter>"

        try:
            ''' validate property value '''
            if not isinstance(category, str) or "".join(category.split())=="":
                raise AttributeError("Cannot set class property with empty %s" 
                                     % type(category))

            self._category = category

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._category


    ''' --- CATEGORY LIST --- '''
    @property
    def category_list(self):

        __s_fn_id__ = f"{self.__name__} function <@property category_list>"

        try:
            ''' validate category_list value '''
            if not isinstance(self._category_list, list) or len(self._category_list)<=0:
                self._category_list = [x[0] for x in 
                                       self.data\
                                       .filter(F.col('realm').isin(self._realm))\
                                       .select("category").distinct().collect()]
                
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._category_list


    ''' Function --- GET REFERENCE ---

        author(s): <nuwan.waidyanatha@rezgateway.com>
    '''
    def get_reference(
        self,
        realm: str = None,
        category:str=None,
        **kwargs
    ) -> DataFrame:
        """
        """

        __s_fn_id__ = f"{self.__name__} function <@property get_reference>"

        try:
            filt_ref_=self.data
            if isinstance(realm,str) and "".join(realm.split())!="":
                self.realm = realm
                filt_ref_=filt_ref_.filter(F.col('realm').isin(self._realm))
            if isinstance(category,str) and "".join(category.split())!="":
                self.category=category
                filt_ref_=filt_ref_.filter(F.col('category').isin(self._category))

            if not isinstance(filt_ref_, DataFrame) or filt_ref_.count()<=0:
                raise RuntimeError("returned %s empty reference data for %s realm and %s category" 
                                   % (type(filt_ref_), self._realm, self._category))
            self._data = filt_ref_
            logger.debug("%s filtered %d rows for %s realm and %s category", 
                         __s_fn_id__, self._data.count(), self._realm, self._category)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data


    ''' Function --- IMPORT TO RDBM  ---

        author(s): <nuwan.waidyanatha@rezgateway.com>
    '''
    def uspert_data(func):

        @functools.wraps(func)
        def upsert_wrapper(self,folder_path,file_type,db_name,db_schema,**kwargs):
            """
            """

            __s_fn_id__ = f"{self.__name__} function <upsert_wrapper>"
            
            _tbl_name='util_refer'
            _pk = 'ref_pk'
            _cols_not_for_update = ['created_dt','created_by','created_proc']
            _options={
                "BATCHSIZE":1000,   # batch size to partition the dtaframe
                "PARTITIONS":1,    # number of parallel clusters to run
                "OMITCOLS":_cols_not_for_update,    # columns to be excluded from update

            }

            try:
                ref_data_ = func(self,folder_path,file_type,db_name,db_schema,**kwargs)
                if not isinstance(ref_data_,DataFrame) or ref_data_.count()<=0:
                    raise AttributeError("Cannot process empty %s dataset" % type(ref_data_))

                ''' verify database attributes '''
                if not isinstance(db_name,str) or "".join(db_name.split())=="":
                    logger.warning("%s Unspefied empty db_name %s set to default value %s",
                                   __s_fn_id__, type(db_name), clsSDB.dbName)
                    db_name = clsSDB.dbName
                if not isinstance(db_schema,str) or "".join(db_schema.split())=="":
                    logger.warning("%s Unspefied empty db_schema %s set to default value %s",
                                   __s_fn_id__, type(db_schema), clsSDB.dbSchema)
                    db_schema = clsSDB.dbSchema

#                 tmp_sdf = clsSDB.read_data_from_table(
#                     db_table=_tbl_name,
#                 )

#                 tmp_sdf = tmp_sdf.drop(*[x for x in tmp_sdf.columns 
#                                          if x not in _util_ref_sdf.columns])
                ''' get the table structure to cast the data '''
                _tbl_schema = clsSDB.get_table_schema(
                    tbl_name=_tbl_name,
                    db_schema=db_schema,
                )
                _tbl_fields_lst=None
                _tbl_fields_lst = [field.name for field in _tbl_schema.fields]
                if len(_tbl_fields_lst)<=0:
                    raise ValueError("Unable to recover field names from table %s, returned empty %s" 
                                     % (_tbl_name, type(_tbl_fields_lst)))
                logger.debug("%s %s fields list: %s", __s_fn_id__, _tbl_name, _tbl_fields_lst)
                _filter_sdf = ref_data_.drop(*[x for x in ref_data_.columns 
                                         if x not in _tbl_fields_lst])
                if len(_filter_sdf.columns)<=0:
                    raise UnboundLocalError("Unmatched schema columns returned %d filtered columns"
                                            % len(_filter_sdf.columns))
                logger.debug("%s resulted in %s columns \nafter dropping columns not in %s schema \n%s",
                             __s_fn_id__, _filter_sdf.columns, _tbl_name, _tbl_fields_lst)
                self._data = clsSDB.session.createDataFrame(
                    data=_filter_sdf.collect(), schema=_filter_sdf.schema)
                logger.debug("%s created %d rows in dataframe with %s table schema", 
                             __s_fn_id__, self._data.count(), _tbl_name)

                ''' create insert (no PK) and upsert datasets (with PK)'''
                _ins_sdf = self._data.filter(F.col(_pk).isNull() |
                                             F.col(_pk).isin('','NaN','None','none'))\
                                        .select('*')
                _upd_sdf = self._data.filter(F.col(_pk).isNotNull() |
                                             ~F.col(_pk).isin('','NaN','None','none'))\
                                        .select('*')
                _ins_count, _upd_count = 0, 0
                if _ins_sdf.count()>0:
                    _ins_sdf=_ins_sdf.drop(_pk)
                    _records=clsSDB.insert_sdf_into_table(
                        save_sdf=_ins_sdf,
                        db_name =db_name,
                        db_table=_tbl_name,
                    )

                if _upd_sdf.count()>0:
                    _upd_count=clsSDB.upsert_sdf_to_table(
                    save_sdf=_upd_sdf,
                    db_name =db_name,
                    db_table=_tbl_name,
                    unique_keys=[_pk],
                    **_options,
                )


            except Exception as err:
                logger.error("%s %s \n",__s_fn_id__, err)
                logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)

            return _ins_count, _upd_count #self._data

        return upsert_wrapper

    @uspert_data
    def import_to_db(
        self,
        folder_path:str=None,
        file_type: str= None,
        db_name : str = None,
        db_schema:str = None,
        **kwargs
    ) -> DataFrame:
        """
        """

        __s_fn_id__ = f"{self.__name__} function <@property import_to_rdbm>"

        __def_file_type__="csv"
        __as_type__ = "spark"  # specify data type to return from read file

        try:
            ''' verify file attributes '''
            if not isinstance(folder_path,str) or "".join(folder_path.split())=="":
                raise AttributeError("Unspecified file_path empty %s" % type(file_path))
            if file_type not in ['csv','json']:
                logger.warning("%s unspecified empty %s file_type attribute set to default %s",
                               __s_fn_id__, type(file_type), __def_file_type__)
                file_type = __def_file_type__

            ''' options key value pairs must match the pyspark standard'''
            options = {
                "inferSchema":True,
                "header":True,
                "delimiter":",",
                "pathGlobFilter":".".join(['*',file_type]),
                "recursiveFileLookup":True,
            }
            _util_ref_sdf = None
            _util_ref_sdf = clsFile.read_files_to_dtype(
                as_type=__as_type__,
                folder_path=folder_path,
                file_name=None,
                file_type=file_type,
                **options,
            )

            if not isinstance(_util_ref_sdf, DataFrame) or _util_ref_sdf.count()<=0:
                raise RuntimeError("read_files_to_dtype returned %s empty reference data" 
                                   % type(_util_ref_sdf))
            self._data = _util_ref_sdf
            logger.debug("%s process read %d rows from folder %s with file type %s", 
                         __s_fn_id__, self._data.count(), folder_path.upper(), file_type.upper())

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data


