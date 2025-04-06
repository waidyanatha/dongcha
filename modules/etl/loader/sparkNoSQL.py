#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "sparkNoSQL"
__package__ = "loader"
__module__ = "etl"
__app__ = "rezaware"
__ini_fname__ = "app.ini"
__conf_fname__= "app.cfg"

''' Load necessary and sufficient python librairies that are used throughout the class'''
try:
    import os
    import sys
    import functools
    import configparser    
    import logging
    import traceback
    import re
    from pymongo import MongoClient
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.types import *
    import pandas as pd
    from bson.objectid import ObjectId

    from rezaware.modules.etl.loader import __propAttr__ as attr

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))

'''
    CLASS read and write data to a given location:
        1) local directory
        2) Amazon S3 bucket

    Contributors:
        * nuwan.waidyanatha@rezgateway.com

    Resources:

'''
class dataWorkLoads(attr.properties):
    ''' Function
            name: __init__
            parameters:
                    @name (str)

            procedure: 

            return DataFrame

    '''
    def __init__(self,
                 db_type: str = None, # database type postgres (preferred), mysql, etc
                 db_name : str = None,
                 db_format:str = None,
#                  db_user : str = None,
#                  db_pswd : str = None,
                 db_auth_source : str = None,
                 db_auth_mechanism:str=None,
                 desc:str = "noSQL DB collection and document CRUD",
                 **kwargs,
                ):

        ''' instantiate property attributes '''
        super().__init__(
#             desc=self.__desc__,
            realm="NOSQLDB"
        )

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

        __s_fn_id__ = f"{self.__name__} function <__init__>"
        
#         ''' instantiate property attributes '''
#         super().__init__(
#             desc=self.__desc__,
#             realm="NOSQLDB"
#         )
        
        self._dbType = db_type
        self._dbName = db_name #None
        self._dbFormat = db_format #None
        self._dbAuthSource = db_auth_source       #None
        self._dbAuthMechanism = db_auth_mechanism #None

        ''' initiate to load app.cfg data '''
        global logger
        global pkgConf
        global appConf
        
        try:
            self.cwd=os.path.dirname(__file__)
            pkgConf = configparser.ConfigParser()
            pkgConf.read(os.path.join(self.cwd,__ini_fname__))

            self._rezHome = pkgConf.get("CWDS","PROJECT")
            sys.path.insert(1,self._rezHome)
            ''' initialize the logger '''
            from rezaware.utils import Logger as logs
            logger = logs.get_logger(
                cwd=self._rezHome,
                app=self.__app__, 
                module=self.__module__,
                package=self.__package__,
                ini_file=self.__ini_fname__)
            ''' set a new logger section '''
            logger.info('########################################################')
            logger.info("%s Class %s Package",self.__name__,self.__package__)

            appConf = configparser.ConfigParser()
            appConf.read(os.path.join(self._appDir, self.__conf_fname__))

            logger.debug("%s initialization for %s module package %s %s done. Starting workloads: %s."
                         %(self.__app__.upper(),
                           self.__module__.upper(),
                           self.__package__.upper(),
                           self.__name__.upper(),
                           self.__desc__))
            print("%s Class initialization complete" % self.__name__)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None


    ''' Function - read_doc

            parameters:
                store_mode - local-fs sets to read and write on your local machine file system
                           aws-s3-bucket sets to read and write with an AWS S3 bucket 
            procedure: checks if it is a valid value and sets the mode
            return (str) self._documents

            author: <nuwan.waidyanatha@rezgateway.com>
            
    '''
    def converter(func):

        @functools.wraps(func)
        def wrapper_converter(self,
                 as_type :str,   # mandatory - define the data type to return
                 db_name :str,
                 db_coll :str,      # optional, will retrieve all collections if not specified
                 doc_find:dict={}, # optional - name of the file to read
#                  doc_type:str=None    # optional - read all the files of same type
                  **kwargs,        # HASINAME to 
                ):

            __s_fn_id__ = f"{self.__name__} function <wrapper_converter>"
            self._documents = None
            _the_docs = None

            try:
                _the_docs = func(self,as_type,db_name,db_coll,doc_find, **kwargs)
                ''' validate before proceeding '''
                if not isinstance(_the_docs,list) or len(_the_docs)<=0:
                    raise AttributeError("No docs of %s received to process" % type(_the_docs))

                if as_type.upper() == 'DICT':
                    self._documents = list(_the_docs)
                elif as_type.upper() == 'STR':
                    self._documents=' '.join(list(_the_docs))
                elif as_type.upper() == 'PANDAS':
                    self._documents=pd.DataFrame(_the_docs)
                    if "_id" in self._documents.columns:
                        self._documents['_id'] = self._documents['_id'].astype('str')
                elif as_type.upper() == 'SPARK':
                    self._documents=_the_docs
                else:
                    ''' dtype unspecified return as dictionary '''
                    self._documents=list(_the_docs)

            except Exception as err:
                logger.error("%s %s \n",__s_fn_id__, err)
                logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)

            return self._documents

        return wrapper_converter

    @converter
    def read_documents(
        self,
        as_type:str = "",
        db_name:str = "",
        db_coll:list= [],
        doc_find:dict={},
        **kwargs):

        __s_fn_id__ = f"{self.__name__} function <read_documents>"

        _appName = " ".join([
            self.__app__,
            self.__name__,
            self.__package__,
            self.__module__
        ])   # spark app name

        doc_list_ = None
        doc_dics = None
        _docs_sdf = None

        try:
#             if db_name is not None and "".join(db_name.split())!="":
#                 self.dbName = db_name
            self.dbName = db_name
            if "DBAUTHSOURCE" in kwargs.keys():
                self.dbAuthSource = kwargs['DBAUTHSOURCE']
            elif self.dbAuthSource is None:
                self.dbAuthSource = self.dbName
            else:
                pass
            ''' read collections list from DB '''
            if isinstance(db_coll,list) and len(db_coll)>0:
                self.collections={"COLLLIST":db_coll}
            elif isinstance(kwargs,dict):
                self.collections=kwargs
            if self. _collections is None or len(self._collections)<=0:
                raise ValueError("%s database: %s has no collections" 
                                        % (self._dbType, self._dbName))
            logger.debug("%s Filtered %d collection(s) %s ... from %s %s",
                         __s_fn_id__,len(self._collections),
                         str(self._collections[:3]),self._dbType, self._dbName)

            if doc_find is None:
                doc_find = {}

#             logger.debug("%s Prepared to read documents from "+\
#                          "database %s, %d collection %s with %s find condition",
#                          __s_fn_id__,self.dbName,len(self.collections),
#                          str(self.collections[:3])+"...",doc_find)

#             if self.dbType.lower() == 'mongodb':
#                 ''' get data from MongoDB collection '''
#                 db = self.connect[self.dbName]
#                 _coll_list = db.list_collection_names()
#                 if _coll_list is None or len(_coll_list)<=0:
#                     raise DatabaseError("%s database: %s has no collections" 
#                                         % (self._dbType, self._dbName))
#                 logger.debug("%s %s database: %s has %d collections",
#                              __s_fn_id__,self._dbType, self._dbName,len(_coll_list))
#                 if self.collections:
# #                     logger.debug("Filtering collections by %s",str(self.collections))
#                     self._collections = list(filter(lambda _coll: 
#                                                     _coll in self.collections, 
#                                                     _coll_list
#                                                    ))
#                 else:
#                     logger.debug("%s No filters appied collections are %s",
#                                  __s_fn_id__,str(self.collections))
#                     self._collections = _coll_list

#                 if self.collections is not None and len(self.collections)
#                 logger.debug("%s Filtered %d set of collection %s",
#                              __s_fn_id__,str(self.collections[:3]))

            ''' read data from all the collections '''
            if as_type.upper() == "SPARK":
                ''' read with spark '''
#                     spark = SparkSession.builder.appName(_appName).getOrCreate()
#                     empty_rdd = spark.sparkContext.emptyRDD()
#                     _docs_sdf = spark.createDataFrame(data=empty_rdd,schema=StructType([]))
#                     _docs_sdf = spark.emptyDataFrame()
#                     _inp_uri = f"{self.dbType}://"+\
#                                 f"{self.dbUser}:"+\
#                                 f"{self.dbPswd}@"+\
#                                 f"{self.dbHostIP}/"+\
#                                 f"{self.dbName}."
                for _coll_idx,_coll in enumerate(self._collections):
                    try:
                        _inp_uri = f"{self.dbType}://"+\
                                    f"{self.dbUser}:"+\
                                    f"{self.dbPswd}@"+\
                                    f"{self.dbHostIP}/"+\
                                    f"{self.dbName}."+\
                                    f"{_coll}"+ \
                                    f"?authSource={self.dbAuthSource}"
#                             _inp_uri = _inp_uri + f"{_coll}" + f"?authSource={self.dbAuthSource}"
#                             logger.debug("%s input uri: %s",__s_fn_id__,_inp_uri)

                        # Create Spark session
                        spark = SparkSession.builder \
                            .appName(_appName) \
                            .master(self.sparkMaster) \
                            .config("spark.mongodb.input.uri", _inp_uri) \
                            .getOrCreate()

                        sdf = spark.read.format(self.dbFormat)\
                            .option( "uri", _inp_uri)\
                            .load()

                        if _coll_idx == 0:
                            _docs_sdf = sdf.alias('_docs_sdf')
                        else:
                            _old_docs_sdf = _docs_sdf.alias('_docs_sdf')
                            _docs_sdf = _old_docs_sdf.unionByName(sdf)
                            logger.debug("%s",str(_docs_sdf.head(10)))
                        logger.debug("%s Union sdf size %d",__s_fn_id__,_docs_sdf.count())

                    except Exception as err:
                        logger.warning("%s collection: %s in database: %s had errors: %s \n",
                                       __s_fn_id__,_coll, self.dbName, err)
                        logger.error(traceback.format_exc())
                        pass

                if (not _docs_sdf is None) and (_docs_sdf.count() > 0):
                    doc_list_=_docs_sdf

#                 logger.info("%s %s method Loaded %d documents from %d collections",
#                             __s_fn_id__,as_type.upper(), doc_list_.count(),len(self.collections))
            else:
                ''' read with pymongo '''
                doc_list_=[]
                if self.dbType.lower() == 'mongodb':
                    ''' get data from MongoDB collection '''
                    db = self.connect[self.dbName]

                    for _coll in self._collections:
                        try:
#                             logger.debug("%s Find %s in %s collection",__s_fn_id__,doc_find,_coll)
                            _coll_cur = db[_coll].find(doc_find)
                            if len(list(_coll_cur.clone())) <=0:
                                raise ValueError("No data")
                            doc_list_.extend(list(_coll_cur.clone()))

                        except Exception as err:
                            logger.warning("%s collection: %s in database: %s had errors: %s \n",
                                           __s_fn_id__,_coll, self._dbName, err)
                            pass

            if doc_list_ is None or len(doc_list_)<=0:
                raise RuntimeError("doc_list_ is an empty %s object" % type(doc_list_))
            logger.debug("%s %s method Loaded %d documents from %d collections",
                        __s_fn_id__,as_type.upper(), len(doc_list_),len(self._collections))

#             elif self.dbType.lower() == 'cassandra':
#                 ''' get data from Cassandra collection '''
#                 raise RuntimeError("cassandra read is tbd")
#             else:
#                 raise ValueError("Something was wrong")

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return doc_list_

    ''' Function - write collection

            parameters:

            author: <nuwan.waidyanatha@rezgateway.com>
            
    '''
    def createDocs(func):

        @functools.wraps(func)
        def create_docs_wrapper(
            self,
            db_name:str,   # optional - name of the file to read
            db_coll:str,   # mandatory - relative path, w.r.t. self.storeRoot
            data,   # data to be stored
            uuid_list,
            **kwargs,
        ):

            __s_fn_id__ = f"{self.__name__} function <create_docs_wrapper>"

            try:
                if isinstance(data,list):
                    ''' list to collection '''
                    self._documents = data
                elif isinstance(data,dict):
                    self._documents = [data]
                elif isinstance(self.documents,pd.DataFrame):
                    ''' dataframe to collection '''
                    self._documents = [self.documents.to_dict()]
                else:
                    raise TypeError("Unrecognized data type %s must be either of\n%s"
                                    % (type(self.documents),str(self._asTypeList)))

                _collection=func(self,
                                 db_name,
                                 db_coll,
                                 data,
                                 uuid_list,
                                 **kwargs,
                                )

    #             if self.dbType.lower == 'mongodb':
    #                 _insert_ids = _collection.insert_many(self.documents)

    #             elif self.dbType.lower() == 'cassandra':
    #                 ''' get data from cassandra collection '''
    #                 raise RuntimeError("cassandra write is tbd")
    #             else:
    #                 raise ValueError("Something was wrong")

            except Exception as err:
                logger.error("%s %s \n",__s_fn_id__, err)
                logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)

            return self._documents

        return create_docs_wrapper

    @createDocs
    def write_documents(
        self,
        db_name:str,   # optional - name of the file to read
        db_coll:str,   # mandatory - relative path, w.r.t. self.storeRoot
        data=None,   # data to be stored
        uuid_list:list=[],   # unique identifier name to check if document exists
        **kwargs
    ):
        """
        Description:
        Attributes :
        Returns :
        Exceptions :
        """

        __s_fn_id__ = f"{self.__name__} function <write_documents>"
        _collection = None, 
        _objIds = None

        try:
            ''' set connection parameters from kwargs '''
            if "DBUSER" in kwargs.keys() \
                and isinstance(kwargs['DBUSER'],str) \
                and "".join(kwargs['DBUSER'].split())!="":
                self.dbUser=kwargs['DBUSER']
            if "DBPSWD" in kwargs.keys() and \
                isinstance(kwargs['DBPSWD'],str) \
                and "".join(kwargs['DBPSWD'].split())!="":
                self.dbUser=kwargs['DBPSWD']
            if "DBAUTHSOURCE" in kwargs.keys() \
                and isinstance(kwargs['DBAUTHSOURCE'],str) \
                and "".join(kwargs['DBAUTHSOURCE'].split())!="":
                self.dbUser=kwargs['DBUSER']
            if "DBAUTHMECHANISM" in kwargs.keys() \
                and isinstance(kwargs['DBAUTHMECHANISM'],str) \
                and "".join(kwargs['DBAUTHMECHANISM'].split())!="":
                self.dbUser=kwargs['DBAUTHMECHANISM']

            logger.debug("%s Begin writing document to %s",__s_fn_id__,self.dbType.upper())

            ''' check if collection exists; else create one '''
            if self.dbType.lower() == 'mongodb':
                db = self.connect[db_name]
                ''' confirm database exists '''
                if not db_name in self.connect.list_database_names():
                    if "FORCEDB" in kwargs.keys() and kwargs['FORCEDB']==True:
                        ''' create a new database '''
                        db_name = self.create_db(
                            db_name = db_name,
                            db_user = self._dbUser, # 'rezawareAdmin',
                            db_pswd = self._dbPswd, # 'rezaware',
                            db_authSource=self._dbAuthSource,    # 'admin',
                            db_authMech = self._dbAuthMechanism, # 'SCRAM-SHA-256',
                        )
                        logger.debug("%s forced to create new database %s", __s_fn_id__, db_name.upper())
                        ''' reconnect to newly created db '''
                        db = self.connect[db_name]
                    else:
                        raise RuntimeError("database %s does not exist" % db_name.upper())
                ''' get data from MongoDB collection '''
                if db_coll not in db.list_collection_names():
                    _collection = db[db_coll]
                    logger.info("Created a new collection %s",_collection)
                    ''' insert all the documents '''
                    results = db[db_coll].insert_many(self.documents)
                    logger.info("Inserted %d documents",db[db_coll].count_documents({}))
                else:
                    _insert_count = 0
                    _modify_count = 0
                    for data_dict in self.documents:
#                         ''' add ObjectIds if not in data '''
#                         if '_id' not in data_dict.keys() \
#                             or data_dict['_id'] is None \
#                             or data_dict['_id']=="":
#                             new_id = ObjectId()
#                             while not db[db_coll].find({"_id": {"$eq": ObjectId(new_id)}}):
#                                 new_id = ObjectId()
#                             data_dict['_id']=ObjectId(new_id)

                        ''' find and update on the key list '''
                        if not (len(uuid_list) > 0):
#                             _filter_dict = {}
#                             for _key in uuid_list:
#                                 _filter_dict[_key]=data_dict[_key]
                            uuid_list = list(data_dict.keys())
#                         else:
#                             uuid_list = data_dict.keys()

#                         print(uuid_list)
                        _filter_dict = {}
                        for _uuid in uuid_list:
                            _filter_dict[_uuid]=data_dict[_uuid]
#                         print(_filter_dict)
                        _filtered_cur = db[db_coll].find(_filter_dict)
#                         for x in _filtered_cur.clone():
#                             print(x)
                        ''' if no match then insert with new ObjectID '''
                        if len(list(_filtered_cur.clone())) <=0:
                            ''' add ObjectIds if not in data '''
                            doc = db[db_coll].insert_one(data_dict)
                            _insert_count += 1
#                             print('insert count:',doc.inserted_count)
#                             logger.info("Modified %d documents",doc.inserted_count)
#                             if '_id' not in data_dict.keys() \
#                                 or data_dict['_id'] is None \
#                                 or data_dict['_id']=="":
# #                                 _filtered_cur = data_dict
#                                 new_id = ObjectId()
#                                 while not db[db_coll].find({"_id": {"$eq": ObjectId(new_id)}}):
#                                     new_id = ObjectId()
#                                 data_dict['_id']=ObjectId(new_id)
# #                                 _filtered_cur['_id']=ObjectId(new_id)
#                             ''' insert the new data '''
#                             _filtered_cur.append(data_dict)
#                             print('inserted',doc)
                        else:
                            ''' update all matching documents '''
                            for _cur in _filtered_cur:

                                ''' update with existing ObjectId '''
#                                 doc = db[db_coll].update_one({'_id':_cur['_id']},{"$set": data_dict})
                                doc = db[db_coll].update_one(
                                    filter={
                                        '_id' : _cur['_id'],
                                    },
                                    update={
    #                                             '$setOnInsert': data_dict,
                                        '$set': data_dict,
                                    },
                                    upsert=True,
                                )
                                _modify_count += 1
                    print('Total %d documents, successful insert count = %d & modify count = %d'
                          %(len(self.documents),_insert_count, _modify_count))
                    logger.info("Total %d documents, successful insert count = %d & modify count = %d",
                                len(self.documents),_insert_count, _modify_count)

            elif self.dbType.lower() == 'cassandra':
                ''' get data from cassandra collection '''
                raise RuntimeError("cassandra write is tbd")
            else:
                raise ValueError("Something was wrong")

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return db[db_coll]

    ''' Function --- CREATE DATABASE ---

        author <nuwan.waidyanatha@rezgateway.com>
    '''
    def create_db(
        self,
        db_type : str = None,
        db_name : str = None,   # Valid string
        db_user : str = None,
        db_pswd : str = None,
        db_authSource : str = None,
        db_authMech:str=None,
        **kwargs,
    ) -> str:
        """
            Description:
                Create a database, for the set dbType with given name, authemtication, and
                authentication mechanism. 
                If a username and password is not provided, then the system user and the system
                paswword is used.
            Attributes:
            
        """
        __s_fn_id__ = f"{self.__name__} function <create_db>"

        try:
            self.dbType = db_type
            self.dbName = db_name
            self.dbUser = db_user
            self.dbPswd = db_pswd
            self.dbAuthSource=db_authSource
            self.dbAuthMechanism=db_authMech

            if self._connect is None:
                self.connect={}
            try:
                _dbs = self._connect.list_database_names()
                if self._dbName in _dbs:
                    logger.debug("%s %s is 1 of %d databases in %s",
                                 __s_fn_id__,self._dbName,len(_dbs),self._dbType)
                else:
                    db = self.connect[self._dbName]
                    coll = db["about"]
                    _about_doc = {"name":self._dbName,"desc":self._dbName}
                    ret_ = coll.insert_one(_about_doc)
                    logger.debug("%s id:%s inserted about collection in %s",
                                 __s_fn_id__,ret_.inserted_id,self._dbName)

            except Exception as con_err:
                logger.warning("%s %s \n",__s_fn_id__, con_err)
                logger.warning("%s %s does not exist, creating new database"
                               ,__s_fn_id__,self._dbName)
        
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._dbName


    ''' convert to dict '''
    def convert_2_dict_mongodb(obj):
        result = {}
        for key, val in obj.items():
            if not isinstance(val, dict):
                result[key] = val
                continue

            for sub_key, sub_val in val.items():
                new_key = '{}.{}'.format(key, sub_key)
                result[new_key] = sub_val
                if not isinstance(sub_val, dict):
                    continue

                result.update(convert_2_dict_mongodb(result))
                if new_key in result:
                    del result[new_key]

        return result