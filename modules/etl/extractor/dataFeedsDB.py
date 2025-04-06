#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "FeedWorkLoads"
__module__ = "etl"
__package__ = "extractor"
__app__ = "rezaware"
__ini_fname__ = "app.ini"
__conf_fname__ = "app.cfg"

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

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))


class FeedWorkLoads():
    """ ***************************************************************************************
    CLASS spefic to storing and retrieving data source feeds information
    
    The overall function here is to provide methods for storing the various domain, locations,
    and any other feature specific data source feed hooks for retrieving data. 
    
    We implement mongodb and pyspark to perform the the data source feeds management package.
    class is instantiated, it will inherit properties and methods from packages 
    (1) rezaware/etl/loader/sparkNoSQLwls - to read/write data from NoSQL DB collections & documents
    
    contributors:
        * nuwan.waidyanatha@rezgateway.com
        
    ******************************************************************************************* 
    """
    def __init__(self,
                 desc:str="noSQL DB collection CRUD",
                 **kwargs,
                ):
        """
        Description:
            Initializes the FeedWorkloads: class property attributes, app configurations, 
                logger function, data store directory paths, and global classes
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
        self.__desc__ = desc
        ''' defines the realm madatory elements '''
        self._realm = {
            "module" : None,
            "entity" : None,
            "package" :None,
            "function":None
        } # is associated with the module or domain taxanomy of the data source
        ''' defines uri mandatory elements '''
        self._uri = {
            "urn" : None,
            "protocol": None,
            "domain" : None,
            "port" : None,
            "path" : None,
            "query": None,
            "fragments" :None,
        }
        ''' defines the madatory dict element '''
        self._query = {
            "expression":None,
            "parameter" :None,             
        }
        ''' defines a single feed dict '''
        self._feed = {
            "source" :None, # information about the data source owner, activer periods, and so on
            "context":None, # taxonomy for categorizing the data feed
            "realm" : None, # uniquely identifies by the module, entity, package, function framework
            "uri" : None,   # the domain, port, path, fragments, query, parameter, etc information for setting the feed
            "get" : None    # describes the method and data object the feeds will return 
        }
        self._feedsList=None # holds a list of feeds dictionaries defined by self._feed

        ''' initiate to load app.cfg data '''
        global logger
        global pkgConf
        global appConf
        global clsNsql
        
        __s_fn_id__ = f"{self.__name__} function <__init__>"
        
        try:
            self.cwd=os.path.dirname(__file__)
            pkgConf = configparser.ConfigParser()
            pkgConf.read(os.path.join(self.cwd,__ini_fname__))
#             appConf = configparser.ConfigParser()
#             appConf.read(os.path.join(self.cwd,__conf_fname__))

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
            logger.info("%s Class %s Package",self.__name__,self.__package__)
            logger.debug("%s initialization for %s module package %s %s done. Starting workloads: %s."
                         %(self.__app__.upper(),
                           self.__module__.upper(),
                           self.__package__.upper(),
                           self.__name__.upper(),
                           self.__desc__))

            from rezaware.modules.etl.loader import sparkNoSQLwls as nosql
            clsNsql=nosql.NoSQLWorkLoads(desc=self.__desc__)
            print("%s Class initialization complete" % self.__name__)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None


    ''' Function --- realm ---
        author: <nuwan.waidyanatha@rezgateway.com>
    '''
    @property
    def realm(self) -> dict:
        """ 
            Definition: 
                supports the class property for getting the instance specific realm dict. 
                The data must not be None-Type and must be a python dict with valid entries
                for the keys='module','entity','package','function'.

            returns
                self._realm (dict)
            Exceptions: 
               if None or Empty dict, throws AttributeError
        """

        __s_fn_id__ = f"{self.__name__} function <@property realm>"

        try:
            if not isinstance(self._realm,dict):
                raise AttributeError("Undefined realm of type %s", type(self._realm))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._realm

    @realm.setter
    def realm(self,realm:dict) -> dict:

        __s_fn_id__ = f"{self.__name__} function <@realm.setter>"
        
#         realm_keys = ['module','entity','package','function']

        try:
            ''' validate realm data structure '''
            if not isinstance(realm,dict) and len(realm)<=0:
                raise AttributeError("Cannot assign an empty %s type realm" % type(realm))
            if len(list(set(realm.keys()).intersection(set(self._realm.keys())))) < len(self._realm.keys()):
                raise AttributeError("Ivalid Realm keys %s, must be a dict with %s keys" 
                                     % (str(realm.keys()),str(self._realm.keys()).upper()))
            logger.debug("%s %s are vaild",__s_fn_id__,realm.keys())
            if realm['module'] is None or "".join(realm['module'].split())=="" \
                or realm['entity'] is None or "".join(realm['entity'].split())=="":
                raise AttributeError("Ivalid Realm module %s or entity %s cannot be empty" 
                                     % (type(realm['entity']),type(realm['entity'])))
            logger.debug("%s None empty module %s and entity %s",
                         __s_fn_id__,realm['module'].upper(),realm['entity'].upper())

            ''' validate realm modules, entities, packages, and functions '''
            projHome = pkgConf.get("CWDS","PROJECT")
            valid_mod = False
            valid_ent = False
            for app in ['rezaware','mining','wrangler','visuals']:
                app_cfg = configparser.ConfigParser()
                app_cfg.read(os.path.join(projHome,app,__conf_fname__))
                for _mod_key, _mod_val in app_cfg.items('MODULES'):
                    if realm['module'].lower() == _mod_key.lower():
                        valid_mod = True
                        logger.debug("%s realm module %s is valid in %s app",
                                     __s_fn_id__,realm['module'].upper(),app.upper())
                        if realm['entity'].lower() in _mod_val.split(','):
                            valid_ent=True
                            logger.debug("%s realm entity %s is valid in %s app",
                                         __s_fn_id__,realm['entity'].upper(),app.upper())
                            break

            if not (valid_mod and valid_ent):
                raise AttributeError("Invalid module %s or entity %s; check %s files" 
                                     % (realm['module'].upper(),realm['entity'].upper(),
                                        __conf_fname__.upper()))

            self._realm = realm
            logger.debug("%s realm property dict %s set",__s_fn_id__,str(self._realm))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._realm


    ''' Function --- FEED ---
        author: <nuwan.waidyanatha@rezgateway.com
    '''
    @property
    def feed(self) -> dict:

        __s_fn_id__ = f"{self.__name__} function <@property feed>"
        
#         feed_prim_key_list = ['source', 'realm', 'uri', 'get']

        try:
            if not isinstance(self._feed,dict) or len(self._feed)<=0:
                raise AttributeError("Invalid feed empty property %s type" % type(self._feed))
            if len(list(set(self._feed.keys()).intersection(set(self._feed.keys())))) \
                    < len(self._feed.keys()):
                raise AttributeError("Missing feed primary key information, must contain %s" 
                                     % (self._feed.keys()))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._feed

    @feed.setter
    def feed(self,feed:dict) -> dict:

        __s_fn_id__ = f"{self.__name__} function <@feed.setter>"
        
        _prim_elem_list = ['source', 'realm', 'context','uri', 'get']

        try:
            if not isinstance(feed,dict) or len(feed)<=0:
                raise AttributeError("Invalid feed empty property %s type" % type(feed))
            if len(list(set(feed.keys()).intersection(set(self._feed.keys())))) \
                    < len(self._feed.keys()):
                raise AttributeError("Missing feed primary keys %s, madatory keys %s" 
                                     % (feed.keys(),_prim_elem_list))
            self._feed = feed
            logger.debug("%s feed property set for realm %s",__s_fn_id__,self._feed['realm'])

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._feed

    ''' --- FEEDS LIST --- '''
    @property
    def feedsList(self) -> list:
        """
        Description:
            list of data feeds; where a feed is defined by the class property self.feed
        Attributes:
            None
        Returns:
            self._feedsLis(list)feedsList 
        Exceptions:
          feedsListif feedsList is empty, throw AttributeError exception
        """

        __s_fn_id__ = f"{self.__name__} function <@property feedsList>"
        
        try:
            if not isinstance(self._feedsList,list) or len(self._feedsList)<=0:
                raise AttributeError("Invalid empty feedsList property %s type" % type(self._feedsList))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._feedsList

    @feedsList.setter
    def feedsList(self,feeds_list:list) -> list:

        __s_fn_id__ = f"{self.__name__} function <@feedsList.setter>"

        try:
            if not isinstance(feeds_list,list) or len(feeds_list)<=0:
                raise AttributeError("Invalid empty feeds_list setter of %s type" % type(feed_list))

            self._feedsList = feeds_list
            logger.debug("%s property feedsList set with %d feeds",__s_fn_id__,len(self._feedsList))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._feedsList

    ''' --- URI ---
        TODO add propert for uri with dict validation
    '''

    ''' Function --- READ FEEDS TO LIST ---
    
        authors: <nuwan.waidyanatha@rezgateway.com>
    '''
    def read_feeds_to_list(
        self,
        search_with_list:list=[],
        **kwargs
    ) -> list:
        """
            Description:
                Objective is to provide a filtered list of data feeds. The first level of filtering is the database
                name and secondly the collection list. The database name and collection name will be derived from
                the realm. Additionally, the context key/value pairs are used in further filtering the set of
                documents.
            Attributes:
                search_with_list (list)- [mandatory] list of dictionaries with the realm and context information
                **kwargs (dict)) - dictionary of key value pair options to provide for filtering the search list
                    'COLLHASINNAME' (list) of strings to filter collections
            Returns:
                self._feed_list (list) of dictionaries with feed information
            Exceptions:
                if db_name not in NoSQL instance, throw AttributeError
                if colllection not in NoSQL instance, throw AttributeError
        """

        __s_fn_id__ = f"{self.__name__} function <read_feeds_to_list>"

        filtered_doc_list_ = []
        feed_list_ = []

        try:
            if search_with_list is None or len(search_with_list) <=0:
                raise AttributeError("Invalid empty %s type search_with_list" % type(search_with_list))
            for search_dict in search_with_list:
                try:
                    ''' construct database name and collection name '''
                    self.realm = search_dict['realm']
                    _db_name = "_".join([self._realm['module'],self._realm['entity']]).lower()
                    _coll_name="_".join([self._realm['package'],self._realm['function']]).lower()
                    ''' read all documents in collection and database '''
                    _docs_list = None
                    _docs_list = clsNsql.read_documents(
                        as_type = "DICT",
                        db_name = _db_name,
                        db_coll = _coll_name,
                        doc_find={},
                        **kwargs)
                    if _docs_list is None or len(_docs_list)<=0:
                        raise RuntimeError("database %s or collection %s does not exist" % (_db_name,_coll_name))
                    logger.debug("%s database: %s and collection: %s read %d documents ",
                                 __s_fn_id__, _db_name, _coll_name, len(_docs_list))

                    ''' verify context keys and values exist '''
                    if "context" not in search_dict.keys() or len(search_dict['context'])<=0:
                        raise AttributeError("search dictionary has no context key or values to filter documents")

                    ''' loop through the documents to filter by context '''
                    for _doc_dict in _docs_list:
                        try:
                            if "context" not in _doc_dict.keys() \
                                or len(_doc_dict['context'])<=0:
                                raise AttributeError("document of search dictionary has no context key or values")

                            ''' filter documents by context values '''
                            _match_contx_count = 0
                            for contx_key in search_dict['context']:
                                ''' compare lower case values '''
                                if _doc_dict['context'][contx_key].lower() == search_dict['context'][contx_key].lower():
                                    _match_contx_count += 1
                                ''' if perfect matching then add to valid filtered list '''
                                if _match_contx_count == len(search_dict['context']):
                                    filtered_doc_list_.append(_doc_dict)

                        except Exception as contx_err:
                            logger.warning("%s %s \n",__s_fn_id__, contx_err)

                except Exception as doc_err:
                    logger.warning("%s %s \n",__s_fn_id__, doc_err)

            if len(filtered_doc_list_)<=0:
                raise RuntimeError("Invalid empty filtered_doc_list_; of type %s" % type(filtered_doc_list_))
            self._feedsList = filtered_doc_list_
            logger.debug("%s process completed with %d feeds added to feedsList", __s_fn_id__, len(self._feedsList))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._feedsList


    ''' Function --- WRITE DATA FEED ---
        author <nuwan.waidyanatha@rezgateway.com 
    '''
    def write_feeds_to_nosql(
        self,
        feed_list:list=[],
        **kwargs,
    ) -> list:
        """
            Definition:
                The function takes a list of structured feed dictionaries, comprising a: 
                * source - owner, a summary describing the utility, a set of key value pairs
                    for categorizing the feed, and a dates that define the active status
                * context - key/value pairs, specific to the data feed, necessary for further tagging
                    and categorizing the data feed; i.e. sets the taxonomy 
                * realm - that defines the module, entity, package, and function for creating
                    the database and collection for storing the data
                * uri - (univeral resource identifier) with spefics of the feed protocol, domain, 
                    port, path, query, fragment, 
                * get - methods of the feed (e.g., download) and the receiving data object 
                    (e.g. JSON, CSV, etc)
                The feed information is stored in a NoSQL database
            Attributes:
                feed_list (list) - of feed information dictionaries
                **kwargs -
                    DBNAME (str) - specify a database to store the collections; overrides default
                        of using realm module and entity strings to construct the db name
                    COLLNAME (str) - specifies the collection to store all the feeds in the
                        feed_list; overrides default of using package and function to create the
                        collection name
            Returns:
                self._source (list) - dictionary of the databases and collections the feeds were
                    saved in the NoSQL database
            Exceptions:
                if module is None and kwargs['DBNAME'] is None, throw AttributeError - cannot create
                    db name
                if package is None and kwargs['COLLNAME'] is None throw AttributeError - cannot 
                    create collection name
        """

        __s_fn_id__ = f"{self.__name__} function <write_feeds_to_db>"
        
        _def_uuid_list =[]
        docs_list_ = []

        try:
            self.feedsList = feed_list
            for feed in self._feedsList:
                try:
                    self.feed = feed
                    self.realm= self._feed['realm']
                    if self._realm is None:
                        raise RuntimeError("%s setting realm failed for %s" % self._feed['realm'])
                    ''' construct DB Name from realm and entity '''
                    db_name = "_".join([self._realm['module'],self._realm['entity']]).lower()
                    if db_name is None or "".join(db_name.split())=="":
                        raise AttributeError("Empty %s db_name; cannot write feed to DB" % type(db_name))
                    logger.debug("%s database name set to %s", __s_fn_id__, db_name)
                    ''' construct collection name from package and function '''
                    coll_name = "_".join([self._realm['package'],self._realm['function']]).lower()
                    if coll_name is None or "".join(coll_name.split())=="":
                        raise AttributeError("Empty %s coll_name; cannot write feed to collection in %s" 
                                             % (type(coll_name),db_name))
                    logger.debug("%s collection name set to %s", __s_fn_id__, coll_name)
                    ''' if db doesn't exist then force to create '''
#                     if "FORCEDB" not in kwargs.keys():
#                         kwargs['FORCEDB']=True
                    kwargs['FORCEDB']=True
                    ''' write feed as a collection document to NoSQL DB '''
                    written_doc_ = clsNsql.write_documents(
                        db_name = db_name,
                        db_coll = coll_name,
                        data = feed,
                        uuid_list=[],
                        **kwargs,
                    )
                    ''' confirm collection was written and append name to list '''
                    _doc_dict = {"database":db_name, "collection":coll_name, "_id":written_doc_[0]['_id']}
                    if written_doc_ is None or len(written_doc_)<=0:
                        raise RuntimeError("write_document returned a %s written_doc_" % type(written_doc_))
                    docs_list_.append(_doc_dict)
                    logger.debug("%s data feed added to collection %s", __s_fn_id__, str(_doc_dict))

                except Exception as feed_err:
                    logger.warning("%s feed had errors", __s_fn_id__)

            ''' confirm collection list '''
            if len(docs_list_)<=0:
                raise RuntimeError("No collections were returned, got empty %s list" % type(docs_list_))
            logger.debug("%s wrote %d documents to %s",__s_fn_id__,len(docs_list_),clsNsql.dbType)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return docs_list_


    ''' Function --- READ FEED DATA ---

        author(s) : nuwan.waidyanatha@rezgateway.com
    '''
    def read_feed_data(
        self,
        feed : dict = None,
        as_type : str=DataFrame,
    ) -> any:
        """
        Description: 
        Attributes :
        Returns : 
        Exceptions:
        """
        
        __s_fn_id__ = f"{self.__name__} function <write_feeds_to_db>"

        try:
            pass

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None
