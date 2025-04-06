#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ ="FeedWorkLoads"
__module__ = "etl"
__package__= "extractor"
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
    import re

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))


class FeedWorkLoads():
    """ ***************************************************************************************
    CLASS spefic to retrieving data feeds and ingesting the parameter values into the feeds to
        construct the specific feeds for retrieving the data. It will use the 
    
    We implement mongodb and pyspark to perform the the data source feeds management package.
    class is instantiated, it will inherit properties and methods from packages 
    (1) rezaware/etl/extractor/dataFeedsDB - to read feeds information from NoSQL DB collections & documents
    
    contributors:
        * nuwan.waidyanatha@rezgateway.com
        
    ******************************************************************************************* 
    """
    def __init__(self,
                 desc:str="build data feeds with parameter key value pairs",
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
#         self._realm = {
#             "module" : None,
#             "entity" : None,
#             "package" :None,
#             "function":None
#         } # is associated with the module or domain taxanomy of the data source
#         ''' defines uri mandatory elements '''
#         self._uri = {
#             "urn" : None,
#             "protocol": None,
#             "domain" : None,
#             "port" : None,
#             "path" : None,
#             "query": None,
#             "fragments" :None,
#         }
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
        self._feedsList = None    # holds a list of feeds dictionaries defined by self._feed
        self._queryConstruct=None # the list of query constructed ingesting parameters values
        self._execFeeds = None    # dictionary list of executable feeds

        ''' initiate to load app.cfg data '''
        global logger
        global pkgConf
        global appConf
        global clsFDB
        
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

            from rezaware.modules.etl.extractor import dataFeedsDB as fdb
            clsFDB=fdb.FeedWorkLoads(desc=self.__desc__)
            print("%s Class initialization complete" % self.__name__)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None


#     ''' Function --- FEED ---
#         author: <nuwan.waidyanatha@rezgateway.com
#     '''
#     @property
#     def feed(self) -> dict:

#         __s_fn_id__ = f"{self.__name__} function <@property feed>"

# #         feed_prim_key_list = ['source', 'realm', 'uri', 'get']

#         try:
#             if not isinstance(self._feed,dict) or len(self._feed)<=0:
#                 raise AttributeError("Invalid feed empty property %s type" % type(self._feed))
#             if len(list(set(self._feed.keys()).intersection(set(self._feed.keys())))) \
#                     < len(self._feed.keys()):
#                 raise AttributeError("Missing feed primary key information, must contain %s" 
#                                      % (self._feed.keys()))
#         except Exception as err:
#             logger.error("%s %s \n",__s_fn_id__, err)
#             logger.debug(traceback.format_exc())
#             print("[Error]"+__s_fn_id__, err)

#         return self._feed

#     @feed.setter
#     def feed(self,feed:dict) -> dict:

#         __s_fn_id__ = f"{self.__name__} function <@feed.setter>"
        
# #         _prim_elem_list = ['source', 'realm', 'context','uri', 'get']

#         try:
#             if not isinstance(feed,dict) or len(feed)<=0:
#                 raise AttributeError("Invalid feed empty property %s type" % type(feed))
#             if len(list(set(feed.keys()).intersection(set(self._feed.keys())))) \
#                     < len(self._feed.keys()):
#                 raise AttributeError("Missing feed primary keys %s, madatory keys %s" 
#                                      % (feed.keys(),_prim_elem_list))
#             self._feed = feed
#             logger.debug("%s feed property set for realm %s",__s_fn_id__,self._feed['realm'])

#         except Exception as err:
#             logger.error("%s %s \n",__s_fn_id__, err)
#             logger.debug(traceback.format_exc())
#             print("[Error]"+__s_fn_id__, err)

#         return self._feed


#     ''' --- FEEDS LIST --- '''
#     @property
#     def feedsList(self) -> list:
#         """
#         Description:
#             list of data feeds; where a feed is defined by the class property self.feed
#         Attributes:
#             None
#         Returns:
#             self._feedsLis(list)feedsList 
#         Exceptions:
#           feedsListif feedsList is empty, throw AttributeError exception
#         """

#         __s_fn_id__ = f"{self.__name__} function <@property feedsList>"
        
#         try:
#             if not isinstance(self._feedsList,list) or len(self._feedsList)<=0:
#                 raise AttributeError("Invalid empty feedsList property %s type" % type(self._feedsList))

#         except Exception as err:
#             logger.error("%s %s \n",__s_fn_id__, err)
#             logger.debug(traceback.format_exc())
#             print("[Error]"+__s_fn_id__, err)

#         return self._feedsList

#     @feedsList.setter
#     def feedsList(self,feeds_list:list) -> list:

#         __s_fn_id__ = f"{self.__name__} function <@feedsList.setter>"

#         try:
#             if not isinstance(feeds_list,list) or len(feeds_list)<=0:
#                 raise AttributeError("Invalid empty feeds_list setter of %s type" % type(feed_list))

#             self._feedsList = feeds_list
#             logger.debug("%s property feedsList set with %d feeds",__s_fn_id__,len(self._feedsList))

#         except Exception as err:
#             logger.error("%s %s \n",__s_fn_id__, err)
#             logger.debug(traceback.format_exc())
#             print("[Error]"+__s_fn_id__, err)

#         return self._feedsList


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


    ''' Function --- QUERY CONSTRUCT ---
        author: <nuwan.waidyanatha@rezgateway.com
    '''
    @property
    def queryConstruct(self) -> pd.DataFrame:

        __s_fn_id__ = f"{self.__name__} function <@property queryConstruct>"

        try:
            if not isinstance(self._queryConstruct,pd.DataFrame):
                raise AttributeError("Invalid queryConstruct dtype %s; must be pd.DataFrame" 
                                     % type(self._queryConstruct))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._queryConstruct


    @queryConstruct.setter
    def queryConstruct(self,query_construct:pd.DataFrame) -> pd.DataFrame:
        """
        Description:
            Sets the queryConstruct propert as a pandas dataframe.
        Attributes :
            query_construct (pd.DataFrame) 
        Returns :
            self._queryConstruct (pd.Dataframe) with columns: 'feed_id','protocol','domain','port',
            'path','query','fragments'
        Exceptions :
            * If input var query_construct is not a pd.DataFrame or empty, then raise AttributeError
        """

        __s_fn_id__ = f"{self.__name__} function <@queryParam.setter>"

        try:
            if not isinstance(query_construct,pd.DataFrame) or query_construct.shape[0]<=0:
                raise AttributeError("Trying to set an invalid query_construct empty property %s type" 
                                     % type(query_construct))
            self._queryConstruct = query_construct
            logger.debug("%s queryConstruct property set with %d rows",
                         __s_fn_id__,self._queryConstruct.shape[0])

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._queryConstruct


    ''' Function --- EXECUTABLE FEEDS LIST ---
        author(s) : nuwan.waidyanatha@rezgateway.com
    '''
    @property
    def execFeeds(self) -> dict:
        """
        Description:
            Property comprising executable feeds built for each URI from the self._feedsList
        Attributes :
        Returns :
            self._execFeeds (dict) with the source info and constructed executable feed
        Exceptions :
            * If execFeeds is not a dict, then raise AttributeError
        """

        __s_fn_id__ = f"{self.__name__} function <@property execFeeds>"

        try:
            if not isinstance(self._execFeeds,dict):
                raise AttributeError("Invalid execFeeds propert dtype %s; must be a dict" 
                                     % type(self._execFeeds))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._execFeeds

    @execFeeds.setter
    def execFeeds(self,executable_feeds:pd.DataFrame) -> pd.DataFrame:
        """
        Description:
            Sets the execFeeds propert as a dictionary.
        Attributes :
            executable_feeds (dict) 
        Returns :
            self._execFeeds
        Exceptions :
            * If input var executable_feeds is not a dict or empty, then raise AttributeError
        """

        __s_fn_id__ = f"{self.__name__} function <@execFeeds.setter>"

        try:
            if not isinstance(executable_feeds,dict) or len(executable_feeds)<=0:
                raise AttributeError("Trying to set an invalid executable_feeds empty property %s type" 
                                     % type(executable_feeds))
            self._execFeeds = executable_feeds
            logger.debug("%s _execFeeds property set with %d records",
                         __s_fn_id__,len(self._execFeeds))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._execFeeds

    ''' Function --- CONSTRUCT EXECUTABLE FEEDS ---
        author(s) : nuwan.waidyanatha@rezgateway.com
    '''
    def execFeeds(func):

        @functools.wraps(func)
        def wrapper_exec_feeds(
        self,
        params_list,
        feeds_list,
        ) -> list:
            """
            Description:
                Wrapper function to the build_query_list for each row in dataframe.
                * Takes the constructed query list
                * Augments the domain, port, path, and fragments
                * adds extra info about source and get method
                * appends to the self._execFeeds list
            Attributes:
                func - wraps build_query_list
            Exceptions:
            Returns:
                self._execFeeds (dict) with source and executable feed
            """
            __s_fn_id__ = f"{self.__name__} function <wrapper_exec_feeds>"

            keep_cols = ['feed_id','owner','method', 'object']
            feeds_list_ = []
            
            try:
                built_feeds = func(self, params_list, feeds_list,)
                built_feeds = built_feeds.reset_index()
                
                for idx, row in built_feeds.iterrows():
                    exec_feed = None
                    ''' --- 1. protocol --- '''
                    if row['protocol'] in ['http','https','http://','https://']:
                        exec_feed = "://".join([row['protocol'].replace("://","").lower(),""])
                    ''' --- 2. domain --- '''
                    if "".join(row['domain'].split())!="":
                        exec_feed = "".join([exec_feed,row['domain']])
                    if "".join(row['port'].split())!="" and (1 <= int(row['port']) <= 65535):
                        exec_feed = ":".join([exec_feed,row['port']])
                    ''' --- 3. path --- '''
                    if "".join(row['path'].split())!="":
                        if row['path'][0] == "/":
                            exec_feed = "/".join([exec_feed,row['path'][1:]])
                        else:
                            exec_feed = "/".join([exec_feed,row['path']])
                    ''' --- 4. query --- '''
                    if "".join(row['query'].split())!="":
                        if row['query'][0] == "?":
                            exec_feed = "?".join([exec_feed,row['query'][1:]])
                        else:
                            exec_feed = "?".join([exec_feed,row['query']])
                    ''' --- 4. fragments --- '''
                    if "".join(row['fragments'].split())!="":
                        if row['fragments'][0] == "#":
                            exec_feed = "#".join([exec_feed,row['fragments'][1:]])
                        else:
                            exec_feed = "#".join([exec_feed,row['fragments']])
                    ''' validate feed construction '''
                    if exec_feed is None or "".join(exec_feed.split())=="":
                        raise ValueError("row %s with feed id: %s returned an empty %s exec_feed" 
                                         % (idx,row['feed_id'],type(exec_feed)))
                    ''' build the dict with constructed feed and append to list'''
                    feed_dict_ = row.to_dict()
                    feed_dict_ = dict([(key, val) for key, val in
                                       feed_dict_.items() if key in keep_cols])
                    feed_dict_['exec_feed'] = exec_feed
                    feeds_list_.append(feed_dict_)

                if len(feeds_list_)<=0:
                    raise RuntimeError("Failed process resulted in empty %s executable feeds list" 
                                       % type(feeds_list_))
                self._execFeeds = feeds_list_
                logger.debug("%s Successfully constructed %d executable feeds from %d query constructs",
                             __s_fn_id__, len(self._execFeeds),built_feeds.shape[0])
            
            except Exception as err:
                logger.error("%s %s \n",__s_fn_id__, err)
                logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)

            return self._execFeeds

        return wrapper_exec_feeds


    ''' Function --- CONSTRUCT URI DICT ---
        author(s) : nuwan.waidyanatha@rezgateway.com
    '''
#     @execFeeds
    def build_query_list(
        self,
        params_list:list = None,
        feeds_list : list= None,
    ) -> list:
        """
        Description:
            For each dictionary of parameters and values, the function constructs an explicit data feed.
            * Validate the feed list and prameter list
            * Loop through each feed to acquire the _id (ObjId)
            * filter the parameter dictionaries by the feed uuid
            * Loop through the filtered set of parameters to:
               * validate the parameter values against the data type defined in the feed parameter
               * replace the place holder attributes in the query to generate the explicity query
               * construct the URI with the given protocol, domin, port, path, query and fragment
        Attributes :
            params_list (list) dictionary of key/value pairs to injest into the parameter placeholders
                A dictionary must contain the _id (ObjID) and all paramter attributes and values
            feeds_list (list) is a list of dictionaries comprising the source, realm, context, uri, and 
                get dictionaries
        Returns :
            self._explicit_queries (list) of protocol specific executable queries 
        Exceptions:
            * if params_list is empty or none raise AttributeError
            * if feeds_list is empty or none raise AttributeError
            * if param attribute/value pairs 
               * don't match the required set of parameters for the query raise an AttributeError 
               * any feed parameter data type doesn't agree with the corresponding parameter value raise
                   any AttributeError
        """
        
        __s_fn_id__ = f"{self.__name__} function <construct_query_list>"

        try:
            if not isinstance(params_list,list) or len(params_list)<=0:
                raise AttributeError("Invalid empty params_list of type %s" % type(params_list))
            self.feedsList = feeds_list
            ''' init dataframe to store processed queries '''
            cols_list = ['feed_id','owner','method', 'object',
                         'protocol','domain','port','path','query','fragments']
            filled_query_df_ = pd.DataFrame(columns = cols_list)

            ''' process feeds '''
            for feed in self._feedsList:
                try:
                    ''' validate and set the feed property '''
                    self.feed = feed
                    ''' select parameter dictionaries for respective feed _id '''
                    filt_param_by_id=None
                    filt_param_by_id = list(
                        filter(
                            lambda param_list: str(param_list['_id']) == str(self._feed['_id']), 
                            params_list))
                    if filt_param_by_id is None or len(filt_param_by_id)<=0:
                        err_txt = "filtering paramters for feed id {0} failed "\
                                                            .format(self._feed['_id'])
                        err_txt += "{0} list. \n".format(type(filt_param_by_id))
                        err_txt += "Did you mean one of the following ids: {0}"\
                                .format(str(set([x['_id'] for x in params_list])))
                        raise RuntimeError(err_txt)
                    logger.debug("%s processing fitered %d parameter sets for _id: %s", 
                                 __s_fn_id__, len(filt_param_by_id), self._feed['_id'])
                    ''' process each feed query expression '''
                    for each_uri in self._feed['uri']:
                        _query_template = each_uri['query']['expression']
                        ''' insert each set of input parameter values in query expression'''
                        for in_param_dict in filt_param_by_id:
                            try:
                                built_query_ = _query_template
                                ''' validate parameter dtype against defined in feed '''
                                if len(in_param_dict['parameter'].keys()) < \
                                    len(each_uri['query']['parameter'].keys()):
                                    err_str = "Mismatch between defined feed paramters {0} "\
                                                .format(len(each_uri['query']['parameter'].keys()))
                                    err_str += "and supplied input parameters {0}"\
                                                .format(len(in_param_dict['parameter'].keys()))
                                    raise AttributeError(err_str)
                                ''' insert input parameter value in respective slot in query '''
                                for param_key, param_val in in_param_dict['parameter'].items():
                                    _s_regex = r"{"+param_key+"}"
                                    urlRegex = re.compile(_s_regex, re.IGNORECASE)
                                    param = urlRegex.search(built_query_)
                                    if param:
                                        _s_repl_val = str(param_val).replace(" ","%20")
                                        built_query_ = re.sub(_s_regex, _s_repl_val, built_query_)
                                    
                                    feed_param = None
                                    feed_param = next(({x:y} 
                                                       for x,y in each_uri['query']['parameter'].items() 
                                                           if x == param_key),None)

                            except Exception as uri_err:
                                logger.warning("%s - URI iteration error - %s \n",__s_fn_id__, uri_err)
                                built_query_ = None
                                logger.debug(traceback.format_exc())

                            if built_query_ is not None:
                                ''' create data frame with uri objects '''
                                filled_query_df_ = filled_query_df_.append(
                                    {"feed_id":self._feed['_id'],
                                     "owner" : self._feed['source']['owner'],
                                     "method": self._feed['get']['method'],
                                     "object": self._feed['get']['object'],
                                     "protocol":each_uri['protocol'],
                                     "domain" : each_uri['domain'],
                                     "port" : each_uri['port'],
                                     "path" : each_uri['path'],
                                     "query":built_query_,
                                     "fragments" : "#".join([x.replace("#","")
                                                             for x in each_uri['fragments']]),
                                    },
                                    ignore_index=True)
                                logger.debug("%s built query added to dataframe %s",
                                             __s_fn_id__,built_query_)

                except Exception as feed_err:
                    logger.warning("%s - FEED iteration error - %s \n",__s_fn_id__, feed_err)
                    logger.debug(traceback.format_exc())

            if filled_query_df_.shape[0]<=0:
                raise RuntimeError("failed process with resulted in empty result %s " 
                                   % type(filled_query_df_))
            self._queryConstruct = filled_query_df_
            logger.debug("%s succesfully process %d queries",
                         __s_fn_id__, self._queryConstruct.shape[0])

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._queryConstruct
