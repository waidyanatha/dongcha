#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "__propAttr__"
__package__= "dimreduce"
__module__ = "ml"
__app__ = "dongcha"
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
    import inspect
    # from dotenv import load_dotenv
    # load_dotenv()
    ''' SPARK PACKAGES '''
    import findspark
    findspark.init()
    from pyspark.sql import functions as F
    from pyspark.sql import DataFrame
    from pyspark.sql.types import *
    from pyspark.sql.window import Window
    from typing import List, Iterable, Dict, Tuple
    # from pymongo import MongoClient
    ''' DATATIME PACKAGES '''
    from datetime import datetime, date, timedelta

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except ImportError as e:
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

        self._realmList = [
            'SELECT',  # feature selection pipeline
            'EXTRACT', # feature extraction pipeline
            'REDUCE',  # feature reduction pipeline
            'IMPORTANCE', # feature importance
        ]
        self._realm= None
        self._data = None
        self._stages=None
        self._target=None
        self._features =None
        self._featNames=None
        # self._session=None


        # global pkgConf  # this package configparser class instance
        # global appConf  # configparser class instance
        # global logger   # dongcha logger class instance
#         global clsSDB   # etl loader sparkRDB class instance

        __s_fn_id__ = f"{self.__name__} function  method {inspect.currentframe().f_code.co_name}" #<__init__>"
        
        try:
            self.cwd=os.path.dirname(__file__)
            self._pkgConf = configparser.ConfigParser()
            self._pkgConf.read(os.path.join(self.cwd,__ini_fname__))

            self._projHome = self._pkgConf.get("CWDS","PROJECT")
            sys.path.insert(1,self._projHome)
            
            ''' innitialize the logger '''
            from dongcha.utils import Logger as logs
            self._logger = logs.get_logger(
                cwd=self._projHome,
                app=self.__app__, 
                module=self.__module__,
                package=self.__package__,
                ini_file=self.__ini_fname__)

            ''' set a new logger section '''
            self._logger.info('########################################################')
            self._logger.info("%s Class %s",self.__name__, self.__class__.__name__)

            ''' Set the wrangler root directory '''
            self._appDir = self._pkgConf.get("CWDS",self.__app__)
            ''' get the path to the input and output data '''
            self._appConf = configparser.ConfigParser()
            self._appConf.read(os.path.join(self._appDir, self.__conf_fname__))

            _done_str = f"{self.__name__} initialization for {self.__module__} module package "
            _done_str+= f"{self.__package__} in {self.__app__} done.\nStart workloads: {self.__desc__}."
            self._logger.debug("%s",_done_str)
            print(_done_str)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None


    ''' Function --- DATA SETTER & GETTER ---

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
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

        __s_fn_id__ = f"{self.__name__} method <@property {inspect.currentframe().f_code.co_name}>"

        try:
            if self._realm.upper() not in self._realmList:
                raise KeyError("Invalid realm; must be one of %s"
                                 % self._realmList)
        except Exception as err:
            self._logger.error("%s %s \n",__s_fn_id__, err)
            self._logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._realm.upper()

    @realm.setter
    def realm(self,realm) -> DataFrame:

        __s_fn_id__ = f"{self.__name__} method <@{inspect.currentframe().f_code.co_name}.setter>"

        try:
            if realm.upper() not in self._realmList:
                raise KeyError("Invalid %s realm; must be one of %s"
                                 % (type(realm), self._realmList))

            self._realm = realm.upper()

        except Exception as err:
            self._logger.error("%s %s \n",__s_fn_id__, err)
            self._logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._realm

    ''' --- DATA --- '''
    @property
    def data(self):

        __s_fn_id__ = f"{self.__name__} method <@property {inspect.currentframe().f_code.co_name}>"

        try:
            ''' validate property value '''
            if not isinstance(self._data,DataFrame):
                self._data = self.session.createDataFrame(self._data)
            if self._data.count() <= 0:
                raise ValueError("No records found in data") 
                
        except Exception as err:
            self._logger.error("%s %s \n",__s_fn_id__, err)
            self._logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data

    @data.setter
    def data(self,data):

        __s_fn_id__ = f"{self.__name__} method <@{inspect.currentframe().f_code.co_name}.setter>"

        try:
            ''' validate property value '''
            if data is None:
                raise AttributeError("Dataset cannot be empty")
            if not isinstance(data,DataFrame):
                self._data = self.session.createDataFrame(data)
                self._logger.debug("%s %s dtype convereted to %s with %d rows %d columns",
                         __s_fn_id__,type(data),type(self._data),
                         self._data.count(),len(self._data.columns))
            else:
                self._data = data

        except Exception as err:
            self._logger.error("%s %s \n",__s_fn_id__, err)
            self._logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data

    ''' --- STAGES --- '''
    @property
    def stages(self) -> List:
        """
        Description:
            stages @property and @setter functions manage the pipeline stages
        Attributes:
            stages in @setter will instantiate self._stages  
        Returns :
            self._stages (list) 
        """

        __s_fn_id__ = f"{self.__name__} method <@property {inspect.currentframe().f_code.co_name}>"

        try:
            if not isinstance(self._stages, list):
                self._logger.warning("Setting invalid %s stages property to empty list" 
                               % type(self._stages))
                self._stages=[]

        except Exception as err:
            self._logger.error("%s %s \n",__s_fn_id__, err)
            self._logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._stages

    @stages.setter
    def stages(self,stages:list) -> List[str]:

        __s_fn_id__ = f"{self.__name__} method <@{inspect.currentframe().f_code.co_name}.setter>"

        try:
            if not isinstance(stages, list) or len(stages)<=0:
                raise KeyError("Invalid %s stages; must be list with > 0 elements"
                                 % (type(stages)))

            self._stages = stages

        except Exception as err:
            self._logger.error("%s %s \n",__s_fn_id__, err)
            self._logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._stages


    ''' --- FEATURES --- '''
    @property
    def features(self) -> str:
        """
        Description:
            features @property and @setter functions manage the pipeline features
        Attributes:
            features in @setter will instantiate self._features
        Returns :
            self._features (str) 
        """

        __s_fn_id__ = f"{self.__name__} method <@property {inspect.currentframe().f_code.co_name}>"

        try:
            if not hasattr(self, "data"):
                raise AttributeError("Undefined data class property")
            if not isinstance(self._features, str) or self._features not in self._data.columns:
                self._logger.warning("Setting invalid %s stages property to empty str" 
                               % type(self._features))
                self._features=None

        except Exception as err:
            self._logger.error("%s %s \n",__s_fn_id__, err)
            self._logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._features

    @features.setter
    def features(self,features) -> str:

        __s_fn_id__ = f"{self.__name__} method <@{inspect.currentframe().f_code.co_name}.setter>"

        try:
            # if not hasattr(self, "data"):
                # raise AttributeError("Undefined data class property")
            if not isinstance(features, str) or "".join(features.split())=="":
                features = "features"
                logger.warning("Invalid %s features set to default: %s"
                                 % (type(features), features.upper()))
            self._features = features

        except Exception as err:
            self._logger.error("%s %s \n",__s_fn_id__, err)
            self._logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._features

    ''' --- TARGET --- '''
    @property
    def target(self) -> str:
        """
        Description:
            target @property and @setter functions are for evaluating
                feature importance
        Attributes:
            target in @setter will instantiate self._target
        Returns :
            self._target (str) 
        """

        __s_fn_id__ = f"{self.__name__} method <@property {inspect.currentframe().f_code.co_name}>"

        try:
            if not hasattr(self, "data"):
                raise AttributeError("Undefined data class property")
            if not isinstance(self._target, str) or self._target not in self._data.columns:
                raise KeyError("Invalid %s target; must be one of columns: %s"
                                 % (type(self._target), ", ".join(self._data.columns).upper()))
                # self._logger.warning("Setting invalid %s target property to an empty str" 
                #                % type(self._target))
                # self._target=""

        except Exception as err:
            self._logger.error("%s %s \n",__s_fn_id__, err)
            self._logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._target

    
    @target.setter
    def target(self,target) -> str:

        __s_fn_id__ = f"{self.__name__} method <@{inspect.currentframe().f_code.co_name}.setter>"

        try:
            if not hasattr(self, "data"):
                raise AttributeError("Undefined data class property")
            if not isinstance(target, str) or target not in self._data.columns:
                raise KeyError("Invalid %s target; must be one of columns: %s"
                                 % (type(target), ", ".join(self._data.columns).upper()))

            self._target = target

        except Exception as err:
            self._logger.error("%s %s \n",__s_fn_id__, err)
            self._logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._target


    ''' --- FEATURE NAMES --- '''
    @property
    def featNames(self) -> List[str]:
        """
        Description:
            featNames @property and @setter functions are for evaluating
                feature importance
        Attributes:
            featNames in @setter will instantiate self._featNames
        Returns :
            self._featNames (List) 
        """

        __s_fn_id__ = f"{self.__name__} method <@property {inspect.currentframe().f_code.co_name}>"

        try:
            # print("%s %s" % (__s_fn_id__, self._featNames))
            if not isinstance(self._featNames, list) or len(self._featNames)<=0:
                self._logger.warning("%s empty %s, retrieving feature names", 
                                     __s_fn_id__, type(self._featNames))
                self._featNames=self._get_feature_names()
                if not isinstance(self._featNames, list) or len(self._featNames)<=0:
                    raise ChildProcessError("failed to construct feature names, returned %s"
                                            % type(self._featNames))
            # print("%s %s" % (__s_fn_id__, self._featNames))
            # if not hasattr(self, "data"):
            #     raise AttributeError("Undefined data class property")
            # _common_cols = set(self.__featNames).intersection(set(self._data.columns))
            # if _common_cols != len(self._featNames):
            #     raise ValueError("One of more feature names did not match, must be: %s" 
            #                    % ", ".join(self._data.columns))

        except Exception as err:
            self._logger.error("%s %s \n",__s_fn_id__, err)
            self._logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._featNames

    
    @featNames.setter
    def featNames(self,featNames) -> List[str]:

        __s_fn_id__ = f"{self.__name__} method <@{inspect.currentframe().f_code.co_name}.setter>"

        try:
            if not isinstance(featNames, list) or len(featNames)<=0:
                raise AttributeError("feature names must be a list of strings and not %s" 
                                     % type(featNames))
                
            # if not hasattr(self, "data"):
            #     raise AttributeError("Undefined data class property")
            # _common_cols = set(featNames).intersection(set(self._data.columns))
            # if _common_cols != len(featNames):
            #     raise ValueError("One of more feature names did not match, must be: %s" 
            #                    % ", ".join(self._data.columns))

            self._featNames = featNames

        except Exception as err:
            self._logger.error("%s %s \n",__s_fn_id__, err)
            self._logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._featNames

    # class BaseImportanceMethod(Transformer):
    #     """Base class for feature importance methods"""
    #     features_col = Param(Params._dummy(), "features_col", 
    #                          "Name of the features column", TypeConverters.toString)
    #     feature_names= Param(Params._dummy(), "feature_names", 
    #                          "feature column names", TypeConverters.toListString)

    #     @keyword_only
    #     def __init__(self, features_col="features", feature_names:List[str]=[], logger=None):
    #         super().__init__()
    #         self.logger = logger  # Initialize logger directly, not as a Param
    #         self._setDefault(features_col="features", feature_names:List[str]=[], )
    #         kwargs = self._input_kwargs
    #         if 'logger' in kwargs:
    #             del kwargs['logger']
    #         self._set(**kwargs)

    #     @keyword_only
    #     def setParams(self, features_col="features", feature_names:List[str]=[], logger=None):
    #         kwargs = self._input_kwargs
    #         # Remove logger from kwargs before _set
    #         if 'logger' in kwargs:
    #             self.logger = kwargs.pop('logger')
    #         self._set(**kwargs)
    #         return self

    #     def _transform(self, dataset: DataFrame) -> DataFrame:
    #         raise NotImplementedError
            
    ''' Function --- DATA SETTER & GETTER ---

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    def _get_feature_names(self) -> List[str]:
        """Extract feature names from VectorAssembler output, DataFrame, or construct"""

        __s_fn_id__ = f"{self.__name__} method <{inspect.currentframe().f_code.co_name}>"

        try:
            # features_col = self.getOrDefault("features_col")
            # feature_names= self.getOrDefault("feature_names")

            if hasattr(self,"features") and hasattr(self,"featNames"):
                sample = self._data.select(self._features).first()[0]
                if hasattr(sample, "indices"):  # Sparse vector
                    size = sample.size
                else:  # Dense vector
                    size = len(sample)
                if len(self._featNames)!=size:
                    logger.warning("%s prep default columns, features %s size: %d <> feature names: %d", 
                                   __s_fn_id__, self._features, size, len(self._featNames))
                    feature_names=[f"feature_{i}" for i in range(size)]

            elif hasattr(self,"features") and not hasattr(self,"featNames"):
                sample = dataset.select(self._features).first()[0]
                if hasattr(sample, "indices"):  # Sparse vector
                    size = sample.size
                else:  # Dense vector
                    size = len(sample)
                self._featNames=[f"feature_{i}" for i in range(size)]
            elif not hasattr(self,"features") and not hasattr(self,"featNames"):
                self._featNames=dataset.columns
                logger.warning("%s set %d feature names to dataset columns: %s", 
                                   __s_fn_id__, len(self._featNames), ", ".join(dataset.columns))
                
            # if hasattr(self, "feature_names"):
            #     return self._feature_names
                
            # # Get the first row of features
            # sample = dataset.select(self.features).first()[0]
            # if hasattr(sample, "indices"):  # Sparse vector
            #     size = sample.size
            # else:  # Dense vector
            #     size = len(sample)
                
            if not isinstance(self._featNames, list) or len(self._featNames)<=0:
                raise RuntimeError('Failed to construct feature_names returned %s' 
                                   % type(self._featNames))

        except Exception as err:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error("%s %s \n", __s_fn_id__, err)
                self.logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            raise

        finally:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.info("%s result %d feature names: %s", 
                                 __s_fn_id__, len(self._featNames), ", ".join(self._featNames))
            return self._featNames
            # return [f"feature_{i}" for i in range(size)]