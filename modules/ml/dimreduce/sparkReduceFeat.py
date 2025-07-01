#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "sparkReduceFeat"
__package__ = "dimreduce"
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
    import functools
    import traceback
 
    import numpy as np
    import findspark
    findspark.init()
    from pyspark import keyword_only
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        NumericType, IntegerType, DoubleType, FloatType, StringType)
    from pyspark.sql import DataFrame
    # from pyspark.ml.feature import Imputer, VarianceThresholdSelector, VectorAssembler, StandardScaler
    # from pyspark.ml.feature import UnivariateFeatureSelector, OneHotEncoder, StringIndexer
    from pyspark.ml import Pipeline, Transformer
    from pyspark.ml.feature import (
        StringIndexer, OneHotEncoder, VectorAssembler, 
        Imputer, VarianceThresholdSelector, StandardScaler,
        PCA,
    )
    # from pyspark.ml.feature import PCA as SparkPCA, VectorAssembler
    from pyspark.ml.param.shared import Param, Params, TypeConverters
    from pyspark.ml.stat import Correlation
    # from pyspark.ml import Pipeline, Transformer
    from pyspark import keyword_only
    from typing import List, Iterable, Dict, Tuple, Optional, Union
    from sklearn.manifold import TSNE
    import umap

    from dongcha.modules.ml.dimreduce import __propAttr__ as attr

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))


except ImportError as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))

'''
    Class applies feature selection techniques to reduce the data

    Contributors:
        * samana.thetha@gmail.com

    Resources:
'''
class mlWorkLoads(attr.properties):
    """
    Unified dimensionality reduction pipeline supporting:
    - PCA (Spark and scikit-learn)
    - t-SNE (scikit-learn)
    - UMAP (umap-learn)
    - Autoencoders (TensorFlow/PyTorch)
    """

    def __init__(
        self, 
        data=None, 
        realm='SELECT',
        features_col: str = "features", 
        mode: str = "spark"):
        """
        Args:
            features_col: Name of the feature vector column
            mode: 'spark' or 'pandas' (for non-Spark methods)
        """
        # self.features_col = features_col
        # self.stages = []

        super().__init__(
#             desc=self.__desc__,
            realm=realm,

        )

        self.__name__ = __name__
        self.__package__ = __package__
        self.__module__ = __module__
        self.__app__ = __app__
        self.__ini_fname__ = __ini_fname__
        self.__conf_fname__ = __conf_fname__

        ''' required setting data before features '''
        self.data = data
        self.features=features_col
        self.mode = mode
    
        __s_fn_id__ = f"{self.__name__} method {self.__class__.__name__}"

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
            self._logger.error("%s %s \n",__s_fn_id__, err)
            self._logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None

    # --------------------------
    # Spark-Native Methods
    # --------------------------

    class PCAReducer(Transformer):
        """Spark-native PCA implementation"""
        k = Param(Params._dummy(), "k", "Number of principal components", TypeConverters.toInt)
        features_col = Param(Params._dummy(), "features_col", 
                             "Name of the features column", TypeConverters.toString)

        @keyword_only
        def __init__(self, k: int = 2, features_col="features", logger=None):
            super().__init__()
            self.logger = logger  # Initialize logger directly, not as a Param
            self._setDefault(k=2, features_col="features", )
            kwargs = self._input_kwargs
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)

        @keyword_only
        def setParams(self, k = 2, features_col="features", logger=None):
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                self.logger = kwargs.pop('logger')
            self._set(**kwargs)
            return self

        def _transform(self, dataset):

            __s_fn_id__ = f"class {self.__class__.__name__}"

            try:
                k = self.getOrDefault("k") # Get the k value
                features_col = self.getOrDefault("features_col")  # Get the features column name
                pca = PCA(
                    k=k,
                    inputCol=features_col,
                    outputCol="pca_features"
                )
                model = pca.fit(dataset)
                explained_var = sum(model.explainedVariance)
                if not explained_var:
                    raise RuntimeError('Failed to construct explained variance returned %s' 
                                       % type(explained_var))

            except Exception as err:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)
                raise
    
            finally:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.info(f"PCA: Explained variance ratio: {explained_var:.2%}")

                return model.transform(dataset)

    # --------------------------
    # Non-Spark Methods (require pandas)
    # --------------------------

    class TSNEReducer(Transformer):
        """t-Distributed Stochastic Neighbor Embedding"""
        perplexity = Param(Params._dummy(), "perplexity", "t-SNE perplexity", TypeConverters.toFloat)
        n_components = Param(Params._dummy(), "n_components", "Output dimensions", TypeConverters.toInt)
        features_col = Param(Params._dummy(), "features_col", 
                             "Name of the features column", TypeConverters.toString)

        @keyword_only
        def __init__(self, n_components: int = 2, perplexity: float = 30.0, 
                     features_col="features", logger=None):
            super().__init__()

            self.logger = logger  # Initialize logger directly, not as a Param
            self._setDefault(n_components=2, perplexity=30.0, features_col="features")
            kwargs = self._input_kwargs
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)

        @keyword_only
        def setParams(self, n_components=2, perplexity=30.0, 
                      features_col="features", logger=None):
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                self.logger = kwargs.pop('logger')
            self._set(**kwargs)
            return self

        def _transform(self, dataset):

            __s_fn_id__ = f"class {self.__class__.__name__}"

            try:
                pdf = dataset.toPandas()
                features_col = self.getOrDefault("features_col")  # Get the features column name
                features = np.array([x.toArray() for x in pdf[features_col]])
                n_components=self.getOrDefault("n_components")
                perplexity=self.getOrDefault("perplexity")
                
                tsne = TSNE(
                    n_components=n_components, #self.getOrDefault("n_components"),
                    perplexity=perplexity, #self.getOrDefault("perplexity")
                )
                reduced = tsne.fit_transform(features)
                
                # Add results back to DataFrame
                for i in range(reduced.shape[1]):
                    pdf[f"tsne_{i}"] = reduced[:, i]
                
                if pdf.shape[0]<=0:
                    raise RuntimeError('Failed to construct tsne dataframe %s' % type(pdf))

                from dongcha.modules.lib.spark import execSession
                _spark = execSession.Spawn()
                # _spark.session.createDataFrame()

            except Exception as err:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)
                raise
    
            finally:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.info("%s returned %s %d columns and %d rows", 
                                     __s_fn_id__, type(pdf), pdf.shape[1], pdf.shape[0])
                    _tsne_cols = [col for col in pdf.columns \
                                      if 'tsne' in col.split("_")]
                    self.logger.info("%s with %d components and %0.2f perplexity columns: %s", 
                                     __s_fn_id__, n_components, perplexity, 
                                     ", ".join(_tsne_cols))

                return _spark.session.createDataFrame(pdf)
            # return dataset.sql_ctx.createDataFrame(pdf)
            # except ImportError:
            #     logging.error("scikit-learn required for t-SNE")
            #     return dataset

    class UMAPReducer(Transformer):
        """Uniform Manifold Approximation and Projection"""
        n_components = Param(Params._dummy(), "n_components", "Output dimensions", TypeConverters.toInt)
        n_neighbors = Param(Params._dummy(), "n_neighbors", "UMAP neighbors", TypeConverters.toInt)
        min_dist = Param(Params._dummy(), "min_dist", "Minimum distance", TypeConverters.toFloat)
        features_col = Param(Params._dummy(), "features_col", 
                             "Name of the features column", TypeConverters.toString)

        @keyword_only
        def __init__(self, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1,
                     features_col="features", logger=None):
            super().__init__()

            self.logger = logger  # Initialize logger directly, not as a Param
            self._setDefault(n_components=2, n_neighbors=15, min_dist=0.1, features_col="features")
            kwargs = self._input_kwargs
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)

        @keyword_only
        def setParams(self, n_components=2, n_neighbors=15, min_dist=0.1,
                      features_col="features", logger=None):
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                self.logger = kwargs.pop('logger')
            self._set(**kwargs)
            return self

        def _transform(self, dataset):
            
            __s_fn_id__ = f"class {self.__class__.__name__}"

            try:
                # import umap
                pdf = dataset.toPandas()
                # features = np.array([x.toArray() for x in pdf[self._features]])
                features_col = self.getOrDefault("features_col")  # Get the features column name
                features = np.array([x.toArray() for x in pdf[features_col]])
                _n_components=self.getOrDefault("n_components")
                _n_neighbors=self.getOrDefault("n_neighbors")
                _min_dist=self.getOrDefault("min_dist")
                
                reducer = umap.UMAP(
                    n_components=_n_components, #self.getOrDefault("n_components"),
                    n_neighbors=_n_neighbors, #self.getOrDefault("n_neighbors"),
                    min_dist=_min_dist, #self.getOrDefault("min_dist")
                )
                reduced = reducer.fit_transform(features)
                
                for i in range(reduced.shape[1]):
                    pdf[f"umap_{i}"] = reduced[:, i]
                
                if pdf.shape[0]<=0:
                    raise RuntimeError('Failed to construct UMAP dataframe %s' % type(pdf))

                from dongcha.modules.lib.spark import execSession
                _spark = execSession.Spawn()
                # return dataset.sql_ctx.createDataFrame(pdf)
            # except ImportError:
            #     logging.error("umap-learn required for UMAP")
            #     return dataset

            except Exception as err:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)
                raise
    
            finally:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.info("%s returned %s %d columns and %d rows", 
                                     __s_fn_id__, type(pdf), pdf.shape[1], pdf.shape[0])
                    self.logger.info("%s with %d components, %d neghbours, and %0.2f minimum distance", 
                                     __s_fn_id__, _n_components, _n_neighbors, _min_dist)
                    _tsne_cols = [col for col in pdf.columns \
                                      if 'umap' in col.split("_")]
                    self.logger.info("%s columns: %s", 
                                     __s_fn_id__, ", ".join(_tsne_cols))

                return _spark.session.createDataFrame(pdf)


    # --------------------------
    # Builder Methods
    # --------------------------

    def add_pca(self, k: int = 2):
        """Add Spark-native PCA reduction"""
        if self.mode != "spark":
            self._logger.warning("PCA running in Spark mode despite pandas setting")
        self.stages.append(self.PCAReducer(
            k=k, 
            features_col=self._features,
            logger=self._logger))
        return self

    def add_tsne(self, n_components: int = 2, perplexity: float = 30.0):
        """Add t-SNE reduction (requires pandas)"""
        if self.mode == "spark":
            self._logger.warning("t-SNE requires pandas conversion")
        self.stages.append(self.TSNEReducer(
            n_components=n_components, 
            perplexity=perplexity,
            features_col=self._features,
            logger=self._logger,
            ))
        return self

    def add_umap(self, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1):
        """Add UMAP reduction (requires pandas)"""
        if self.mode == "spark":
            self._logger.warning("UMAP requires pandas conversion")
        self.stages.append(self.UMAPReducer(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            features_col=self._features,
            logger=self._logger,
            ))
        return self

    # --------------------------
    # Execution
    # --------------------------

    def exec_pipe_with_stages(self):
        """
        Executes pipeline step-by-step, ensuring each stage receives the 
        correctly transformed DataFrame from the previous stage.
        
        Returns:
            - intermediates (dict): {"stage_name": intermediate_df}
            - final_df (DataFrame): Final transformed output
        """
        current_df = self.data
        self.intermediate_results = {}
        
        for i, stage in enumerate(self.stages):
            stage_name = f"stage_{i}_{stage.__class__.__name__}"
            print(f"Executing {stage_name}...")
            
            # Apply the current stage
            current_df = stage.transform(current_df)
            
            # Store intermediate result
            self.intermediate_results[stage_name] = current_df
            
            # Debug: Print column count to verify correctness
            print(f"After {stage_name}, columns: {len(current_df.columns)}")
            
            # Force execution (avoids lazy evaluation issues)
            current_df.cache().count()
        
        return self.intermediate_results, current_df

    def build(self) -> Pipeline:
        """Build the final pipeline"""
        return Pipeline(stages=self.pipeline_stages)

    def run(self, dataset: DataFrame) -> DataFrame:
        """Execute the pipeline"""
        pipeline = self.build()
        return pipeline.fit(dataset).transform(dataset)