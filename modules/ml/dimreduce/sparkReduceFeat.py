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

    from dongcha.modules.ml.dimreduce import __propAttr__ as attr

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))

'''
    Class applies feature selection techniques to reduce the data

    Contributors:
        * samana.thetha@gmail.com

    Resources:
'''
class mlWorkLoads:
    """
    Unified dimensionality reduction pipeline supporting:
    - PCA (Spark and scikit-learn)
    - t-SNE (scikit-learn)
    - UMAP (umap-learn)
    - Autoencoders (TensorFlow/PyTorch)
    """

    def __init__(self, features_col: str = "features", mode: str = "spark"):
        """
        Args:
            features_col: Name of the feature vector column
            mode: 'spark' or 'pandas' (for non-Spark methods)
        """
        self.features_col = features_col
        self.mode = mode
        self.pipeline_stages = []

    # --------------------------
    # Spark-Native Methods
    # --------------------------

    class PCAReducer(Transformer):
        """Spark-native PCA implementation"""
        k = Param(Params._dummy(), "k", "Number of principal components", TypeConverters.toInt)

        @keyword_only
        def __init__(self, k: int = 2):
            super().__init__()
            self._setDefault(k=2)
            kwargs = self._input_kwargs
            self._set(**kwargs)

        def _transform(self, dataset):
            k = self.getOrDefault("k")
            pca = SparkPCA(
                k=k,
                inputCol=self.features_col,
                outputCol="pca_features"
            )
            model = pca.fit(dataset)
            explained_var = sum(model.explainedVariance)
            logging.info(f"PCA: Explained variance ratio: {explained_var:.2%}")
            return model.transform(dataset)

    # --------------------------
    # Non-Spark Methods (require pandas)
    # --------------------------

    class TSNEReducer(Transformer):
        """t-Distributed Stochastic Neighbor Embedding"""
        perplexity = Param(Params._dummy(), "perplexity", "t-SNE perplexity", TypeConverters.toFloat)
        n_components = Param(Params._dummy(), "n_components", "Output dimensions", TypeConverters.toInt)

        @keyword_only
        def __init__(self, n_components: int = 2, perplexity: float = 30.0):
            super().__init__()
            self._setDefault(n_components=2, perplexity=30.0)
            kwargs = self._input_kwargs
            self._set(**kwargs)

        def _transform(self, dataset):
            try:
                from sklearn.manifold import TSNE
                pdf = dataset.toPandas()
                features = np.array([x.toArray() for x in pdf[self.features_col]])
                
                tsne = TSNE(
                    n_components=self.getOrDefault("n_components"),
                    perplexity=self.getOrDefault("perplexity")
                )
                reduced = tsne.fit_transform(features)
                
                # Add results back to DataFrame
                for i in range(reduced.shape[1]):
                    pdf[f"tsne_{i}"] = reduced[:, i]
                
                return dataset.sql_ctx.createDataFrame(pdf)
            except ImportError:
                logging.error("scikit-learn required for t-SNE")
                return dataset

    class UMAPReducer(Transformer):
        """Uniform Manifold Approximation and Projection"""
        n_neighbors = Param(Params._dummy(), "n_neighbors", "UMAP neighbors", TypeConverters.toInt)
        min_dist = Param(Params._dummy(), "min_dist", "Minimum distance", TypeConverters.toFloat)

        @keyword_only
        def __init__(self, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1):
            super().__init__()
            self._setDefault(n_components=2, n_neighbors=15, min_dist=0.1)
            kwargs = self._input_kwargs
            self._set(**kwargs)

        def _transform(self, dataset):
            try:
                import umap
                pdf = dataset.toPandas()
                features = np.array([x.toArray() for x in pdf[self.features_col]])
                
                reducer = umap.UMAP(
                    n_components=self.getOrDefault("n_components"),
                    n_neighbors=self.getOrDefault("n_neighbors"),
                    min_dist=self.getOrDefault("min_dist")
                )
                reduced = reducer.fit_transform(features)
                
                for i in range(reduced.shape[1]):
                    pdf[f"umap_{i}"] = reduced[:, i]
                
                return dataset.sql_ctx.createDataFrame(pdf)
            except ImportError:
                logging.error("umap-learn required for UMAP")
                return dataset

    # --------------------------
    # Builder Methods
    # --------------------------

    def add_pca(self, k: int = 2):
        """Add Spark-native PCA reduction"""
        if self.mode != "spark":
            logging.warning("PCA running in Spark mode despite pandas setting")
        self.pipeline_stages.append(self.PCAReducer(k=k))
        return self

    def add_tsne(self, n_components: int = 2, perplexity: float = 30.0):
        """Add t-SNE reduction (requires pandas)"""
        if self.mode == "spark":
            logging.warning("t-SNE requires pandas conversion")
        self.pipeline_stages.append(self.TSNEReducer(n_components=n_components, perplexity=perplexity))
        return self

    def add_umap(self, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1):
        """Add UMAP reduction (requires pandas)"""
        if self.mode == "spark":
            logging.warning("UMAP requires pandas conversion")
        self.pipeline_stages.append(self.UMAPReducer(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        ))
        return self

    # --------------------------
    # Execution
    # --------------------------

    def build(self) -> Pipeline:
        """Build the final pipeline"""
        return Pipeline(stages=self.pipeline_stages)

    def run(self, dataset: DataFrame) -> DataFrame:
        """Execute the pipeline"""
        pipeline = self.build()
        return pipeline.fit(dataset).transform(dataset)