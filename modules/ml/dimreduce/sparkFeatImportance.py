#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "sparkFeatImportance"
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
    # import functools
    import traceback
    import inspect
 
    import findspark
    findspark.init()
    from pyspark.sql import functions as F
    from pyspark.sql import DataFrame
    from pyspark.ml.functions import vector_to_array
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml import Pipeline, Transformer
    # from pyspark.ml.param.shared import Param, Params
    from pyspark.ml.param.shared import Param, Params, TypeConverters
    from pyspark.sql.types import (NumericType, DoubleType, 
                                    FloatType, IntegerType)
    from pyspark import keyword_only
    from typing import List

    import pandas as pd
    import numpy as np
    from sklearn.base import BaseEstimator
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    import shap
    from xgboost import XGBClassifier, XGBRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    import matplotlib.pyplot as plt

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
# class dataWorkLoads(attr.properties):
    """Main class for feature engineering pipeline"""
    
    def __init__(
        self, 
        data: DataFrame = None, 
        realm: str = 'IMPORTANCE', 
        target_col: str = "target", 
        features_col:str= "features", # optional, will use all or mentioned feature names
        feature_names:List[str]=[],   # optional, use to filter dataframe else asign defaults
    ):
        """
        Initialize the feature importance analyzer.
        
        Args:
            data: Spark DataFrame containing features and target
            realm: Domain/realm for the analysis
            target_col: Name of the target column
            features_col: Name of the assembled features column
            feature_names: List of feature names to use
            stages: List of pipeline stages to initialize with
        """
        super().__init__(
#             desc=self.__desc__,
            realm=realm,

        )
        self.data = data
        self.realm= realm
        self.target=target_col
        # if isinstance(features_col,str) and "".join(features_col.split())!="":
        #     self.features = features_col
        self.features = features_col if isinstance(features_col,str) \
                                        and "".join(features_col.split())!="" \
                                    else "features" 
        self.featNames = feature_names if isinstance(feature_names,list) \
                                        and len(feature_names)>0 else []
        # if isinstance(feature_names,list) and len(feature_names)>0:
        #     self.featNames= feature_names
        self._pandas_df = None
        self._shap_values = None
        self._importance_results = {}
        
    
        __s_fn_id__ = f"{self.__name__} method {inspect.currentframe().f_code.co_name}" # <__init__>"

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

    class BaseImportanceMethod(Transformer):
        """Base class for feature importance methods"""
        features_col = Param(Params._dummy(), "features_col", 
                           "Name of the features column", TypeConverters.toString)
        feature_names = Param(Params._dummy(), "feature_names", 
                            "feature column names", TypeConverters.toListString)
        target_col = Param(Params._dummy(), "target_col",
                         "target column name", TypeConverters.toString)
        
        @keyword_only
        def __init__(self, features_col: str = "features", feature_names: List[str] = [],
                     target_col: str = "target", logger=None):
            super().__init__()
            self.logger = logger
            self._setDefault(features_col="features", feature_names=[], target_col="target")
            kwargs = self._input_kwargs
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)

        @keyword_only
        def setParams(self, features_col="features", feature_names=[], 
                     target_col="target", logger=None):
            kwargs = self._input_kwargs
            if 'logger' in kwargs:
                self.logger = kwargs.pop('logger')
            self._set(**kwargs)
            return self

        def _transform(self, dataset):
            raise NotImplementedError
    
    class SHAPAnalyzer(BaseImportanceMethod):
        """Compute SHAP (SHapley Additive exPlanations) values for feature importance

        * Shapley values, to assign importance values to each feature contribution in a model.
        * Beeswarm plot provides both local (for each instance) and 
            global (overall feature importance) insights into the model's behavior.
            * Features with a wider spread of points (more horizontal variation) 
                on the x-axis have a greater overall impact on the model's predictions.
            * Features with points clustered tightly around zero have less impact.
            * Color of the points indicates whether high or low feature values tend to push 
                the prediction in a positive or negative direction.
        """
        model_type = Param(Params._dummy(), "model_type", "Type of model (classifier/regressor)")
        algorithm = Param(Params._dummy(), "algorithm", "Algorithm to use (xgboost, lightgbm, randomforest)")
        n_samples = Param(Params._dummy(), "n_samples", "Number of samples to use for SHAP calculation")

        @keyword_only
        def __init__(
            self, 
            model_type: str = "classifier", 
            algorithm: str = "xgboost", 
            n_samples: int = 1000, 
            features_col: str = "features",
            feature_names: List[str] = [],
            target_col: str = "target", 
            logger=None, 
            **kwargs,
        ):
            ''' change the BaseImportanceMethod '''
            super().__init__(
                features_col=features_col, 
                feature_names=feature_names, 
                target_col=target_col, 
                logger=logger
            )
            # super().__init__(**kwargs)
            self._setDefault(model_type="classifier", algorithm="xgboost", n_samples=1000)
            kwargs = self._input_kwargs
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)

        def _train_model(self, X: pd.DataFrame, y: pd.Series, 
                        model_type: str, algorithm: str) -> BaseEstimator:
            """Train an appropriate model for SHAP analysis"""
            __s_fn_id__ = f"{self.__class__.__name__} method {inspect.currentframe().f_code.co_name}"
            
            # First detect if we should force regression
            if len(np.unique(y)) > 100:  # heuristic for continuous values
                model_type = "regressor"
                if self.logger:
                    self.logger.warning("%s switching to regressor for continuous target (found %d unique values)",
                                        __s_fn_id__, len(np.unique(y)))
            
            if model_type == "classifier":
                # Ensure y contains integers for classification
                y = y.astype(int)
                if algorithm == "xgboost":
                    model = XGBClassifier(random_state=42)
                elif algorithm == "lightgbm":
                    model = LGBMClassifier(random_state=42)
                else:
                    model = RandomForestClassifier(random_state=42)
            else:  # regression
                if algorithm == "xgboost":
                    model = XGBRegressor(random_state=42)
                elif algorithm == "lightgbm":
                    model = LGBMRegressor(random_state=42)
                else:
                    model = RandomForestRegressor(random_state=42)
                    
            model.fit(X, y)
            return model

        def _transform(self, dataset: DataFrame) -> DataFrame:
            __s_fn_id__ = f"class {self.__class__.__name__}"

            try:
                _model_type = self.getOrDefault(self.model_type)
                _algorithm = self.getOrDefault(self.algorithm)
                _n_samples = self.getOrDefault(self.n_samples)
                _target_col = self.getOrDefault(self.target_col)
                
                # Convert to pandas for SHAP analysis
                pdf = self._prepare_pandas_data(dataset)
                X = pdf.drop(columns=[_target_col])
                y = pdf[_target_col]
                
                # Train appropriate model
                model = self._train_model(X, y, _model_type, _algorithm)
                
                # Compute SHAP values
                explainer = shap.Explainer(model, X)
                self._shap_values = explainer(X)
                self._feature_names = list(X.columns)
                
                if self.logger:
                    self.logger.info("SHAP values computed successfully")

            except Exception as err:
                if self.logger:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                raise RuntimeError(f"[Error]{__s_fn_id__}: {err}")
    
            return dataset
            
        def _prepare_pandas_data(self, dataset: DataFrame) -> pd.DataFrame:
            """Convert Spark DataFrame to pandas for SHAP analysis with proper vector handling"""
            _features_col = self.getOrDefault(self.features_col)
            _target_col = self.getOrDefault(self.target_col)
            _feature_names = self.getOrDefault(self.feature_names)
            
            # Handle vector-assembled features
            if _features_col in dataset.columns:
                # Convert vector to array of floats
                dataset = dataset.withColumn(_features_col, vector_to_array(_features_col))
                
                # Convert to pandas and explode vector
                pdf = dataset.select(_features_col, _target_col).toPandas()
                features = np.array([x for x in pdf[_features_col]])
                feature_cols = [f"feature_{i}" for i in range(features.shape[1])]
                pdf[feature_cols] = pd.DataFrame(features, index=pdf.index)
                return pdf.drop(columns=[_features_col])
            
            # Handle individual columns (including vector-encoded columns)
            if not _feature_names:
                _feature_names = [col for col in dataset.columns if col != _target_col]
            
            # Convert all vector columns to arrays first
            vector_cols = [col for col in _feature_names 
                          if str(dataset.schema[col].dataType).startswith("Vector")]
            
            if vector_cols:
                dataset = dataset.select(
                    [vector_to_array(col).alias(col) if col in vector_cols else col 
                     for col in _feature_names] + [_target_col]
                )
            
            # Convert to pandas and ensure numeric types
            pdf = dataset.select(_feature_names + [_target_col]).toPandas()
            
            # Convert all feature columns to numeric
            for col in _feature_names:
                if pdf[col].dtype == object:
                    try:
                        # Handle array columns that became object dtype
                        if isinstance(pdf[col].iloc[0], (list, np.ndarray)):
                            # Explode array columns
                            arr_cols = {f"{col}_{i}": pdf[col].str[i] for i in range(len(pdf[col].iloc[0]))}
                            pdf = pd.concat([pdf.drop(columns=[col]), pd.DataFrame(arr_cols)], axis=1)
                        else:
                            pdf[col] = pd.to_numeric(pdf[col], errors='coerce')
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Could not convert column {col} to numeric: {str(e)}")
                        pdf[col] = pd.to_numeric(pdf[col], errors='coerce')
            
            return pdf

        def get_shap_feature_importance(self) -> pd.DataFrame:
            """Return DataFrame with SHAP feature importance"""
            if self._shap_values is None:
                raise ValueError("SHAP values not computed. Run transform first.")
                
            importance = pd.DataFrame({
                'feature': self._feature_names,
                'shap_importance': np.abs(self._shap_values.values).mean(axis=0)
            }).sort_values('shap_importance', ascending=False)
            
            return importance
            
        def get_shap_summary_plot(self, plot_type: str = "dot", **kwargs):
            """Generate SHAP summary plot"""
            if self._shap_values is None:
                raise ValueError("SHAP values not computed. Run transform first.")
                
            plt.figure()
            shap.summary_plot(self._shap_values, plot_type=plot_type, show=False, **kwargs)
            plt.tight_layout()
            return plt


    class PermutationImportanceAnalyzer(BaseImportanceMethod):
        """Compute permutation importance for feature evaluation.

        * a model-agnostic technique used to assess the importance of individual features in a ML model
        * bar plot visually represents how much a model's performance degrades 
            when a specific feature is randomly shuffled
                * Features with long bars are considered important - shuffling them significantly 
                    reduces the model's ability to make accurate predictions.
                * Features with short bars are less important -  permutation doesn't 
                    substantially affect model performance.
                * Feature with negative importance score suggests that the model might be 
                    making better predictions with that feature randomly shuffled. 
        """
        model_type = Param(Params._dummy(), "model_type", "Type of model (classifier/regressor)")
        algorithm = Param(Params._dummy(), "algorithm", "Algorithm to use (xgboost, lightgbm, randomforest)")
        n_repeats = Param(Params._dummy(), "n_repeats", "Number of repeats for permutation importance")
        
        @keyword_only
        def __init__(
            self, 
            model_type: str = "classifier", 
            algorithm: str = "randomforest",
            n_repeats: int = 5,
            features_col: str = "features",
            feature_names: List[str] = [],
            target_col: str = "target", 
            logger=None, 
            **kwargs,
        ):
            super().__init__(
                features_col=features_col,
                feature_names=feature_names,
                target_col=target_col,
                logger=logger
            )
            self._setDefault(model_type="classifier", algorithm="randomforest", n_repeats=5)
            kwargs = self._input_kwargs
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)
    
        def _train_model(self, X: pd.DataFrame, y: pd.Series, 
                        model_type: str, algorithm: str) -> BaseEstimator:
            """Train an appropriate model for permutation importance"""
            __s_fn_id__ = f"{self.__class__.__name__} method {inspect.currentframe().f_code.co_name}"
            
            # Auto-detect regression if target has many unique values
            if len(np.unique(y)) > 100:
                model_type = "regressor"
                if self.logger:
                    self.logger.warning("%s switching to regressor for continuous target (found %d unique values)",
                                      __s_fn_id__, len(np.unique(y)))
            
            if model_type == "classifier":
                y = y.astype(int)  # Ensure y contains integers for classification
                if algorithm == "xgboost":
                    model = XGBClassifier(random_state=42)
                elif algorithm == "lightgbm":
                    model = LGBMClassifier(random_state=42)
                else:
                    model = RandomForestClassifier(random_state=42)
            else:  # regression
                if algorithm == "xgboost":
                    model = XGBRegressor(random_state=42)
                elif algorithm == "lightgbm":
                    model = LGBMRegressor(random_state=42)
                else:
                    model = RandomForestRegressor(random_state=42)
                    
            model.fit(X, y)
            return model
    
        def _transform(self, dataset: DataFrame) -> DataFrame:
            __s_fn_id__ = f"{self.__class__.__name__} method {inspect.currentframe().f_code.co_name}"
        
            try:
                _model_type = self.getOrDefault(self.model_type)
                _algorithm = self.getOrDefault(self.algorithm)
                _n_repeats = self.getOrDefault(self.n_repeats)
                _target_col = self.getOrDefault(self.target_col)
                
                # Convert to pandas
                pdf = self._prepare_pandas_data(dataset)
                X = pdf.drop(columns=[_target_col])
                y = pdf[_target_col]
                
                # Train model
                model = self._train_model(X, y, _model_type, _algorithm)
                
                # Compute permutation importance
                result = permutation_importance(
                    model, X, y, n_repeats=_n_repeats, random_state=42
                )
                
                # First store the basic results
                self._importance_results = {
                    'importances_mean': result.importances_mean,
                    'importances_std': result.importances_std,
                    'feature_names': list(X.columns),
                    'summary_plot': None  # Will be populated later
                }
                
                # Now generate and store the plot
                plt = self.plot_permutation_importance()
                self._importance_results['summary_plot'] = plt.gcf()
                plt.close()
                
                if self.logger:
                    self.logger.info("Permutation importance computed successfully")
        
            except Exception as err:
                if self.logger:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                raise RuntimeError(f"[Error]{__s_fn_id__}: {err}")
        
            return dataset

        def _prepare_pandas_data(self, dataset: DataFrame) -> pd.DataFrame:
            """Convert Spark DataFrame to pandas with proper vector handling"""
            _features_col = self.getOrDefault(self.features_col)
            _target_col = self.getOrDefault(self.target_col)
            _feature_names = self.getOrDefault(self.feature_names)
            
            # Handle vector-assembled features
            if _features_col in dataset.columns:
                dataset = dataset.withColumn(_features_col, vector_to_array(_features_col))
                pdf = dataset.select(_features_col, _target_col).toPandas()
                features = np.array([x for x in pdf[_features_col]])
                feature_cols = [f"feature_{i}" for i in range(features.shape[1])]
                pdf[feature_cols] = pd.DataFrame(features, index=pdf.index)
                return pdf.drop(columns=[_features_col])
            
            # Handle individual columns
            if not _feature_names:
                _feature_names = [col for col in dataset.columns if col != _target_col]
            
            # Convert vector columns to arrays
            vector_cols = [col for col in _feature_names 
                          if str(dataset.schema[col].dataType).startswith("Vector")]
            
            if vector_cols:
                dataset = dataset.select(
                    [vector_to_array(col).alias(col) if col in vector_cols else col 
                     for col in _feature_names] + [_target_col]
                )
            
            # Convert to pandas and ensure numeric types
            pdf = dataset.select(_feature_names + [_target_col]).toPandas()
            
            for col in _feature_names:
                if pdf[col].dtype == object:
                    try:
                        if isinstance(pdf[col].iloc[0], (list, np.ndarray)):
                            arr_cols = {f"{col}_{i}": pdf[col].str[i] for i in range(len(pdf[col].iloc[0]))}
                            pdf = pd.concat([pdf.drop(columns=[col]), pd.DataFrame(arr_cols)], axis=1)
                        else:
                            pdf[col] = pd.to_numeric(pdf[col], errors='coerce')
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Could not convert column {col} to numeric: {str(e)}")
                        pdf[col] = pd.to_numeric(pdf[col], errors='coerce')
            
            return pdf
    
        def get_permutation_importance(self) -> pd.DataFrame:
            """Return DataFrame with permutation feature importance"""
            if not hasattr(self, '_importance_results') or self._importance_results is None:
                raise ValueError("Permutation importance not computed. Run transform first.")
            
            if 'importances_mean' not in self._importance_results:
                raise ValueError("Importance values not found in results")
                
            return pd.DataFrame({
                'feature': self._importance_results['feature_names'],
                'importance_mean': self._importance_results['importances_mean'],
                'importance_std': self._importance_results['importances_std']
            }).sort_values('importance_mean', ascending=False)
            
        def plot_permutation_importance(self, top_n: int = 20, **kwargs):
            """Plot permutation importance results"""
            importance_df = self.get_permutation_importance()
            top_features = importance_df.head(top_n)
            
            plt.figure(figsize=(10, 6))
            plt.barh(
                top_features['feature'], 
                top_features['importance_mean'],
                xerr=top_features['importance_std'], 
                capsize=5,
                **kwargs
            )
            plt.xlabel('Mean Importance Score')
            plt.title('Permutation Importance (mean ± std)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            return plt

    class ModelFeatureImportance(BaseImportanceMethod):
        """
        Extract feature importance from trained models (Gini/impurity for tree-based models).

        * Analysis is crucial for model interpretability, feature selection, and identifying potential biases
        * bar plot visually represents how much each feature contributes to a model's predictions
            * Look for the longest bars, as they represent the features with the highest impact
            * standard deviation (error bars) indicates how consistent the importance 
                of a feature is across different data points
        """
        model_type = Param(Params._dummy(), "model_type", "Type of model (classifier/regressor)")
        algorithm = Param(Params._dummy(), "algorithm", "Algorithm to use (xgboost, lightgbm, randomforest)")
        
        @keyword_only
        def __init__(
            self, 
            model_type: str = "classifier", 
            algorithm: str = "randomforest",
            features_col: str = "features",
            feature_names: List[str] = [],
            target_col: str = "target",
            logger=None,
            **kwargs
        ):
            super().__init__(
                features_col=features_col,
                feature_names=feature_names,
                target_col=target_col,
                logger=logger
            )
            self._setDefault(model_type="classifier", algorithm="randomforest")
            kwargs = self._input_kwargs
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)
    
        def _train_model(self, X: pd.DataFrame, y: pd.Series,
                        model_type: str, algorithm: str) -> BaseEstimator:
            """Train an appropriate model and return feature importances"""
            __s_fn_id__ = f"{self.__class__.__name__} method {inspect.currentframe().f_code.co_name}"
            
            # Auto-detect regression if target has many unique values
            if len(np.unique(y)) > 100:
                model_type = "regressor"
                if self.logger:
                    self.logger.warning("%s switching to regressor for continuous target (found %d unique values)",
                                      __s_fn_id__, len(np.unique(y)))
            
            if model_type == "classifier":
                y = y.astype(int)
                if algorithm == "xgboost":
                    model = XGBClassifier(random_state=42)
                elif algorithm == "lightgbm":
                    model = LGBMClassifier(random_state=42)
                else:
                    model = RandomForestClassifier(random_state=42)
            else:  # regression
                if algorithm == "xgboost":
                    model = XGBRegressor(random_state=42)
                elif algorithm == "lightgbm":
                    model = LGBMRegressor(random_state=42)
                else:
                    model = RandomForestRegressor(random_state=42)
                    
            model.fit(X, y)
            return model
    
        def _transform(self, dataset: DataFrame) -> DataFrame:
            __s_fn_id__ = f"{self.__class__.__name__} method {inspect.currentframe().f_code.co_name}"
        
            try:
                _model_type = self.getOrDefault(self.model_type)
                _algorithm = self.getOrDefault(self.algorithm)
                _target_col = self.getOrDefault(self.target_col)
                
                # Convert to pandas
                pdf = self._prepare_pandas_data(dataset)
                X = pdf.drop(columns=[_target_col])
                y = pdf[_target_col]
                
                # Train model
                model = self._train_model(X, y, _model_type, _algorithm)
                
                # Get feature importance
                importances = model.feature_importances_
                feature_names = list(X.columns)
                
                # First store the basic results
                self._importance_results = {
                    'importances': importances,
                    'feature_names': feature_names,
                    'summary_plot': None  # Will be populated later
                }
                
                # Now generate and store the plot using the stored results
                plt = self._generate_importance_plot()
                self._importance_results['summary_plot'] = plt.gcf()
                plt.close()
                
                if self.logger:
                    self.logger.info("Model feature importance computed successfully")
        
            except Exception as err:
                if self.logger:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                raise RuntimeError(f"[Error]{__s_fn_id__}: {err}")
        
            return dataset

        def _prepare_pandas_data(self, dataset: DataFrame) -> pd.DataFrame:
            """Convert Spark DataFrame to pandas with proper vector handling"""
            _features_col = self.getOrDefault(self.features_col)
            _target_col = self.getOrDefault(self.target_col)
            _feature_names = self.getOrDefault(self.feature_names)
            
            # Handle vector-assembled features
            if _features_col in dataset.columns:
                dataset = dataset.withColumn(_features_col, vector_to_array(_features_col))
                pdf = dataset.select(_features_col, _target_col).toPandas()
                features = np.array([x for x in pdf[_features_col]])
                feature_cols = [f"feature_{i}" for i in range(features.shape[1])]
                pdf[feature_cols] = pd.DataFrame(features, index=pdf.index)
                return pdf.drop(columns=[_features_col])
            
            # Handle individual columns
            if not _feature_names:
                _feature_names = [col for col in dataset.columns if col != _target_col]
            
            # Convert vector columns to arrays
            vector_cols = [col for col in _feature_names 
                          if str(dataset.schema[col].dataType).startswith("Vector")]
            
            if vector_cols:
                dataset = dataset.select(
                    [vector_to_array(col).alias(col) if col in vector_cols else col 
                     for col in _feature_names] + [_target_col]
                )
            
            # Convert to pandas and ensure numeric types
            pdf = dataset.select(_feature_names + [_target_col]).toPandas()
            
            for col in _feature_names:
                if pdf[col].dtype == object:
                    try:
                        if isinstance(pdf[col].iloc[0], (list, np.ndarray)):
                            arr_cols = {f"{col}_{i}": pdf[col].str[i] for i in range(len(pdf[col].iloc[0]))}
                            pdf = pd.concat([pdf.drop(columns=[col]), pd.DataFrame(arr_cols)], axis=1)
                        else:
                            pdf[col] = pd.to_numeric(pdf[col], errors='coerce')
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Could not convert column {col} to numeric: {str(e)}")
                        pdf[col] = pd.to_numeric(pdf[col], errors='coerce')
            
            return pdf
    
        def get_model_importance(self) -> pd.DataFrame:
            """Return DataFrame with model feature importance"""
            if not hasattr(self, '_importance_results') or self._importance_results is None:
                raise ValueError("Model importance not computed. Run transform first.")
                
            if 'importances' not in self._importance_results:
                raise ValueError("Importance values not found in results")
                
            return pd.DataFrame({
                'feature': self._importance_results['feature_names'],
                'importance': self._importance_results['importances']
            }).sort_values('importance', ascending=False)
            
        def _generate_importance_plot(self) -> plt:
            """Internal method to generate importance plot without validation checks"""
            importance_df = pd.DataFrame({
                'feature': self._importance_results['feature_names'],
                'importance': self._importance_results['importances']
            }).sort_values('importance', ascending=False)
            
            top_features = importance_df.head(20)
            
            plt.figure(figsize=(10, 6))
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('Importance Score')
            plt.title('Model Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            return plt
        
        def plot_model_importance(self, top_n: int = 20, **kwargs):
            """Public method to plot feature importance"""
            if not hasattr(self, '_importance_results') or self._importance_results is None:
                raise ValueError("Model importance not computed. Run transform first.")
                
            plt = self._generate_importance_plot()
            if kwargs:
                for bar in plt.gca().patches:
                    bar.set(**kwargs)
            return plt
    # Builder methods
    def add_shap_analysis(self, model_type: str = "classifier", 
                         algorithm: str = "xgboost", n_samples: int = 1000) -> 'mlWorkLoads':
        """Add SHAP analysis to the pipeline"""
        self.stages.append(
            self.SHAPAnalyzer(
                model_type=model_type,
                algorithm=algorithm,
                n_samples=n_samples,
                features_col=self.features,
                feature_names=self.featNames,
                target_col=self.target,
                logger=self._logger,
            )
        )
        return self
        
    def add_permutation_importance(self, model_type: str = "classifier",
                                  algorithm: str = "randomforest", n_repeats: int = 5):
        """Add permutation importance analysis to the pipeline"""
        self.stages.append(
            self.PermutationImportanceAnalyzer(
                model_type=model_type,
                algorithm=algorithm,
                n_repeats=n_repeats,
                features_col=self.features,
                feature_names=self.featNames,
                target_col=self.target,
                logger=self._logger,
            )
        )
        return self
        
    def add_model_importance(self, model_type: str = "classifier",
                           algorithm: str = "randomforest"):
        """Add model-specific feature importance to the pipeline"""
        self.stages.append(
            self.ModelFeatureImportance(
                model_type=model_type,
                algorithm=algorithm,
                features_col=self.features,
                feature_names=self.featNames,
                target_col=self.target,
                logger=self._logger,
            )
        )
        return self

    def add_chi_sq_selection(self, k: int = 10):
        """Add Chi-Squared feature selection (requires categorical target)"""
        if not self.target_col:
            raise ValueError("ChiSqSelector requires target_col to be set")
        self.stages.append(self.ChiSqFeatureSelector(k=k))
        return self

    def build(self):
        """Finalize and return the pipeline model"""
        return Pipeline(stages=self.stages)
        
    def exec_pipe_with_stages(self) -> tuple:
        """
        Executes pipeline step-by-step, collecting importance results and plots from each stage.
    
        Returns:
            tuple: (intermediate_results, final_df, importance_results)
                intermediate_results: dict of stage_name -> transformed DataFrame
                final_df: final transformed DataFrame
                importance_results: dict containing {
                    'shap': {
                        'values': shap_values,
                        'feature_names': list,
                        'summary_plot': matplotlib.figure.Figure
                    },
                    'permutation': {...},
                    ...
                }
        """
    
        __s_fn_id__ = f"{self.__name__} method {self.__class__.__name__}"
    
        try:
            if self.data is None:
                raise ValueError("No data provided for pipeline execution")
            current_df = self.data
            intermediate_results = {}
            importance_results = {}
            
            for i, stage in enumerate(self.stages):
                stage_name = f"stage_{i}_{stage.__class__.__name__}"
                self._logger.info(f"Executing {stage_name}...")
                
                current_df = stage.transform(current_df)
                intermediate_results[stage_name] = current_df
                
                # Collect importance results if available
                if hasattr(stage, '_shap_values'):
                    # Generate SHAP summary plot if the method exists
                    if hasattr(stage, 'get_shap_summary_plot'):
                        try:
                            plt = stage.get_shap_summary_plot()
                            importance_results['shap'] = {
                                'values': stage._shap_values,
                                'feature_names': getattr(stage, '_feature_names', []),
                                'summary_plot': plt.gcf()  # Get the current figure
                            }
                            plt.close()  # Close the plot to free memory
                        except Exception as plot_err:
                            self._logger.warning(f"Failed to generate SHAP plot: {plot_err}")
                    else:
                        importance_results['shap'] = {
                            'values': stage._shap_values,
                            'feature_names': getattr(stage, '_feature_names', [])
                        }
                elif hasattr(stage, '_importance_results'):
                    method = stage.__class__.__name__.replace("Analyzer", "").lower()
                    importance_results[method] = stage._importance_results
                
                current_df.cache().count()
        
        except Exception as err:
            self._logger.error("%s %s \n",__s_fn_id__, err)
            self._logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
    
        finally:
            return intermediate_results, current_df, importance_results