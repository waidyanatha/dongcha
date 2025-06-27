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
    import functools
    import traceback
 
    import findspark
    findspark.init()
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        NumericType, IntegerType, DoubleType, FloatType, StringType)
    from pyspark.sql import DataFrame
    from pyspark.ml.feature import Imputer, VarianceThresholdSelector, VectorAssembler, StandardScaler
    # from pyspark.ml.feature import UnivariateFeatureSelector, OneHotEncoder, StringIndexer
    from pyspark.ml.feature import (
        StringIndexer, OneHotEncoder, VectorAssembler, 
        Imputer, VarianceThresholdSelector, StandardScaler
    )
    from pyspark.ml.stat import Correlation
    from pyspark.ml import Pipeline, Transformer
    from pyspark.ml.param.shared import Param, Params
    from pyspark import keyword_only
    from typing import List, Iterable, Dict, Tuple


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
# class dataWorkLoads(attr.properties):
    """Main class for feature engineering pipeline"""
    
    def __init__(self, df: DataFrame, target_col: str, features_col: str = "features"):
        self.data = sdf
        self.pipeline_stages = []
        self._column_registry = set(df.columns)
    
"""
        Initialize the feature importance analyzer.
        
        Args:
            df (DataFrame): Spark DataFrame containing features and target
            target_col (str): Name of the target column
            features_col (str): Name of the assembled features column (default: "features")
        """
        self.df = df
        self.target_col = target_col
        self.features_col = features_col
        self._pandas_df = None
        self._feature_names = None
        self._shap_values = None
        self._importance_results = {}
        
    class BaseImportanceMethod(Transformer):
        """Base class for feature importance methods"""
        @keyword_only
        def __init__(self):
            super().__init__()
            
        def _transform(self, dataset: DataFrame) -> DataFrame:
            raise NotImplementedError
            
        def _get_feature_names(self, dataset: DataFrame) -> List[str]:
            """Extract feature names from VectorAssembler output"""
            if hasattr(self, "_feature_names"):
                return self._feature_names
                
            # Get the first row of features
            sample = dataset.select(self.features_col).first()[0]
            if hasattr(sample, "indices"):  # Sparse vector
                size = sample.size
            else:  # Dense vector
                size = len(sample)
                
            return [f"feature_{i}" for i in range(size)]
    
    class SHAPAnalyzer(BaseImportanceMethod):
        """
        Compute SHAP (SHapley Additive exPlanations) values for feature importance.
        Supports both classification and regression problems.
        """
        model_type = Param(Params._dummy(), "model_type", "Type of model (classifier/regressor)")
        algorithm = Param(Params._dummy(), "algorithm", "Algorithm to use (xgboost, lightgbm, randomforest)")
        n_samples = Param(Params._dummy(), "n_samples", "Number of samples to use for SHAP calculation")
        
        @keyword_only
        def __init__(self, model_type: str = "classifier", algorithm: str = "xgboost", 
                    n_samples: int = 1000):
            super().__init__()
            self._setDefault(model_type="classifier", algorithm="xgboost", n_samples=1000)
            kwargs = self._input_kwargs
            self._set(**kwargs)
            
        def _transform(self, dataset: DataFrame) -> DataFrame:
            model_type = self.getOrDefault(self.model_type)
            algorithm = self.getOrDefault(self.algorithm)
            n_samples = self.getOrDefault(self.n_samples)
            
            # Convert to pandas for SHAP analysis
            pdf = self._prepare_pandas_data(dataset)
            X = pdf.drop(columns=[self.target_col])
            y = pdf[self.target_col]
            
            # Train appropriate model
            model = self._train_model(X, y, model_type, algorithm)
            
            # Compute SHAP values
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            
            # Store results
            self._shap_values = shap_values
            self._feature_names = list(X.columns)
            
            # Return original DataFrame (SHAP results are stored in the analyzer)
            return dataset
            
        def _prepare_pandas_data(self, dataset: DataFrame) -> pd.DataFrame:
            """Convert Spark DataFrame to pandas for SHAP analysis"""
            # Convert vector features to columns
            assembler = VectorAssembler(
                inputCols=[col for col in dataset.columns if col != self.target_col],
                outputCol=self.features_col
            )
            assembled = assembler.transform(dataset)
            
            # Convert to pandas
            pdf = assembled.select(self.features_col, self.target_col).toPandas()
            
            # Convert vector to columns
            features = np.array([x.toArray() for x in pdf[self.features_col]])
            feature_cols = [f"feature_{i}" for i in range(features.shape[1])]
            pdf[feature_cols] = pd.DataFrame(features, index=pdf.index)
            
            return pdf.drop(columns=[self.features_col])
            
        def _train_model(self, X: pd.DataFrame, y: pd.Series, 
                        model_type: str, algorithm: str) -> BaseEstimator:
            """Train an appropriate model for SHAP analysis"""
            if model_type == "classifier":
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
            
        def get_shap_summary_plot(self, plot_type: str = "dot", **kwargs):
            """Generate SHAP summary plot"""
            if self._shap_values is None:
                raise ValueError("SHAP values not computed. Run transform first.")
                
            plt.figure()
            shap.summary_plot(self._shap_values, plot_type=plot_type, **kwargs)
            plt.tight_layout()
            return plt
            
        def get_shap_feature_importance(self) -> pd.DataFrame:
            """Return DataFrame with SHAP feature importance"""
            if self._shap_values is None:
                raise ValueError("SHAP values not computed. Run transform first.")
                
            # Calculate mean absolute SHAP values
            importance = pd.DataFrame({
                'feature': self._feature_names,
                'shap_importance': np.abs(self._shap_values.values).mean(axis=0)
            }).sort_values('shap_importance', ascending=False)
            
            return importance
    
    class PermutationImportanceAnalyzer(BaseImportanceMethod):
        """
        Compute permutation importance for feature evaluation.
        """
        model_type = Param(Params._dummy(), "model_type", "Type of model (classifier/regressor)")
        algorithm = Param(Params._dummy(), "algorithm", "Algorithm to use (xgboost, lightgbm, randomforest)")
        n_repeats = Param(Params._dummy(), "n_repeats", "Number of repeats for permutation importance")
        
        @keyword_only
        def __init__(self, model_type: str = "classifier", algorithm: str = "randomforest",
                    n_repeats: int = 5):
            super().__init__()
            self._setDefault(model_type="classifier", algorithm="randomforest", n_repeats=5)
            kwargs = self._input_kwargs
            self._set(**kwargs)
            
        def _transform(self, dataset: DataFrame) -> DataFrame:
            model_type = self.getOrDefault(self.model_type)
            algorithm = self.getOrDefault(self.algorithm)
            n_repeats = self.getOrDefault(self.n_repeats)
            
            # Convert to pandas
            pdf = self._prepare_pandas_data(dataset)
            X = pdf.drop(columns=[self.target_col])
            y = pdf[self.target_col]
            
            # Train model
            model = self._train_model(X, y, model_type, algorithm)
            
            # Compute permutation importance
            result = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=42
            )
            
            # Store results
            self._importance_results = {
                'importances_mean': result.importances_mean,
                'importances_std': result.importances_std,
                'feature_names': list(X.columns)
            }
            
            return dataset
            
        def _prepare_pandas_data(self, dataset: DataFrame) -> pd.DataFrame:
            """Convert Spark DataFrame to pandas for analysis"""
            # Convert vector features to columns
            assembler = VectorAssembler(
                inputCols=[col for col in dataset.columns if col != self.target_col],
                outputCol=self.features_col
            )
            assembled = assembler.transform(dataset)
            
            # Convert to pandas
            pdf = assembled.select(self.features_col, self.target_col).toPandas()
            
            # Convert vector to columns
            features = np.array([x.toArray() for x in pdf[self.features_col]])
            feature_cols = [f"feature_{i}" for i in range(features.shape[1])]
            pdf[feature_cols] = pd.DataFrame(features, index=pdf.index)
            
            return pdf.drop(columns=[self.features_col])
            
        def _train_model(self, X: pd.DataFrame, y: pd.Series, 
                        model_type: str, algorithm: str) -> BaseEstimator:
            """Train an appropriate model for permutation importance"""
            if model_type == "classifier":
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
            
        def get_permutation_importance(self) -> pd.DataFrame:
            """Return DataFrame with permutation feature importance"""
            if not self._importance_results:
                raise ValueError("Permutation importance not computed. Run transform first.")
                
            importance = pd.DataFrame({
                'feature': self._importance_results['feature_names'],
                'importance_mean': self._importance_results['importances_mean'],
                'importance_std': self._importance_results['importances_std']
            }).sort_values('importance_mean', ascending=False)
            
            return importance
            
        def plot_permutation_importance(self, top_n: int = 20):
            """Plot permutation importance results"""
            importance_df = self.get_permutation_importance()
            top_features = importance_df.head(top_n)
            
            plt.figure(figsize=(10, 6))
            plt.barh(top_features['feature'], top_features['importance_mean'],
                    xerr=top_features['importance_std'], capsize=5)
            plt.xlabel('Mean Importance Score')
            plt.title('Permutation Importance (mean ± std)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            return plt
    
    class ModelFeatureImportance(BaseImportanceMethod):
        """
        Extract feature importance from trained models (Gini/impurity for tree-based models).
        """
        model_type = Param(Params._dummy(), "model_type", "Type of model (classifier/regressor)")
        algorithm = Param(Params._dummy(), "algorithm", "Algorithm to use (xgboost, lightgbm, randomforest)")
        
        @keyword_only
        def __init__(self, model_type: str = "classifier", algorithm: str = "randomforest"):
            super().__init__()
            self._setDefault(model_type="classifier", algorithm="randomforest")
            kwargs = self._input_kwargs
            self._set(**kwargs)
            
        def _transform(self, dataset: DataFrame) -> DataFrame:
            model_type = self.getOrDefault(self.model_type)
            algorithm = self.getOrDefault(self.algorithm)
            
            # Convert to pandas
            pdf = self._prepare_pandas_data(dataset)
            X = pdf.drop(columns=[self.target_col])
            y = pdf[self.target_col]
            
            # Train model
            model = self._train_model(X, y, model_type, algorithm)
            
            # Get feature importance
            importances = model.feature_importances_
            feature_names = list(X.columns)
            
            # Store results
            self._importance_results = {
                'importances': importances,
                'feature_names': feature_names
            }
            
            return dataset
            
        def get_model_importance(self) -> pd.DataFrame:
            """Return DataFrame with model feature importance"""
            if not self._importance_results:
                raise ValueError("Model importance not computed. Run transform first.")
                
            importance = pd.DataFrame({
                'feature': self._importance_results['feature_names'],
                'importance': self._importance_results['importances']
            }).sort_values('importance', ascending=False)
            
            return importance
            
        def plot_model_importance(self, top_n: int = 20):
            """Plot model feature importance"""
            importance_df = self.get_model_importance()
            top_features = importance_df.head(top_n)
            
            plt.figure(figsize=(10, 6))
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('Importance Score')
            plt.title('Model Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            return plt
    
    # Builder methods
    def add_shap_analysis(self, model_type: str = "classifier", 
                         algorithm: str = "xgboost", n_samples: int = 1000):
        """Add SHAP analysis to the pipeline"""
        self.pipeline_stages.append(
            self.SHAPAnalyzer(
                model_type=model_type,
                algorithm=algorithm,
                n_samples=n_samples
            )
        )
        return self
        
    def add_permutation_importance(self, model_type: str = "classifier",
                                  algorithm: str = "randomforest", n_repeats: int = 5):
        """Add permutation importance analysis to the pipeline"""
        self.pipeline_stages.append(
            self.PermutationImportanceAnalyzer(
                model_type=model_type,
                algorithm=algorithm,
                n_repeats=n_repeats
            )
        )
        return self
        
    def add_model_importance(self, model_type: str = "classifier",
                           algorithm: str = "randomforest"):
        """Add model-specific feature importance to the pipeline"""
        self.pipeline_stages.append(
            self.ModelFeatureImportance(
                model_type=model_type,
                algorithm=algorithm
            )
        )
        return self

    def add_chi_sq_selection(self, k: int = 10):
        """Add Chi-Squared feature selection (requires categorical target)"""
        if not self.target_col:
            raise ValueError("ChiSqSelector requires target_col to be set")
        self.pipeline_stages.append(self.ChiSqFeatureSelector(k=k))
        return self

    def build(self):
        """Finalize and return the pipeline model"""
        return Pipeline(stages=self.pipeline_stages)
        
    def exec_pipe_with_stages(self):
        """
        Executes pipeline step-by-step, collecting importance results from each stage.
        
        Returns:
            - intermediates (dict): {"stage_name": intermediate_df}
            - final_df (DataFrame): Final transformed output
            - importance_results (dict): Results from importance analyses
        """
        current_df = self.df
        self.intermediate_results = {}
        self.importance_results = {}
        
        for i, stage in enumerate(self.pipeline_stages):
            stage_name = f"stage_{i}_{stage.__class__.__name__}"
            print(f"Executing {stage_name}...")
            
            # Apply the current stage
            current_df = stage.transform(current_df)
            
            # Store intermediate result
            self.intermediate_results[stage_name] = current_df
            
            # Collect importance results if available
            if hasattr(stage, '_shap_values'):
                self.importance_results['shap'] = {
                    'values': stage._shap_values,
                    'feature_names': stage._feature_names
                }
            elif hasattr(stage, '_importance_results'):
                method = stage.__class__.__name__.replace("Analyzer", "").replace("Importance", "").lower()
                self.importance_results[method] = stage._importance_results
            
            # Force execution (avoids lazy evaluation issues)
            current_df.cache().count()  
        
        return self.intermediate_results, current_df, self.importance_results