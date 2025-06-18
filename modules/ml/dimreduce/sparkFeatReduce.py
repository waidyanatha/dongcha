#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "sparkFeatReduce"
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
class dataWorkLoads:
# class dataWorkLoads(attr.properties):
    """Main class for feature engineering pipeline"""
    
    def __init__(self, df):
        self.df = df
        self.pipeline_stages = []
        self._column_registry = set(df.columns)
    
    class NullColumnRemover(Transformer):
        """Remove columns with high null percentage"""
        null_threshold = Param(Params._dummy(), "null_threshold", "Threshold for null percentage")
        
        @keyword_only
        def __init__(self, null_threshold=0.7):
            super().__init__()
            kwargs = self._input_kwargs
            self._set(**kwargs)
        
        def _transform(self, dataset):
            null_thresh = self.getOrDefault(self.null_threshold)
            total_rows = dataset.count()
            
            # Calculate null percentages
            null_percentages = {}
            for col in dataset.columns:
                null_count = dataset.filter(F.col(col).isNull()).count()
                null_percentages[col] = null_count / total_rows
            
            # Identify columns to drop
            to_drop = [col for col, percent in null_percentages.items() 
                      if percent > null_thresh]
            
            if to_drop:
                print(f"Dropping columns with >{null_thresh*100}% nulls: {to_drop}")
                return dataset.drop(*to_drop)
            return dataset
    
    class VarianceThresholdRemover(Transformer):
        """Remove low variance columns"""
        variance_threshold = Param(Params._dummy(), "variance_threshold", "Minimum variance threshold")
        
        @keyword_only
        def __init__(self, variance_threshold=0.1):
            super().__init__()
            kwargs = self._input_kwargs
            self._set(**kwargs)
        
        def _transform(self, dataset):
            var_thresh = self.getOrDefault(self.variance_threshold)
            numeric_cols = [f.name for f in dataset.schema.fields 
                          if isinstance(f.dataType, NumericType)]
            
            if not numeric_cols:
                return dataset
                
            # Create null-safe temporary columns
            temp_cols = []
            for col in numeric_cols:
                temp_col = f"temp_{col}"
                dataset = dataset.withColumn(
                    temp_col,
                    F.when(F.col(col).isNull(), 0).otherwise(F.col(col))  # Fixed syntax
                )
                temp_cols.append(temp_col)
            
            # Calculate variance
            assembler = VectorAssembler(
                inputCols=temp_cols,
                outputCol="temp_features",
                handleInvalid="keep"
            )
            assembled = assembler.transform(dataset)
            
            selector = VarianceThresholdSelector(
                varianceThreshold=var_thresh,
                featuresCol="temp_features",
                outputCol="filtered_features"
            )
            model = selector.fit(assembled)
            selected = model.selectedFeatures
            to_keep = [numeric_cols[i] for i in selected]
            to_drop = set(numeric_cols) - set(to_keep)
            
            # Clean up
            if to_drop:
                print(f"Dropping low-variance columns: {to_drop}")
                dataset = dataset.drop(*temp_cols).drop(*to_drop)
            return dataset
    
    class CorrelationFilter(Transformer):
        """Remove highly correlated features"""
        threshold = Param(Params._dummy(), "threshold", "Correlation threshold")
        
        @keyword_only
        def __init__(self, threshold=0.9):
            super().__init__()
            kwargs = self._input_kwargs
            self._set(**kwargs)
        
        def _transform(self, dataset):
            corr_thresh = self.getOrDefault(self.threshold)
            numeric_cols = [f.name for f in dataset.schema.fields 
                          if isinstance(f.dataType, NumericType)]
            
            if len(numeric_cols) < 2:
                return dataset
                
            assembler = VectorAssembler(
                inputCols=numeric_cols,
                outputCol="temp_corr_features",
                handleInvalid="keep"
            )
            assembled = assembler.transform(dataset)
            
            corr_matrix = Correlation.corr(assembled, "temp_corr_features").collect()[0][0]
            corr_array = corr_matrix.toArray()
            
            to_drop = set()
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    if abs(corr_array[i][j]) > corr_thresh:
                        to_drop.add(numeric_cols[j])
            
            if to_drop:
                print(f"Dropping correlated features (r > {corr_thresh}): {sorted(to_drop)}")
                dataset = dataset.drop(*to_drop)
            
            return dataset.drop("temp_corr_features")
    
    class NullHandler(Transformer):
        """Handle null values in numeric and categorical columns"""
        numeric_strategy = Param(Params._dummy(), "numeric_strategy", "Imputation strategy for numerics")
        categorical_strategy = Param(Params._dummy(), "categorical_strategy", "Imputation strategy for categoricals")
        
        @keyword_only
        def __init__(self, numeric_strategy="median", categorical_strategy="mode"):
            super().__init__()
            kwargs = self._input_kwargs
            self._set(**kwargs)
        
        def _transform(self, dataset):
            numeric_strat = self.getOrDefault(self.numeric_strategy)
            categorical_strat = self.getOrDefault(self.categorical_strategy)
            
            # Numeric imputation
            numeric_cols = [f.name for f in dataset.schema.fields 
                          if isinstance(f.dataType, NumericType)]
            
            if numeric_cols:
                imputer = Imputer(
                    inputCols=numeric_cols,
                    outputCols=numeric_cols,  # Overwrite original columns
                    strategy=numeric_strat
                )
                dataset = imputer.fit(dataset).transform(dataset)
            
            # Categorical imputation - more robust implementation
            categorical_cols = [f.name for f in dataset.schema.fields 
                              if str(f.dataType) == 'StringType()']
            
            for col in categorical_cols:
                # Get mode value safely
                mode_df = dataset.groupBy(col).count().orderBy("count", ascending=False)
                if mode_df.count() > 0:
                    mode_row = mode_df.first()
                    if mode_row is not None:
                        mode_val = mode_row[0]
                        if mode_val is not None:  # Only fill if we found a valid mode
                            dataset = dataset.fillna({col: mode_val})
                        else:
                            # If mode is None, fill with a default string
                            dataset = dataset.fillna({col: "missing"})
                    else:
                        dataset = dataset.fillna({col: "missing"})
                else:
                    dataset = dataset.fillna({col: "missing"})
            
            return dataset

    class CategoricalEncoder(Transformer):
        """Encode categorical variables"""
        @keyword_only
        def __init__(self):
            super().__init__()
        
        def _transform(self, dataset):
            categorical_cols = [f.name for f in dataset.schema.fields 
                              if str(f.dataType) == 'StringType()']
            
            indexers = [StringIndexer(
                inputCol=col,
                outputCol=f"{col}_indexed",
                handleInvalid="keep"
            ) for col in categorical_cols]
            
            encoders = [OneHotEncoder(
                inputCol=f"{col}_indexed",
                outputCol=f"{col}_encoded"
            ) for col in categorical_cols]
            
            pipeline = Pipeline(stages=indexers + encoders)
            return pipeline.fit(dataset).transform(dataset)
    
    class FeatureScaler(Transformer):
        """Standardize numeric features"""
        @keyword_only
        def __init__(self):
            super().__init__()
        
        def _transform(self, dataset):
            numeric_cols = [f.name for f in dataset.schema.fields 
                          if isinstance(f.dataType, NumericType)]
            
            scaler = StandardScaler(
                inputCol="features",
                outputCol="scaled_features",
                withStd=True,
                withMean=True
            )
            
            assembler = VectorAssembler(
                inputCols=numeric_cols,
                outputCol="features"
            )
            
            pipeline = Pipeline(stages=[assembler, scaler])
            return pipeline.fit(dataset).transform(dataset)
    
    class FeatureAssembler(Transformer):
        """Assemble final feature vector"""
        output_col = Param(Params._dummy(), "output_col", "Output column name")
        
        @keyword_only
        def __init__(self, output_col="features"):
            super().__init__()
            kwargs = self._input_kwargs
            self._set(**kwargs)
        
        def _transform(self, dataset):
            output_col = self.getOrDefault(self.output_col)
            numeric_cols = [f.name for f in dataset.schema.fields 
                          if isinstance(f.dataType, NumericType)]
            encoded_cols = [col for col in dataset.columns if col.endswith("_encoded")]
            
            assembler = VectorAssembler(
                inputCols=numeric_cols + encoded_cols,
                outputCol=output_col
            )
            return assembler.transform(dataset)
    
    # Builder methods
    def remove_high_null_columns(self, threshold=0.7):
        self.pipeline_stages.append(self.NullColumnRemover(null_threshold=threshold))
        return self
    
    def remove_low_variance_columns(self, threshold=0.1):
        self.pipeline_stages.append(self.VarianceThresholdRemover(variance_threshold=threshold))
        return self
    
    def add_correlation_filter(self, threshold=0.9):
        self.pipeline_stages.append(self.CorrelationFilter(threshold=threshold))
        return self
    
    def add_null_handling(self, numeric_strategy="median", categorical_strategy="mode"):
        self.pipeline_stages.append(self.NullHandler(
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy
        ))
        return self
    
    def add_categorical_encoding(self):
        self.pipeline_stages.append(self.CategoricalEncoder())
        return self
    
    def add_feature_scaling(self):
        self.pipeline_stages.append(self.FeatureScaler())
        return self
    
    def add_feature_assembly(self, output_col="features"):
        self.pipeline_stages.append(self.FeatureAssembler(output_col=output_col))
        return self
    
    def build(self):
        """Finalize and return the pipeline model"""
        return Pipeline(stages=self.pipeline_stages)