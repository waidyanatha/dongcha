#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "sparkFeatSelect2"
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
    from pyspark.ml.param import TypeConverters
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
    
    class ConstantColumnRemover(Transformer):
        """Remove columns with only one unique value (constants)"""
        @keyword_only
        def __init__(self):
            super().__init__()
        
        def _transform(self, dataset):
            to_drop = []
            
            for col in dataset.columns:
                # Get distinct values (optimized for Spark)
                distinct_values = dataset.select(F.col(col)).distinct().limit(2).collect()
                
                if len(distinct_values) < 2:
                    to_drop.append(col)
            
            if to_drop:
                print(f"\n[Dropping {len(to_drop)} constant columns]")
                print(f"Removed columns: {to_drop}")
                print(f"Columns before: {len(dataset.columns)}, after: {len(dataset.columns)-len(to_drop)}")
                return dataset.drop(*to_drop)
            
            print("\n[No constant columns found]")
            return dataset


    class NullColumnRemover(Transformer):
        """Remove columns with high null/zero percentage"""
        null_threshold = Param(Params._dummy(), "null_threshold", "Threshold for null percentage")
        zero_threshold = Param(Params._dummy(), "zero_threshold", "Threshold for zero percentage")
        
        @keyword_only
        def __init__(self, null_threshold=0.90, zero_threshold=0.90):
            super().__init__()
            self._setDefault(zero_threshold=0.95)  # Default 95% zero threshold
            kwargs = self._input_kwargs
            self._set(**kwargs)
        
        def _transform(self, dataset):
            null_thresh = self.getOrDefault(self.null_threshold)
            zero_thresh = self.getOrDefault(self.zero_threshold)
            total_rows = dataset.count()
            
            to_drop = set()
            
            # Check for nulls
            null_percentages = {
                col: dataset.filter(F.col(col).isNull()).count() / total_rows
                for col in dataset.columns
            }
            to_drop.update([col for col, pct in null_percentages.items() if pct > null_thresh])
            
            # Check for zeros in numeric columns
            numeric_cols = [
                f.name for f in dataset.schema.fields 
                if isinstance(f.dataType, (NumericType, DoubleType, FloatType, IntegerType))
            ]
            
            for col in numeric_cols:
                if col not in to_drop:  # Skip already marked columns
                    zero_count = dataset.filter(F.col(col) == 0).count()
                    zero_pct = zero_count / total_rows
                    if zero_pct > zero_thresh:
                        to_drop.add(col)
            
            if to_drop:
                print(f"\n[Dropping {len(to_drop)} problematic columns]")
                print(f"Null columns (> {null_thresh*100}% nulls): {[col for col in to_drop if null_percentages.get(col, 0) > null_thresh]}")
                print(f"Zero columns (> {zero_thresh*100}% zeros): {[col for col in to_drop if col in numeric_cols and null_percentages.get(col, 0) <= null_thresh]}")
                print(f"Columns before: {len(dataset.columns)}, after: {len(dataset.columns)-len(to_drop)}")
                return dataset.drop(*to_drop)
            
            print("\n[No columns dropped - all meet thresholds]")
            return dataset
        
    
    class VarianceThresholdRemover(Transformer):
        """Remove low variance columns with better numeric handling"""
        variance_threshold = Param(Params._dummy(), "variance_threshold", "Minimum variance threshold")
        
        @keyword_only
        def __init__(self, variance_threshold=0.1):
            super().__init__()
            kwargs = self._input_kwargs
            self._set(**kwargs)
        
        def _transform(self, dataset):
            var_thresh = self.getOrDefault(self.variance_threshold)
            numeric_cols = [
                f.name for f in dataset.schema.fields 
                if isinstance(f.dataType, (NumericType, DoubleType, FloatType, IntegerType))
            ]
            
            if len(numeric_cols)<=0:
                print("\n[No numeric columns for variance thresholding]")
                return dataset
                
            print(f"\n[Processing {len(numeric_cols)} numeric columns for variance thresholding]")
            
            # Single-pass imputation and variance calculation
            assembler = VectorAssembler(
                inputCols=numeric_cols,
                outputCol="temp_features",
                handleInvalid="keep"  # Automatically replaces nulls with mean
            )
            
            selector = VarianceThresholdSelector(
                varianceThreshold=var_thresh,
                featuresCol="temp_features",
                outputCol="filtered_features"
            )
            
            model = Pipeline(stages=[assembler, selector]).fit(dataset)
            selected = model.stages[-1].selectedFeatures
            to_keep = [numeric_cols[i] for i in selected]
            to_drop = list(set(numeric_cols) - set(to_keep))
            
            if to_drop:
                print(f"\n[Dropping {len(to_drop)} low-variance columns]")
                print(f"Removed columns: {to_drop}")
                print(f"Columns before: {len(dataset.columns)}, after: {len(dataset.columns)-len(to_drop)}")
                return dataset.drop(*to_drop)
            
            print("\n[No columns dropped - all meet variance threshold]")
            return dataset
    
    
    class CorrelationFilter(Transformer):
        """Remove highly correlated features with priority column support"""
        threshold = Param(Params._dummy(), "threshold", "Correlation threshold")
        priority_cols = Param(Params._dummy(), "priority_cols", "Columns to preserve", 
                             typeConverter=TypeConverters.toListString)
    
        @keyword_only
        def __init__(self, threshold=0.9, priority_cols=None):
            super().__init__()
            self._setDefault(priority_cols=[])
            kwargs = self._input_kwargs
            self._set(**kwargs)
    
        def _transform(self, dataset):
            corr_thresh = self.getOrDefault(self.threshold)
            priority_cols = self.getOrDefault(self.priority_cols)
            
            numeric_cols = [
                f.name for f in dataset.schema.fields 
                if isinstance(f.dataType, (NumericType, DoubleType, FloatType, IntegerType))
                and f.name not in priority_cols
            ]
            
            if len(numeric_cols) < 2:
                print("\n[Not enough numeric columns for correlation analysis]")
                return dataset
                
            print(f"\n[Checking correlation among {len(numeric_cols)} numeric columns]")
            
            assembler = VectorAssembler(
                inputCols=numeric_cols,
                outputCol="temp_corr_features",
                handleInvalid="keep"
            )
            
            corr_matrix = Correlation.corr(
                assembler.transform(dataset), 
                "temp_corr_features"
            ).collect()[0][0].toArray()
            
            col_stats = {
                col: dataset.select(F.countDistinct(col)).first()[0]
                for col in numeric_cols
            }
            
            to_drop = set()
            for i in range(len(numeric_cols)):
                if numeric_cols[i] in to_drop:
                    continue
                for j in range(i+1, len(numeric_cols)):
                    if (abs(corr_matrix[i][j]) > corr_thresh and 
                        numeric_cols[j] not in to_drop):
                        if col_stats[numeric_cols[i]] >= col_stats[numeric_cols[j]]:
                            to_drop.add(numeric_cols[j])
                        else:
                            to_drop.add(numeric_cols[i])
                            break
            
            if to_drop:
                print(f"\n[Dropping {len(to_drop)} highly correlated columns]")
                print(f"Removed columns: {sorted(to_drop)}")
                print(f"Columns before: {len(dataset.columns)}, after: {len(dataset.columns)-len(to_drop)}")
                result = dataset.drop(*to_drop)
                return result.drop("temp_corr_features")
            
            print("\n[No columns dropped - no high correlations found]")
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
    def remove_constant_columns(self):
        self.pipeline_stages.append(self.ConstantColumnRemover())
        return self

    def remove_high_null_columns(self, null_threshold=0.7, zero_threshold=0.95):
        self.pipeline_stages.append(
            self.NullColumnRemover(null_threshold=null_threshold, zero_threshold=zero_threshold))
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

    def exec_pipe_with_stages(self):
        """
        Executes pipeline step-by-step, ensuring each stage receives the 
        correctly transformed DataFrame from the previous stage.
        
        Returns:
            - intermediates (dict): {"stage_name": intermediate_df}
            - final_df (DataFrame): Final transformed output
        """
        current_df = self.df
        self.intermediate_results = {}
        
        for i, stage in enumerate(self.pipeline_stages):
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