#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "sparkSelectFeat"
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
class mlWorkLoads(attr.properties):
    """Main class for feature engineering pipeline"""

    def __init__(self, data=None, realm="CREATIVES"):

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
        
        self.data = data
        # self._column_registry = set(df.columns)
    
        __s_fn_id__ = f"{self.__name__} function {self.__class__.__name__}" #<__init__>"

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

    class ConstantColumnRemover(Transformer):
        """Remove columns with only one unique value (constants)"""
        @keyword_only
        def __init__(this, logger):
            super().__init__()
            this._logger = logger

            return None

        def _transform(
            this, 
            dataset:DataFrame=None, # must be a pysaprk dataframe
        ):
            """
            """

            __s_fn_id__ = f"class {this.__class__.__name__}" #<__init__>"

            to_drop = []   # list of columns to be droppped

            try:
                for col in dataset.columns:
                    # Get distinct values (optimized for Spark)
                    distinct_values = dataset.select(F.col(col)).distinct().limit(2).collect()
                    
                    if len(distinct_values) < 2:
                        to_drop.append(col)
                
                if not isinstance(to_drop, list) or len(to_drop)<=0:
                    this._logger.warning("%s No columns to drop in %s" % type(to_drop))
                # else:

                    # print(f"Columns before: {len(dataset.columns)}, after: {len(dataset.columns)-len(to_drop)}")


            except Exception as err:
                this._logger.error("%s %s \n",__s_fn_id__, err)
                this._logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)
                return None
            finally:
                this._logger.info("%s Dropping %d constant columns from original %d columns", 
                                  __s_fn_id__, len(to_drop), len(dataset.columns))
                this._logger.info("%s columns to drop: %s ", 
                                  __s_fn_id__, ", ".join(to_drop).upper())
                return dataset.drop(*to_drop)


    class NullColumnRemover(Transformer):
        """Remove columns with high null/zero percentage"""
        null_threshold = Param(Params._dummy(), "null_threshold", "Threshold for null percentage")
        zero_threshold = Param(Params._dummy(), "zero_threshold", "Threshold for zero percentage")
        
        @keyword_only
        def __init__(self, null_threshold=0.90, zero_threshold=0.90, logger=None):
            super().__init__()
            self.logger = logger  # Initialize logger directly, not as a Param
            self._setDefault(null_threshold=null_threshold, zero_threshold=zero_threshold)
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)
    
        @keyword_only
        def setParams(self, null_threshold=0.90, zero_threshold=0.90, logger=None):
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                self.logger = kwargs.pop('logger')
            self._set(**kwargs)
            return self
    
        def _transform(self, dataset):
            __s_fn_id__ = f"class {self.__class__.__name__}"
            to_drop = set()
            
            try:
                null_thresh = self.getOrDefault("null_threshold")
                zero_thresh = self.getOrDefault("zero_threshold")
                total_rows = dataset.count()
                
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
                    if col not in to_drop:
                        zero_count = dataset.filter(F.col(col) == 0).count()
                        zero_pct = zero_count / total_rows
                        if zero_pct > zero_thresh:
                            to_drop.add(col)
                
                if not to_drop and hasattr(self, 'logger') and self.logger is not None:
                    self.logger.warning("%s No columns to drop", __s_fn_id__)
                    
            except Exception as err:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)
                raise
    
            finally:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.info("%s Dropping %d Null columns > %0.2f threshold", 
                                  __s_fn_id__, 
                                  len([col for col in to_drop 
                                       if null_percentages.get(col, 0) > null_thresh]),
                                  null_thresh)
                    self.logger.info("%s Dropping %d Zeros columns > %0.2f threshold", 
                                  __s_fn_id__, 
                                  len([col for col in to_drop 
                                       if col in numeric_cols and null_percentages.get(col, 0) <= zero_thresh]),
                                  zero_thresh)
                    self.logger.info("%s Number of columns before: %d and after: %d",
                                  __s_fn_id__, len(dataset.columns), len(dataset.columns)-len(to_drop))
                return dataset.drop(*to_drop)        
    
    class VarianceThresholdRemover(Transformer):
        """Remove low variance columns with better numeric handling"""
        variance_threshold = Param(Params._dummy(), "variance_threshold", "Minimum variance threshold")
        
        @keyword_only
        def __init__(self, variance_threshold=0.1, logger=None):
            super().__init__()
            # kwargs = self._input_kwargs
            # self._set(**kwargs)
            self.logger = logger  # Initialize logger directly, not as a Param
            self._setDefault(variance_threshold=variance_threshold)
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)
    
        @keyword_only
        def setParams(self, null_threshold=0.90, zero_threshold=0.90, logger=None):
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                self.logger = kwargs.pop('logger')
            self._set(**kwargs)
            return self
    
        def _transform(self, dataset):
            __s_fn_id__ = f"class {self.__class__.__name__}"

            try:
                var_thresh = self.getOrDefault(self.variance_threshold)
                numeric_cols = [
                    f.name for f in dataset.schema.fields 
                    if isinstance(f.dataType, (NumericType, DoubleType, FloatType, IntegerType))
                ]

                if len(numeric_cols)>0:                
                    ''' Single-pass imputation and variance calculation '''
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
                    to_drop = []
                    to_drop = list(set(numeric_cols) - set(to_keep))
    
                    if len(to_drop)<=0 and hasattr(self, 'logger') and self.logger is not None:
                        self.logger.warning("%s Columns pass variance threshold > %0.2f, keeping: %s", 
                                            __s_fn_id__, var_thresh, ", ".join(to_keep).upper())
                else:
                    raise AttributeError("No numeric columns to apply variance thresholding, aborting")
    
            except Exception as err:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)
                raise

            finally:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.info("%s Dropping %d low-variance columns < %0.2f threshold", 
                                  __s_fn_id__, len(to_drop), var_thresh)
                    self.logger.info("%s Removed columns: %s", __s_fn_id__, ", ".join(to_drop).upper())
                    self.logger.info("%s Number of columns before: %d and after: %d",
                                  __s_fn_id__, len(dataset.columns), len(dataset.columns)-len(to_drop))
                return dataset.drop(*to_drop)
    
    
    class CorrelationFilter(Transformer):
        """Remove highly correlated features with priority column support"""
        threshold = Param(Params._dummy(), "threshold", "Correlation threshold")
        priority_cols = Param(Params._dummy(), "priority_cols", "Columns to preserve", 
                             typeConverter=TypeConverters.toListString)
    
        @keyword_only
        def __init__(self, threshold=0.9, priority_cols=None, logger=None):
            super().__init__()
            self.logger = logger  # Initialize logger directly, not as a Param
            priority_cols = [] if priority_cols is None else priority_cols
            self._setDefault(threshold=threshold, priority_cols=priority_cols)
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)
    
        @keyword_only
        def setParams(self, threshold=0.90, priority_cols=None, logger=None):
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            # Handle None for priority_cols in setParams
            if 'priority_cols' in kwargs and kwargs['priority_cols'] is None:
                kwargs['priority_cols'] = []
            if 'logger' in kwargs:
                self.logger = kwargs.pop('logger')
            self._set(**kwargs)
            return self
    
        def _transform(self, dataset):
            __s_fn_id__ = f"class {self.__class__.__name__}"

            try:
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

                if not to_drop and hasattr(self, 'logger') and self.logger is not None:
                    self.logger.warning("%s No columns to drop - all meet variance threshold", 
                                        __s_fn_id__, corr_thresh)

            except Exception as err:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)
                raise

            finally:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.info("%s Dropping %d highly correlated columns > %0.2f threshold", 
                                  __s_fn_id__, len(to_drop), corr_thresh)
                    self.logger.info("%s Removed columns: %s", __s_fn_id__, 
                                     ", ".join(sorted(to_drop)).upper())
                    self.logger.info("%s Number of columns before: %d and after: %d",
                                  __s_fn_id__, len(dataset.columns), len(dataset.columns)-len(to_drop))
                # return dataset.drop(*to_drop)
                return dataset.drop("temp_corr_features")


    class StandardDeviationFilter(Transformer):
        """
        Removes rows where any specified column (or all numeric columns) 
        exceeds `n_std` standard deviations from the mean.
        
        Args:
            df: PySpark DataFrame.
            columns: List of column names (optional). If None, uses all numeric columns.
            n_std: Number of standard deviations (default=3).
        
        Returns:
            Filtered DataFrame.
        """        
        n_std = Param(Params._dummy(), "n_std", "number of standard deviations to exclude data")
        priority_cols = Param(Params._dummy(), "priority_cols", "Columns to preserve", 
                             typeConverter=TypeConverters.toListString)
    
        @keyword_only
        def __init__(self, n_std=3, priority_cols=None, logger=None):
            super().__init__()
            self.logger = logger  # Initialize logger directly, not as a Param
            priority_cols = [] if priority_cols is None else priority_cols
            self._setDefault(n_std=n_std, priority_cols=priority_cols)
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)
    
        @keyword_only
        def setParams(self, n_std=3, priority_cols=None, logger=None):
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            # Handle None for priority_cols in setParams
            if 'priority_cols' in kwargs and kwargs['priority_cols'] is None:
                kwargs['priority_cols'] = []
            if 'logger' in kwargs:
                self.logger = kwargs.pop('logger')
            self._set(**kwargs)
            return self
    
        def _transform(self, dataset):
            __s_fn_id__ = f"class {self.__class__.__name__} transform"

            try:
                n_std = self.getOrDefault(self.n_std)
                priority_cols = self.getOrDefault(self.priority_cols)
                
                # Use all numeric columns if none specified
                if priority_cols is None or len(priority_cols)<=0:
                    priority_cols = [
                        f.name for f in dataset.schema.fields 
                        if isinstance(f.dataType,
                                        (NumericType, DoubleType, FloatType, IntegerType))
                    ]
                    self.logger.warning('%s Empty priority columns replaced with all numeric cols:%s',
                                        __s_fn_id__, ", ".join(priority_cols))
                    # columns = numeric_cols
                
                # Compute mean and std for each column
                agg_exprs = []
                for col_name in priority_cols:
                    agg_exprs.extend([F.mean(F.col(col_name)), F.stddev(F.col(col_name))])
                
                stats = dataset.select(*agg_exprs).collect()[0]
                
                # Build filter conditions
                conditions = []
                for i, col_name in enumerate(priority_cols):
                    mean_val = stats[2*i]
                    std_val = stats[2*i + 1]
                    
                    if std_val is not None:  # Skip if std=0 or null
                        lower = mean_val - n_std * std_val
                        upper = mean_val + n_std * std_val
                        conditions.append((F.col(col_name) >= lower) & (F.col(col_name) <= upper))
                
                # Combine conditions with AND
                if conditions:
                    dataset=dataset.filter(*conditions)
                # else:
                #     return df  # No filtering if no valid conditions
                if not isinstance(dataset, DataFrame) or dataset.count()<=0:
                    raise RuntimeError("Filter failed and returned %s" % type(dataset))

            except Exception as err:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)
                raise

            finally:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.info("%s Filter %s columns < %d std, returned %s with %d rows and %d columns", 
                                     __s_fn_id__, ", ".join(priority_cols), n_std, 
                                     type(dataset), dataset.count(), len(dataset.columns))
                return dataset


    class CleanTextColumns(Transformer):
        """
        Removes non-alphanumeric characters and collapses multiple spaces into single spaces
        for specified string columns.
        
        Args:
            priority_cols: List of column names to clean (optional). If None, uses all string columns.
            keep_spaces: Whether to keep spaces (default=True). If False, removes all whitespace.
        """        
        priority_cols = Param(Params._dummy(), "priority_cols", "priority columns to clean", 
                       typeConverter=TypeConverters.toListString)
        keep_spaces = Param(Params._dummy(), "keep_spaces", "Whether to keep spaces",
                           typeConverter=TypeConverters.toBoolean)
    
        @keyword_only
        def __init__(self, priority_cols=None, keep_spaces=True, logger=None):
            super().__init__()
            self.logger = logger  # Initialize logger directly, not as a Param
            priority_cols = [] if priority_cols is None else priority_cols
            self._setDefault(priority_cols=priority_cols, keep_spaces=keep_spaces)
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)
    
        @keyword_only
        def setParams(self, priority_cols=None, keep_spaces=True, logger=None):
            kwargs = self._input_kwargs
            # Handle None for columns in setParams
            if 'priority_cols' in kwargs and kwargs['priority_cols'] is None:
                kwargs['priority_cols'] = []
            if 'logger' in kwargs:
                self.logger = kwargs.pop('logger')
            self._set(**kwargs)
            return self
    
        def _transform(self, dataset):
            __s_fn_id__ = f"class {self.__class__.__name__} transform"
    
            try:
                priority_cols = self.getOrDefault(self.priority_cols)
                keep_spaces = self.getOrDefault(self.keep_spaces)
                
                # Use all string columns if none specified
                if not priority_cols:
                    priority_cols = [
                        f.name for f in dataset.schema.fields 
                        if isinstance(f.dataType, StringType)
                    ]
                    if self.logger:
                        self.logger.warning('%s Empty columns replaced with all string cols: %s',
                                            __s_fn_id__, ", ".join(priority_cols))
                
                if not priority_cols:
                    if self.logger:
                        self.logger.warning('%s No string columns found to clean', __s_fn_id__)
                    return dataset
                
                # Define the cleaning function based on keep_spaces parameter
                if keep_spaces:
                    # Keep alphanumeric and single spaces
                    regex_pattern = r'[^a-zA-Z0-9 ]'
                    replacement = ''
                else:
                    # Remove all non-alphanumeric (including spaces)
                    regex_pattern = r'[^a-zA-Z0-9]'
                    replacement = ''
                
                # Apply cleaning to each column
                for col_name in priority_cols:
                    if col_name in dataset.columns:
                        # First remove non-alphanumeric (and optionally spaces)
                        cleaned_col = F.regexp_replace(F.col(col_name), regex_pattern, replacement)
                        # Then collapse multiple spaces to single space (if keeping spaces)
                        if keep_spaces:
                            cleaned_col = F.regexp_replace(cleaned_col, r' +', ' ')
                        dataset = dataset.withColumn(col_name, cleaned_col)
                    elif self.logger:
                        self.logger.warning('%s Column %s not found in dataset', __s_fn_id__, col_name)
                
                if not isinstance(dataset, DataFrame):
                    raise RuntimeError("Transformation failed and returned %s" % type(dataset))
    
            except Exception as err:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)
                raise
    
            finally:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.info("%s Cleaned columns: %s, keep_spaces=%s, returned %s with %d rows", 
                                     __s_fn_id__, ", ".join(priority_cols), keep_spaces,
                                     type(dataset), dataset.count())
                return dataset

    class NullHandler(Transformer):
        """Handle null values in numeric and categorical columns"""
        numeric_strategy = Param(Params._dummy(), "numeric_strategy", "Imputation strategy for numerics")
        categorical_strategy = Param(Params._dummy(), "categorical_strategy", "Imputation strategy for categoricals")
        
        @keyword_only
        def __init__(self, numeric_strategy="median", categorical_strategy="mode", logger=None):
            super().__init__()
            self.logger = logger  # Initialize logger directly, not as a Param
            self._setDefault(numeric_strategy=numeric_strategy, categorical_strategy=categorical_strategy)
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)
        
        @keyword_only
        def setParams(self, numeric_strategy="median", categorical_strategy="mode", logger=None):
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            # Handle None for priority_cols in setParams
            # if 'priority_cols' in kwargs and kwargs['priority_cols'] is None:
            #     kwargs['priority_cols'] = []
            if 'logger' in kwargs:
                self.logger = kwargs.pop('logger')
            self._set(**kwargs)
            return self

        def _transform(self, dataset):
            __s_fn_id__ = f"class {self.__class__.__name__}"

            try:
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

                if categorical_strat.lower() =='mode':
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
                                    dataset = dataset.fillna({col: "UNKNOWN"})
                            else:
                                dataset = dataset.fillna({col: "UNKNOWN"})
                        else:
                            dataset = dataset.fillna({col: "UNKNOWN"})
                else:
                    raise KeyError('Invalid categorical strategy %s' % categorical_strat)
                # if isinstance(dataset, DataFrame) and hasattr(self, 'logger') and self.logger is not None:
                #     self.logger.warning("%s No columns to drop - all meet variance threshold", 
                #                         __s_fn_id__, var_thresh)

            except Exception as err:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)
                raise

            finally:
                if hasattr(self, 'logger') and self.logger is not None:
                    # self.logger.info("completed %s", self.__class__.__name__)
                    self.logger.info("%s Completed impute on %d columns with %s numeric strategy", 
                                  __s_fn_id__, len(numeric_cols), numeric_strat.upper())
                    self.logger.info("%s Completed impute on %d columns with %s categorical strategy", 
                                  __s_fn_id__, len(categorical_cols), categorical_strat.upper())
                    unknown_cols = [col for col in dataset.columns \
                                            if dataset.filter(F.col(col) == 'UNKNOWN').count() > 0]
                    self.logger.info("%s Filled %d columns with value UNKNOWN, columns: %s",__s_fn_id__, 
                                     len(unknown_cols), ", ".join(unknown_cols).upper())
                return dataset

    class CategoricalEncoder(Transformer):
        """Encode categorical variables"""

        exclude_cols = Param(Params._dummy(), "exclude_cols", "columns to exclude with encoding",
                         typeConverter=TypeConverters.toListString)

        @keyword_only
        def __init__(self, exclude_cols=None, logger=None):
            super().__init__()
            self.logger = logger  # Initialize logger directly, not as a Param
            self._setDefault(exclude_cols=exclude_cols)
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)
        
        @keyword_only
        def setParams(self, exclude_cols=None, logger=None):
            kwargs = self._input_kwargs
            # Handle None for exlude_cols in setParams
            if 'exclude_cols' in kwargs and kwargs['exclude_cols'] is None:
                kwargs['exclude_cols'] = []
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                self.logger = kwargs.pop('logger')
            self._set(**kwargs)
            return self

        def _transform(self, dataset):
            """
            """
            __s_fn_id__ = f"class {self.__class__.__name__}"

            try:
                exclude_cols = self.getOrDefault(self.exclude_cols)
                categorical_cols = [f.name for f in dataset.schema.fields \
                                        if str(f.dataType) == 'StringType()' \
                                            and f.name not in exclude_cols]
                
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
                dataset = pipeline.fit(dataset).transform(dataset)
                if not isinstance(dataset, DataFrame) or dataset.count()<=0:
                    raise RuntimeError("indexing and encoding pipeline retured empty %s" % type(dataset))
                # return pipeline.fit(dataset).transform(dataset)
            except Exception as err:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)
                raise

            finally:
                if hasattr(self, 'logger') and self.logger is not None:
                    # self.logger.info("completed %s", self.__class__.__name__)
                    self.logger.info("%s Completed indexing and encoding %d with OneHotEncoding", 
                                  __s_fn_id__, len(categorical_cols))
                    indexed_cols = [col for col in dataset.columns \
                                            if 'indexed'  in col.split("_")]
                    self.logger.info("%s Indexed %d columns: %s",__s_fn_id__, 
                                     len(indexed_cols), ", ".join(indexed_cols).upper())
                    encoded_cols = [col for col in dataset.columns \
                                            if 'encoded'  in col.split("_")]
                    self.logger.info("%s Encoded %d columns: %s",__s_fn_id__, 
                                     len(encoded_cols), ", ".join(encoded_cols).upper())
                return dataset
                # return pipeline.fit(dataset).transform(dataset)

    
    class FeatureScaler(Transformer):
        """Standardize numeric features"""
        
        output_col = Param(Params._dummy(), "output_col", "Output column name")
        exclude_cols = Param(Params._dummy(), "exclude_cols", "columns to exclude with encoding",
                         typeConverter=TypeConverters.toListString)

        @keyword_only
        def __init__(self, output_col="scaled_features", exclude_cols=None, logger=None):
            super().__init__()
            self.logger = logger  # Initialize logger directly, not as a Param
            self._setDefault(output_col=output_col, exclude_cols=exclude_cols)
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)
        
        @keyword_only
        def setParams(self, output_col="scaled_features", exclude_cols=None, logger=None):
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            # Handle None for priority_cols in setParams
            if 'exclude_cols' in kwargs and kwargs['exclude_cols'] is None:
                kwargs['exclude_cols'] = []
            if 'logger' in kwargs:
                self.logger = kwargs.pop('logger')
            self._set(**kwargs)
            return self

        def _transform(self, dataset):
            """
            """
            __s_fn_id__ = f"class {self.__class__.__name__}"

            try:
                output_col = self.getOrDefault(self.output_col)
                exclude_cols = self.getOrDefault(self.exclude_cols)
                # categorical_cols = [f.name for f in dataset.schema.fields \
                #                         if str(f.dataType) == 'StringType()' \
                #                             and f.name not in exclude_cols]
                
                numeric_cols = [f.name for f in dataset.schema.fields \
                                        if isinstance(f.dataType, NumericType) \
                                            and f.name not in exclude_cols]
                
                assembler = VectorAssembler(
                    inputCols=numeric_cols,
                    outputCol="features"
                )
                
                scaler = StandardScaler(
                    inputCol="features",
                    outputCol=output_col, #"scaled_features",
                    withStd=True,
                    withMean=True
                )
                
                pipeline=Pipeline(stages=[assembler, scaler])
                dataset =pipeline.fit(dataset).transform(dataset)
            
            except Exception as err:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)
                raise

            finally:
                if hasattr(self, 'logger') and self.logger is not None:
                    # self.logger.info("completed %s", self.__class__.__name__)
                    self.logger.info("%s Completed scaling %d numeric columns", 
                                  __s_fn_id__, len(numeric_cols))
                    self.logger.info("%s first vector has size: %d",
                                     __s_fn_id__, int(dataset.select("features").first()[0].size))
                return dataset


    # class TargetColumnFilter(Transformer):
    #     """
    #     Filters the dataset to include only specified priority columns plus the target column.
    #     If no priority columns are specified, keeps only the target column.
        
    #     Args:
    #         priority_cols: List of column names to keep (optional). If None, keeps only target column.
    #         target_col: Name of the target column (required).
    #     """        
    #     priority_cols = Param(Params._dummy(), "priority_cols", "priority columns to keep", 
    #                    typeConverter=TypeConverters.toListString)
    #     target_cols = Param(Params._dummy(), "target_cols", "target columns to always include",
    #                      typeConverter=TypeConverters.toListString)
    
    #     @keyword_only
    #     def __init__(self, priority_cols=None, target_cols=['target'], logger=None):
    #         super().__init__()
    #         self.logger = logger  # Initialize logger directly, not as a Param
    #         priority_cols = [] if priority_cols is None else priority_cols
    #         self._setDefault(priority_cols=priority_cols, target_cols=target_cols)
    #         kwargs = self._input_kwargs
    #         # Remove logger from kwargs before _set
    #         if 'logger' in kwargs:
    #             del kwargs['logger']
    #         self._set(**kwargs)
    
    #     @keyword_only
    #     def setParams(self, priority_cols=None, target_cols=['target'], logger=None):
    #         kwargs = self._input_kwargs
    #         # Handle None for columns in setParams
    #         if 'priority_cols' in kwargs and kwargs['priority_cols'] is None:
    #             kwargs['priority_cols'] = []
    #         if 'logger' in kwargs:
    #             self.logger = kwargs.pop('logger')
    #         self._set(**kwargs)
    #         return self
    
    #     def _transform(self, dataset):
    #         __s_fn_id__ = f"class {self.__class__.__name__} transform"
    
    #         try:
    #             priority_cols = self.getOrDefault(self.priority_cols)
    #             target_cols = self.getOrDefault(self.target_cols)
                
    #             if not target_col:
    #                 raise ValueError("target_col parameter must be specified")
                    
    #             # Combine priority columns with target column (remove duplicates)
    #             columns_to_keep = list(set(priority_cols + target_cols))
                
    #             # Verify all columns exist in dataset
    #             missing_cols = [col for col in columns_to_keep if col not in dataset.columns]
    #             if missing_cols:
    #                 raise ValueError(f"The following columns were not found in dataset: {', '.join(missing_cols)}")
                
    #             # Filter the dataset to only keep the specified columns
    #             dataset = dataset.select(*columns_to_keep).drop(*target_cols)
                
    #             if not isinstance(dataset, DataFrame):
    #                 raise RuntimeError("Transformation failed and returned %s" % type(dataset))
    
    #         except Exception as err:
    #             if hasattr(self, 'logger') and self.logger is not None:
    #                 self.logger.error("%s %s \n", __s_fn_id__, err)
    #                 self.logger.debug(traceback.format_exc())
    #             print("[Error]"+__s_fn_id__, err)
    #             raise
    
    #         finally:
    #             if hasattr(self, 'logger') and self.logger is not None:
    #                 self.logger.info("%s Kept columns: %s, returned %s with %d rows", 
    #                                  __s_fn_id__, ", ".join(columns_to_keep),
    #                                  type(dataset), dataset.count())
    #             return dataset
            
            
    class FeatureAssembler(Transformer):
        """Assemble final feature vector"""
        output_col = Param(Params._dummy(), "output_col", "Output column name")
        exclude_cols = Param(Params._dummy(), "exclude_cols", "columns to exclude with encoding",
                         typeConverter=TypeConverters.toListString)
        scaled_feature_col=Param(Params._dummy(), "scaled_feature_col", "Scalled feature column name")
        
        @keyword_only
        def __init__(self, 
                     output_col="features", 
                     exclude_cols = ['target'], 
                     scaled_feature_col="scaled_features",
                     logger=None):
            super().__init__()
            # kwargs = self._input_kwargs
            # self._set(**kwargs)
            self.logger = logger  # Initialize logger directly, not as a Param
            self._setDefault(output_col=output_col, 
                             exclude_cols=exclude_cols, 
                             scaled_feature_col=scaled_feature_col)
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            if 'logger' in kwargs:
                del kwargs['logger']
            self._set(**kwargs)
        
        @keyword_only
        def setParams(self, 
                      output_col="features", 
                      exclude_cols = ['target'],
                      scaled_feature_col="scaled_features",
                      logger=None):
            kwargs = self._input_kwargs
            # Remove logger from kwargs before _set
            # Handle None for priority_cols in setParams
            if 'exclude_cols' in kwargs and kwargs['exclude_cols'] is None:
                kwargs['exclude_cols'] = []
            if 'logger' in kwargs:
                self.logger = kwargs.pop('logger')
            self._set(**kwargs)
            return self

        def _transform(self, dataset):

            __s_fn_id__ = f"class {self.__class__.__name__}"

            try:
                exclude_cols = self.getOrDefault(self.exclude_cols)
                output_col = self.getOrDefault(self.output_col)
                scaled_feature_col=self.getOrDefault(self.scaled_feature_col)
                if scaled_feature_col:
                    numeric_cols = [scaled_feature_col]
                else:
                    numeric_cols = [f.name for f in dataset.schema.fields \
                                            if isinstance(f.dataType, NumericType) \
                                                and f.name not in exclude_cols]
                encoded_cols = [col for col in dataset.columns if col.endswith("_encoded")]
                
                assembler = VectorAssembler(
                    inputCols=numeric_cols + encoded_cols,
                    outputCol=output_col
                )
    
                dataset = assembler.transform(dataset)

            except Exception as err:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.error("%s %s \n", __s_fn_id__, err)
                    self.logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)
                raise

            finally:
                if hasattr(self, 'logger') and self.logger is not None:
                    # self.logger.info("completed %s", self.__class__.__name__)
                    self.logger.info("%s Completed assembling %d numeric columns and %d encoded columns", 
                                  __s_fn_id__, len(numeric_cols), len(encoded_cols))
                    self.logger.info("%s vector stored in %s with first vector size: %d",
                                     __s_fn_id__, output_col.upper(), 
                                     dataset.select(output_col).first()[0].size)
                return dataset
            # return assembler.transform(dataset)
    
    # Builder methods
    def remove_constant_columns(self):
        self.stages.append(self.ConstantColumnRemover(logger=self._logger))
        return self

    def remove_high_null_columns(self, null_threshold=0.7, zero_threshold=0.95):
        self.stages.append(
            self.NullColumnRemover(
                null_threshold=null_threshold, 
                zero_threshold=zero_threshold, 
                logger=self._logger  # Changed from self._logger to self.logger
            )
        )
        return self

    def remove_low_variance_columns(self, threshold=0.1):
        self.stages.append(
            self.VarianceThresholdRemover(
                variance_threshold=threshold,
                logger=self._logger,
            ))
        return self

    def add_correlation_filter(self, threshold=0.9):
        self.stages.append(
            self.CorrelationFilter(
                threshold=threshold,
                logger=self._logger,
            ))
        return self
        
    def remove_above_stdv_filter(self, n_std=3, priority_cols=None):
        self.stages.append(
            self.StandardDeviationFilter(
                priority_cols=priority_cols,
                n_std=n_std,
                logger=self._logger,
            ))
        return self
    
    def add_clean_text_columns(self, keep_spaces=True, priority_cols=None):
        self.stages.append(
            self.CleanTextColumns(
                priority_cols=priority_cols,
                keep_spaces=True,
                logger=self._logger,
            ))
        return self
    
    def add_null_handling(self, numeric_strategy="median", categorical_strategy="mode"):
        self.stages.append(self.NullHandler(
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            logger=self._logger,
        ))
        return self
    
    def add_categorical_encoding(self, exclude_cols=['target']):
        self.stages.append(
            self.CategoricalEncoder(exclude_cols=exclude_cols, logger=self._logger,))
        return self
    
    def add_feature_scaling(self, output_col="scaled_features", exclude_cols=['target']):
        self.stages.append(
            self.FeatureScaler(output_col=output_col, 
                               exclude_cols=exclude_cols, 
                               logger=self._logger,))
        return self
    
    def add_feature_assembly(self, 
                             output_col="features", 
                             exclude_cols=['target'],
                             scaled_feature_col="scaled_features"):
        self.stages.append(self.FeatureAssembler(
            output_col=output_col,
            exclude_cols=exclude_cols,
            scaled_feature_col=scaled_feature_col,
            logger=self._logger,))
        return self
    
    def build(self):
        """Finalize and return the pipeline model"""
        return Pipeline(stages=self.stages)

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