#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "sparkFeatSelect"
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
    from pyspark.sql.types import NumericType, IntegerType, DoubleType, FloatType
    from pyspark.sql import DataFrame
    from pyspark.ml.feature import Imputer, VarianceThresholdSelector, VectorAssembler, StandardScaler
    from pyspark.ml.feature import UnivariateFeatureSelector, OneHotEncoder, StringIndexer
    from pyspark.ml.stat import Correlation
    from pyspark.ml import Pipeline
    from typing import List, Iterable, Dict, Tuple
    # from psycopg2 import connect, DatabaseError
    # from psycopg2.extras import execute_values

    # from google.cloud import bigquery

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
class dataWorkLoads(attr.properties):

    def __init__(
        self, 
        data : None,
        desc : str="spark workloads", # identifier for the instances
        **kwargs, # unused
    ):
        """
        Description:
            Initializes the dataWorkLoads: class property attributes, app configurations, 
                logger function, data store directory paths, and global classes
        Attributes:
            desc (str) to change the instance description for identification
        Returns:
            None
        """

        ''' instantiate property attributes '''
        super().__init__(
#             desc=self.__desc__,
        )

        self.__name__ = __name__
        self.__package__ = __package__
        self.__module__ = __module__
        self.__app__ = __app__
        self.__ini_fname__ = __ini_fname__
        self.__conf_fname__ = __conf_fname__
        self.__desc__ = desc

        __s_fn_id__ = f"{self.__name__} function <__init__>"

        ''' default values '''
        # self._data = None
        self._stages = []
        # self._column_registry = set()
        self._features = []
       
        ''' initiate to load app.cfg data '''
        global logger
        global pkgConf
        global appConf

        try:
            self.cwd=os.path.dirname(__file__)
            pkgConf = configparser.ConfigParser()
            pkgConf.read(os.path.join(self.cwd,__ini_fname__))

            self.projHome = pkgConf.get("CWDS","PROJECT")
            sys.path.insert(1,self.projHome)

            ''' innitialize the logger '''
            from dongcha.utils import Logger as logs
            logger = logs.get_logger(
                cwd=self.projHome,
                app=self.__app__, 
                module=self.__module__,
                package=self.__package__,
                ini_file=self.__ini_fname__)
            ''' set a new logger section '''
            logger.info('########################################################')
            logger.info("%s %s",self.__name__,self.__package__)

            ''' Set the utils root directory '''
            self.pckgDir = pkgConf.get("CWDS",self.__package__)
            self.appDir = pkgConf.get("CWDS",self.__app__)
            # ''' get the path to the input and output data '''
            # self.dataDir = pkgConf.get("CWDS","DATA")

            appConf = configparser.ConfigParser()
            appConf.read(os.path.join(self.appDir, self.__conf_fname__))
            if data is None:
                raise ValueError("data attribute must be non-empty")
            self._data = data
            self._column_registry = set(self._data.columns)
            self._kept_features = []
            
            _done_str = f"{self.__name__} initialization for {self.__module__} module package "
            _done_str+= f"{self.__package__} in {self.__app__} done.\nStart workloads: {self.__desc__}."
            logger.debug("%s",_done_str)

            print(_done_str)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None


    ''' Function: BUILD PIPELINE

            author: <samana.thetha@gmail.com>
    '''
    def build(self):
        """Finalizes and returns the pipeline model"""
        return Pipeline(stages=self._stages).fit(self._data)


    ''' Function: REMOVE HIGH NULL COLUMNS

            author: <samana.thetha@gmail.com>
    '''
    def remove_high_null_columns(self, null_threshold=0.7):
        """
        Removes columns where null percentage exceeds threshold
        Args:
            null_threshold: Columns with > this % nulls will be dropped
        """
        __s_fn_id__ = f"{dataWorkLoads.__name__} function <remove_high_null_columns>"

        try:
            total_rows = self._data.count()
            null_percentages = {}
            
            for col_name in self._data.columns:
                null_count = self._data.filter(F.col(col_name).isNull()).count()
                null_percent = null_count / total_rows
                null_percentages[col_name] = null_percent
            columns_to_drop = []
            columns_to_drop = [col for col, percent in null_percentages.items() 
                              if percent > null_threshold]
            
            if len(columns_to_drop)<=0:
                logger.warning("%s No columns to drop, all %d columns are > %0.2f null threshold", 
                               __s_fn_id__, len(self._data.columns), null_threshold)
            else:
                self._data = self._data.drop(*columns_to_drop)
                self.column_registry = set(self._data.columns)
                logger.debug("%s dropped %d columns > %0.2f null threshold: %s", 
                             __s_fn_id__, len(columns_to_drop), 
                             null_threshold, ", ".join(columns_to_drop))
        
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            return None

        finally:
            return self._data


    ''' Function: REMOVE LOW VARIANCE COLUMNS

            author: <samana.thetha@gmail.com>
    '''
    def remove_low_variance_columns(self, variance_threshold=0.1):
        """
        Removes columns with variance below threshold (including zero variance)
        Args:
            variance_threshold: Minimum variance required to keep a column
        """
        
        __s_fn_id__ = f"{dataWorkLoads.__name__} function <remove_low_variance_columns>"

        try:
            numeric_cols=[]
            numeric_cols = [f.name for f in self._data.schema.fields 
                           if isinstance(f.dataType, NumericType)]
            if len(numeric_cols)<=0:
                logger.warning("%s No NumericType columns to process", __s_fn_id__)
                return self
            
            # Create temp columns with nulls handled
            temp_cols = []
            for col in numeric_cols:
                temp_col = self._get_unique_colname(f"{col}_temp")
                self._data = self._data.withColumn(
                    temp_col,
                    F.when(F.col(col).isNull(), 0).otherwise(F.col(col))
                )
                temp_cols.append(temp_col)
                self._register_column(temp_col)
            
            # Assemble with null protection
            assembler = VectorAssembler(
                inputCols=temp_cols,
                outputCol="temp_features",
                handleInvalid="keep"  # Critical for handling remaining edge cases
            )
            assembled_df = assembler.transform(self._data)
            
            # Apply variance threshold
            selector = VarianceThresholdSelector(
                varianceThreshold=variance_threshold,
                featuresCol="temp_features",
                outputCol="filtered_features"
            )
            
            model = selector.fit(assembled_df)
            selected_indices = model.selectedFeatures
            kept_columns = [numeric_cols[i] for i in selected_indices]
            
            # Clean up and update
            columns_to_drop = set(numeric_cols) - set(kept_columns)
            if columns_to_drop:
                # print(f"Dropping low-variance columns: {columns_to_drop}")
                self._data = self._data.drop(*columns_to_drop)
                self._column_registry = set(self._data.columns)
                self._kept_features.extend(kept_columns)
            
            # Remove temporary columns
            self._data = self._data.drop(*temp_cols)
            logger.debug("%s dropped %d low-variance columns with < %0.2f variance: %s", 
                         __s_fn_id__, len(columns_to_drop), 
                         variance_threshold, ", ".join(columns_to_drop))
        
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            return None

        finally:
            return self

    ''' Function: ADD CORRELATION FILTER

            author: <samana.thetha@gmail.com>
    '''
    def add_correlation_filter(self, threshold=0.9):
        """
        Removes highly correlated numeric features using pure PySpark
        Args:
            threshold: Correlation threshold above which features will be dropped
        """
        
        __s_fn_id__ = f"{dataWorkLoads.__name__} function <add_correlation_filter>"

        try:
            # Get numeric features
            numeric_cols = []
            numeric_cols = [col for col in self._data.columns 
                           if isinstance(self._data.schema[col].dataType, NumericType) and
                           not col.endswith("_indexed") and 
                           not col.endswith("_encoded")]
            if not isinstance(numeric_cols, list) or len(numeric_cols)<=0:
                logger.warning("% No numeric columns found, skipping process", __s_fn_id__)
                return self
            logger.debug("%s got %d numeric columns", __s_fn_id__, len(numeric_cols))
        
            # Compute correlation matrix with null handling
            temp_col = self._get_unique_colname("corr_features")
            assembler = VectorAssembler(
                inputCols=numeric_cols,
                outputCol=temp_col,
                handleInvalid="keep"  # Critical for handling any remaining nulls
            )
            
            assembled_sdf = assembler.transform(self._data)
            corr_matrix = Correlation.corr(assembled_sdf, temp_col).collect()[0][0]
            corr_array = corr_matrix.toArray()
            
            # Identify correlations using pure PySpark
            to_drop = set()
            n_cols = len(numeric_cols)
            
            for i in range(n_cols):
                if numeric_cols[i] in to_drop:
                    continue  # Skip already marked columns
                    
                for j in range(i+1, n_cols):
                    if abs(corr_array[i][j]) > threshold:
                        to_drop.add(numeric_cols[j])  # Always drop the latter column
            
            # Remove correlated columns
            if isinstance(to_drop, set) and len(to_drop)>0:
                # print(f"Dropping correlated features (r > {threshold}): {sorted(to_drop)}")
                self._data = self._data.drop(*to_drop)
                self.column_registry = set(self._data.columns)
                logger.info("%s dropped %d correlated features (r > %0.2f): %s:", 
                            __s_fn_id__, len(to_drop), threshold, sorted(to_drop))

            # Clean up temporary column
            if temp_col in self._data.columns:
                self._data = self._data.drop(temp_col)
        
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            return None

        finally:
            return self


    ''' Function: ADD NULL HANDLING

            author: <samana.thetha@gmail.com>
    '''
    def add_null_handling(
        self, 
        numeric_cols = None,
        num_impute_strategy=None,
        categorical_cols = None,
        categorical_fillna=None,
    ):
        """Handles missing values for both numeric and categorical columns"""
        
        __s_fn_id__ = f"{dataWorkLoads.__name__} function <add_null_handling>"

        __def_num_imp_strat__ = "median"
        __def_cat_NaN_fillval__="missing"

        try:
            if numeric_cols is None or len(numeric_cols)<=0:
                numeric_cols = [f.name for f in self._data.schema.fields 
                               if isinstance(f.dataType, (IntegerType, DoubleType, FloatType))]
                logger.warning("%s %d numeric columns set as detfault: %s", 
                               __s_fn_id__, len(numeric_cols), ", ".join(numeric_cols))
            if num_impute_strategy is None:
                num_impute_strategy = __def_num_imp_strat__
                logger.warning("%s undefined numeric impute strategy set as detfault %s", 
                               __s_fn_id__, num_impute_strategy.upper())
            # Numeric imputation
            imputed_cols = [self._get_unique_colname(f"{col}_imputed") for col in numeric_cols]
            for col in imputed_cols:
                self._register_column(col)
            
            self._stages.append(
                Imputer(inputCols=numeric_cols, outputCols=imputed_cols, strategy=num_impute_strategy)
            )
            
            # Categorical imputation (fill with 'missing')
            if categorical_fillna is None or "".join(categorical_fillna.split())=="":
                categorical_fillna=__def_cat_NaN_fillval__
                logger.warning("%s undefined categorical_fillna set as detfault to %s", 
                               __s_fn_id__, categorical_fillna.upper())
            # if categorical_cols:
                # from pyspark.sql.functions import when, col, lit
            if not isinstance(categorical_cols,list) or len(categorical_cols)<=0:
                categorical_cols = [f.name for f in self._data.schema.fields if str(f.dataType) == 'StringType()']
                logger.warning("%s %d categorical columns set as detfault: %s", 
                               __s_fn_id__, len(categorical_cols), ", ".join(categorical_cols))

            for cat_col in categorical_cols:
                self._data = self._data.withColumn(cat_col,
                                                   F.when(F.col(cat_col).isNull(),
                                                          F.lit("missing")).otherwise(F.col(cat_col))
                                                   )

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            return None

        finally:
            return self

    ''' Function: ADD CATEGORICAL ENCODING

            author: <samana.thetha@gmail.com>
    '''
    def add_categorical_encoding(
        self, 
        categorical_cols):
        """Converts categorical columns to numeric indices"""

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <add_categorical_encoding>"

        indexers = []
        encoders = []

        try:
            if not isinstance(categorical_cols,list) or len(categorical_cols)<=0:
                categorical_cols = [f.name for f in self._data.schema.fields if str(f.dataType) == 'StringType()']
                logger.warning("%s %d categorical columns set as detfault: %s", 
                               __s_fn_id__, len(categorical_cols), ", ".join(categorical_cols))
            for col in categorical_cols:
                indexed_col = self._get_unique_colname(f"{col}_indexed")
                encoded_col = self._get_unique_colname(f"{col}_encoded")
                logger.debug("%s set %d indexed columns and %d encoided columns", 
                             __s_fn_id__, len(indexed_col), len(encoded_col))
                
                self._register_column(indexed_col)
                self._register_column(encoded_col)
                logger.debug("%s %d registered columns", 
                             __s_fn_id__, len(list(self._column_registry)))
                
                indexers.append(
                    StringIndexer(inputCol=col, outputCol=indexed_col, handleInvalid="keep")
                )
                encoders.append(
                    OneHotEncoder(inputCol=indexed_col, outputCol=encoded_col)
                )
            
            self._stages.extend(indexers)
            self._stages.extend(encoders)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            return None

        finally:
            return self
    
    ''' Function: ADD FEATURE SCALING

            author: <samana.thetha@gmail.com>
    '''
    def add_feature_scaling(self, input_col):
        """ """
        __s_fn_id__ = f"{self.__name__} function <add_feature_scaling>"

        try:
            # from pyspark.ml.feature import StandardScaler
            output_col = self._get_unique_colname(f"{input_col}_scaled")
            self._register_column(output_col)
            logger.debug("%s registered %d columns in registry, now has %d columns", 
                         __s_fn_id__, len(output_col), len(list(self._column_registry)))
            self._stages.append(StandardScaler(inputCol=input_col, outputCol=output_col))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            return None

        finally:
            return self
    

    ''' Function: ADD FEATURE ASSEMBLY

            author: <samana.thetha@gmail.com>
    '''
    def add_feature_assembly(self, feature_cols, output_col="features"):
        """Combines features into a single vector column"""
        output_col = self._get_unique_colname(output_col)
        self._register_column(output_col)
        
        self._stages.append(
            VectorAssembler(inputCols=feature_cols, outputCol=output_col)
        )
        return self
    
    ''' Function: REGISTER COLUMNS

            author: <samana.thetha@gmail.com>
    '''
    def _register_column(self, col_name):
        """Ensures unique column names across all transformations"""
        if col_name in self._column_registry:
            raise ValueError(f"Column {col_name} already exists")
        self._column_registry.add(col_name)
        return col_name
    
    ''' Function: GET UNIQUE NAME

            author: <samana.thetha@gmail.com>
    '''
    def _get_unique_colname(self, base_name):
        """Generates unique column names automatically"""
        counter = 1
        new_name = base_name
        while new_name in self._column_registry:
            new_name = f"{base_name}_{counter}"
            counter += 1
        return new_name


    # ''' Function: DEFAULT FEATURE PIPELINE

    #         author: <samana.thetha@gmail.com>
    # '''
    # def default_feature_pipeline(self, sdf, target_col):
    #     """
    #     Descrption:

    #     """
    #     __s_fn_id__ = f"{self.__name__} function <default_feature_pipeline>"

    #     try:
    #         self.data = sdf
    #         self._column_registry = set(self._data.columns)
    #         if not isinstance(self._column_registry,set) or len(self._column_registry)<=0:
    #             raise AttributeError("Invalid set of columns %s" % type(self._column_registry))
    #         logger.debug("%s %d columns registered: %s" 
    #                      % (__s_fn_id__, len(list(self._column_registry)),", ".join(list(self._column_registry))))
    #         self._data = dataWorkLoads.null_handler(self._data)
    #         if self._data.count() <=0:
    #             raise ChildProcessError("null_handler returned %d rows" % self._data.count())
    #         logger.debug("%s null hander returned %d rows and %d columns", 
    #                      __s_fn_id__, self._data.count(), len(self._data.columns))
    #         # self._data = dataWorkLoads.categorical_encoder(self._data)
    #         # self._data, _ = dataWorkLoads.variance_filter(self._data)
    #         # self._data, _ = dataWorkLoads.correlation_filter(self._data)
    #         # self._data, selected = feature_selector(self._data, target_col)


    #     except Exception as err:
    #         logger.error("%s %s \n",__s_fn_id__, err)
    #         logger.debug(traceback.format_exc())
    #         print("[Error]"+__s_fn_id__, err)
    #         return None

    #     finally:
    #         return self._data, self._features

    
    # ''' Function: NULL HANDLER

    #         author: <samana.thetha@gmail.com>
    # '''
    # @staticmethod
    # def null_handler(sdf, numeric_strategy="median", categorical_strategy="mode"):
    #     """Handles null values for both numeric and categorical columns"""

    #     __s_fn_id__ = f"{dataWorkLoads.__name__} function <null_handler>"

    #     try:

    #         # Numeric imputation
    #         numeric_imputer = None
    #         if numeric_cols is None:
    #             numeric_cols = [f.name for f in self._data.schema.fields 
    #                            if isinstance(f.dataType, (IntegerType, DoubleType, FloatType))]
            
    #         # Numeric imputation
    #         imputed_cols = [self._get_unique_colname(f"{col}_imputed") for col in numeric_cols]
    #         for col in imputed_cols:
    #             self._register_column(col)
            
    #         self._stages.append(
    #             Imputer(inputCols=numeric_cols, outputCols=imputed_cols, strategy="median")
    #         )
    #         if not isinstance(numeric_cols,list) or len(numeric_cols)<=0:
    #             logger.warning("%s No numeric columns to impute, skipping step")
    #         else:
    #             numeric_imputer = Imputer(
    #                 inputCols=numeric_cols,
    #                 outputCols=[f"{c}_imputed" for c in numeric_cols],
    #                 strategy=numeric_strategy
    #             )
            
    #         # Categorical imputation (using StringIndexer + mode imputation)
    #         categorical_cols = [f.name for f in sdf.schema.fields if str(f.dataType) == 'StringType()']
            
    #         stages = [numeric_imputer]
    #         if categorical_cols:
    #             from pyspark.ml.feature import StringIndexer
    #             indexer = StringIndexer(
    #                 inputCols=categorical_cols,
    #                 outputCols=[f"{c}_indexed" for c in categorical_cols],
    #                 handleInvalid="keep"
    #             )
    #             stages.append(indexer)
                
    #             # For categorical mode imputation
    #             # from pyspark.sql.functions import mode
    #             mode_values = sdf.select([F.mode(F.col(c)).alias(c) for c in categorical_cols]).first()
    #             sdf = sdf.fillna(mode_values.asDict())
            
    #         # Create pipeline for null handling
    #         pipeline = Pipeline(stages=stages)

    #     except Exception as err:
    #         logger.error("%s %s \n",__s_fn_id__, err)
    #         logger.debug(traceback.format_exc())
    #         print("[Error]"+__s_fn_id__, err)
    #         return None

    #     finally:
    #         return pipeline.fit(sdf).transform(sdf)


    # ''' Function: CATEGORICAL ENCODER

    #         author: <samana.thetha@gmail.com>
    # '''
    # def categorical_encoder(sdf, max_categories=20):
    #     """Encodes categorical variables with optional cardinality reduction"""
    #     categorical_cols = [f.name for f in sdf.schema.fields if str(f.dataType) == 'StringType()']
        
    #     # Reduce high cardinality
    #     for col_name in categorical_cols:
    #         distinct_count = sdf.agg(F.countDistinct(F.col(col_name))).first()[0]
    #         if distinct_count > max_categories:
    #             top_categories = sdf.groupBy(col_name).count().orderBy("count", ascending=False).limit(max_categories)
    #             sdf = sdf.withColumn(
    #                 col_name,
    #                 F.when(F.col(col_name).isin([row[col_name] for row in top_categories.collect()]), F.col(col_name))
    #                 .otherwise("OTHER")
    #             )
        
    #     # StringIndexer + OneHotEncoder
    #     indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_indexed") for c in categorical_cols]
    #     encoders = [OneHotEncoder(inputCol=f"{c}_indexed", outputCol=f"{c}_encoded") for c in categorical_cols]
        
    #     pipeline = Pipeline(stages=indexers + encoders)
    #     return pipeline.fit(sdf).transform(sdf)


    # ''' Function: VARIANCE FILTER

    #         author: <samana.thetha@gmail.com>
    # '''
    # def variance_filter(sdf, threshold=0.1):
    #     """Removes low variance features"""
    #     numeric_cols = [c for c in sdf.columns if c.endswith("_imputed") or c.endswith("_encoded")]
        
    #     assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    #     selector = VarianceThresholdSelector(
    #         varianceThreshold=threshold,
    #         featuresCol="features",
    #         outputCol="filtered_features"
    #     )
        
    #     pipeline = Pipeline(stages=[assembler, selector])
    #     model = pipeline.fit(sdf)
        
    #     # Get kept features
    #     feature_names = model.stages[0].getInputCols()
    #     selected_indices = model.stages[1].getSelectedFeatures()
    #     kept_features = [feature_names[i] for i in selected_indices]
        
    #     return model.transform(sdf), kept_features


    # ''' Function: CORRELATION FILTER

    #         author: <nuwan.waidyanatha@rezgateway.com>
    # '''
    # def correlation_filter(sdf, features_col="filtered_features", threshold=0.9):
    #     """Removes highly correlated features"""
    #     corr_matrix = Correlation.corr(sdf, features_col).collect()[0][0]
    #     corr_array = corr_matrix.toArray()
        
    #     # Get feature names from metadata
    #     feature_names = sdf.schema[features_col].metadata["ml_attr"]["attrs"]["numeric"]
    #     feature_names = [f["name"] for f in feature_names]
        
    #     to_drop = set()
    #     for i in range(len(feature_names)):
    #         for j in range(i+1, len(feature_names)):
    #             if abs(corr_array[i][j]) > threshold:
    #                 to_drop.add(feature_names[j])
        
    #     # Create new feature vector without correlated features
    #     keep_indices = [i for i, name in enumerate(feature_names) if name not in to_drop]
        
    #     return sdf.withColumn(
    #         "selected_features",
    #         F.col(features_col).getItem(keep_indices)
    #     ), [name for i, name in enumerate(feature_names) if i in keep_indices]


