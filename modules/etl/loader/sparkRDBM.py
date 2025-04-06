#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "sparkRDBM"
__package__ = "loader"
__module__ = "etl"
__app__ = "rezaware"
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
    from dotenv import load_dotenv
    load_dotenv()

    import findspark
    findspark.init()
    from pyspark.sql import functions as F
#     from pyspark.sql.functions import lit, current_timestamp,col,isnan, when, count, countDistinct
    from pyspark.ml.feature import Imputer
    from pyspark.sql import DataFrame
    from typing import List, Iterable, Dict, Tuple
    from psycopg2 import connect, DatabaseError
    from psycopg2.extras import execute_values

    # from google.cloud import bigquery

    from rezaware.modules.etl.loader import __propAttr__ as attr

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))

'''
    Class reads and writes data from and to RDBMS using apache pyspark sql functions
        Current working databases: 
        * postgresql

    Contributors:
        * nuwan.waidyanatha@rezgateway.com
        * samana.thetha@gmail.com
        * farmraider@protonmail.com

    Resources:
        * Notebooks
            * Upsert function evaluation use utils/notebooks/etl/load/sparkDBwls.ipynb
        * Installation guide
            * https://computingforgeeks.com/how-to-install-apache-spark-on-ubuntu-debian/
        * Acknowledgement
            * Many thanks to Santhanu's medium article and code snipets for generating the
                upsert_sdf_to_db function https://tinyurl.com/pyspark-batch-upsert
'''
class dataWorkLoads(attr.properties):

    def __init__(
        self, 
        desc : str="spark workloads", # identifier for the instances
        db_type : str = None, # database type one of self._dbTypeList
        db_driver:str = None,
        db_hostIP:str = None,
        db_port : str = None,
        db_name : str = None,
        db_schema:str = None,
        spark_partitions:int=None,
        spark_format:str = None,
        spark_save_mode:str=None,
        spark_jar_dir: str =None,
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
            realm="DATABASE",
            db_type=db_type,
            spark_format=spark_format,
        )

        self.__name__ = __name__
        self.__package__ = __package__
        self.__module__ = __module__
        self.__app__ = __app__
        self.__ini_fname__ = __ini_fname__
        self.__conf_fname__ = __conf_fname__
        self.__desc__ = desc

        __s_fn_id__ = f"{self.__name__} function <__init__>"

#         ''' instantiate property attributes '''
#         super().__init__(
#             desc=self.__desc__,
#             realm="DATABASE"
#         )

        ''' default values '''
        # self._dbTypeList=['postgresql', 'bigquery']
        # self._formatList=['jdbc','bigquery']
        ''' Initialize the DB parameters '''
        self._dbType = db_type.lower()
        self._dbDriver=db_driver
        self._dbHostIP=db_hostIP
        self._dbPort = db_port
        self._dbName = db_name
        self._dbSchema=db_schema
        self._partitions=spark_partitions
        self._jarDir = spark_jar_dir
        self._rwFormat = spark_format
        self._saveMode = spark_save_mode
#         self._session = None

#         ''' Initialize property var to hold the data '''
#         self._data = None

#         ''' instantiate property attributes '''
#         super().__init__(
#             desc=self.__desc__,
#             realm="NOSQLDB"
#         )
        
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
            from rezaware.utils import Logger as logs
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
            ''' get the path to the input and output data '''
            self.dataDir = pkgConf.get("CWDS","DATA")

            appConf = configparser.ConfigParser()
            appConf.read(os.path.join(self.appDir, self.__conf_fname__))

            logger.debug("%s initialization for %s module package %s %s done.\nStart workloads: %s."
                         %(self.__app__,
                           self.__module__,
                           self.__package__,
                           self.__name__,
                           self.__desc__))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None


    ''' Function

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    def read_data_from_table(
        self,
        select : str="",
        db_table :str="",
        db_column:str="",
        lower_bound=None,
        upper_bound=None,
        **options
    ) -> DataFrame:
        """
        Description:
            There is the option of speficing a valid SQL select statement with clauses to
            retrieve the data from one or more tables. The option is to provide table name
        Attributes:
            select (str)
            db_table (str)
        db_column:str="",
        lower_bound=None,
        upper_bound=None,
        **options) -> DataFrame:

        Returns:
        """

        __s_fn_id__ = f"{self.__name__} function <read_data_from_table>"
        load_sdf = None   # initiatlize return var
        option = {}

        try:
            
            if self.dbType.lower() == 'postgresql':
                ''' set the partitions '''
                if "PARTITIONS" in options.keys():
                    self.partitions = options['PARTITIONS']
                if "FORMAT" in options.keys():
                    self.rwFormat = options['FORMAT']
                if "url" not in options.keys():
                    options['url'] = self.dbConnURL
                if "numPartitions" not in options.keys():
                    options['numPartitions'] = self.partitions
                if "user" not in options.keys():
                    options['user'] = self.dbUser
                if "password" not in options.keys():
                    options['password'] = self.dbPswd
                if "driver" not in options.keys():
                    options['driver'] = self.dbDriver
                ''' use query else use partition column'''
                if select is not None and "".join(select.split())!="":
                    options['query'] = select
                elif db_table is not None and "".join(db_table.split())!="":
                    if db_table.find(self.dbSchema+".") == -1:
                        db_table = self.dbSchema+"."+db_table
                    options["dbtable"]=db_table
                else:
                    raise AttributeError("Invalid set of input variables necesssary "+\
                                         "to determine the read operation")
                load_sdf = self.session.read\
                    .format(self.rwFormat)\
                    .options(**options)\
                    .load()

            elif self.dbType.lower() == 'bigquery':
                ''' --- BIGQUERY --- '''
                ''' check for credentials and project id'''
                if not os.environ.get("GOOGLE_PROJECT_ID") or "".join(os.environ.get("GOOGLE_PROJECT_ID").split())==""\
                    or not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") \
                    or "".join(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS").split())=="":
                    raise OSError("Both GOOGLE_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS must be defined as ENV vars in .env")
                if "project" not in options.keys():
                    options['project'] = os.environ.get("GOOGLE_PROJECT_ID")
                ''' ensure select string is not null '''
                if select is not None and "".join(select.split())!="":
                    options['query'] = select
                    # load_sdf = self.session.read\
                    #     .format(self.rwFormat)\
                    #     .load(select)
                elif db_table is not None and "".join(db_table.split())!="":
                    if db_table.find(self.dbSchema+".") == -1:
                        db_table = self.dbSchema+"."+db_table
                    if db_table.find(options['project']+":") == -1:
                        db_table = options['project']+":"+db_table
                    options["table"]=db_table
                    # ''' read data into pandas and then convert to spark dataframe '''
                    # load_sdf = self.session.read\
                    #     .format(self.rwFormat)\
                    #     .options(**options)\
                    #     .load()
                else:
                    raise AttributeError("Invalid set of input variables necesssary "+\
                                         "to determine the read operation")
                ''' read data into pandas and then convert to spark dataframe '''
                load_sdf = self.session.read\
                    .format(self.rwFormat)\
                    .options(**options)\
                    .load()

            # elif self.dbType.lower() == 'bigquery_old':
            #     ''' --- BIGQUERY --- '''
            #     ''' check for credentials and project id'''
            #     if not os.environ.get("GOOGLE_PROJECT_ID") or "".join(os.environ.get("GOOGLE_PROJECT_ID").split())==""\
            #         or not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "".join(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS").split())=="":
            #         raise OSError("Both GOOGLE_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS must be defined as ENV vars in .env")
            #     # ''' check table string '''
            #     # if "table" in options.keys() and "".join(options['table'].split())!="":
            #     #     options['table']=options['table']
            #     # elif db_table is not None and "".join(db_table.split())!="":
            #     #     if db_table.find(self.dbSchema+".") == -1:
            #     #         option['table'] = ".".join([os.environ.get('GOOGLE_PROJECT_ID'), self.dbSchema, db_table])
            #     #     else:
            #     #         option['table'] = ".".join([os.environ.get('GOOGLE_PROJECT_ID'), db_table])
            #     # else:
            #     #     raise AttributeError("%s improperly defined table option" 
            #     #                          % self._dbType.upper())
            #     ''' ensure select string is not null '''
            #     if not select or "".join(select.split())=="":
            #         raise AttributeError("Undefined %s select string" % type(select))
            #     ''' read data into pandas and then convert to spark dataframe '''
            #     client = bigquery.Client(project=os.environ.get("GOOGLE_PROJECT_ID"))
            #     load_sdf = self.session.createDataFrame(client.query(select).to_dataframe())

            else:
                raise AttributeError("Something went wrong, unrecognized database type %s" 
                                      % self._dbType.upper())

            ''' Temporary commented until bigquery read is resolved '''
            # load_sdf = self.session.read\
            #     .format(self.rwFormat)\
            #     .options(**options)\
            #     .load()
            
            if load_sdf:
                logger.debug("%s loaded %d rows into pyspark dataframe", 
                             __s_fn_id__, load_sdf.count())
            else:
                logger.debug("%s Something went wrong", __s_fn_id__)

            self._data = load_sdf

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data

    ''' Function --- INSERT INTO TABLE ---

            author: <nuwan.waidyanatha@rezgateway.com>
    '''

    def insert_sdf_into_table(
        self,
        save_sdf,
        db_name : str=None,
        db_table: str=None,
        **kwargs):
        
        __s_fn_id__ = f"{self.__name__} function <insert_sdf_into_table>"
        _num_records_saved = 0

        try:
            self.data = save_sdf
#             if self.data.count() <= 0:
#                 raise AttributeError("No data to insert into database table %s"% db_table)
#             logger.debug("%s %s dtype data convereted to %s pysaprk dataframe with %d rows %d columns",
#                          __s_fn_id__,type(save_sdf),type(self._data),
#                          self._data.count(),len(self._data.columns))
            ''' optional db_name given then set property, else use default in setting  connection'''
            if isinstance(db_name,str) and "".join(db_name.split())!="":
                self.dbName=db_name
            ''' mandatory table name must be given '''
            if not isinstance(db_table,str) or "".join(db_table.split())=="":
                raise AttributeError("db_table cannot be empty %" % type(db_table))
            ''' TODO validate table exists '''
            
            if len(kwargs) > 0:
                self.session = kwargs
            else:
                self.session = {}
            
            ''' if created audit columns don't exist add them '''
            listColumns=self._data.columns
            if "created_dt" not in listColumns:
                self._data = self._data.withColumn("created_dt", F.current_timestamp())
            if "created_by" not in listColumns:
                self._data = self._data.withColumn("created_by", F.lit(self.dbUser))
            if "created_proc" not in listColumns:
                self._data = self._data.withColumn("created_proc", F.lit("Unknown"))
            
            ''' TODO: add code to accept options() to manage schema specific
                authentication and access to tables '''

#             if "saveMode" in kwargs.keys():
#                 self.sparkSaveMode = kwargs['saveMode']
#                 self.sparkSaveMode = kwargs['SAVEMODE']
                
#             logger.info("%s Wait a moment while we insert data int %s",__s_fn_id__,db_table)
            
            self._data.select(self.data.columns).\
                    write.format(self.rwFormat).\
                    mode(self.saveMode).\
                options(
                    url=self.dbConnURL,    # 'jdbc:postgresql://10.11.34.33:5432/Datascience', 
                    dbtable=self.dbSchema+"."+db_table,       # '_issuefix_bkdata.customerbookings',
                    user=self.dbUser,     # 'postgres',
                    password=self.dbPswd, # 'postgres',
                    driver=self.dbDriver).save(self.saveMode.lower())

            _num_records_saved = self._data.count()
            logger.debug("%s Saved %d  rows into table %s in database %s complete!",
                        __s_fn_id__,_num_records_saved,
                         ".".join([self.dbSchema,db_table]), self._dbName)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return _num_records_saved

    ''' - Function --- UPDATE TABLE ---

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    def upsert_with_dbtype(func):
        """
        Description:
            wrapper function to upsert dataframe based on self.dbType
        Attribute
            func inherits upsert_sdf_to_table
        returns upsert_dbtype_wrapper
        """
        
        @functools.wraps(func)
        def upsert_wrapper(self,save_sdf, db_name, db_table,unique_keys,**options):
            """
            Description:
                Applies the database type specific @static functions to upsert dataframe
            Attributes:
                same as upsert_sdf_to_table input attributes
            Returns:
                processed record count
            """
                
            __s_fn_id__ = f"{self.__name__} function <upsert_wrapper>"
            _num_records_saved = 0
            total_recs_loaded = None   # initiatlize return attr

            try:
                _data, _omitted_cols, _batch_size = func(
                    self, save_sdf, db_name, db_table, unique_keys, **options
                )

                if self.dbType == 'postgresql':
                    upsert_query = dataWorkLoads.build_pgsql_query(
                        cols=_data.schema.names,
                        table_name=self.dbSchema+"."+db_table,
                        unique_key=unique_keys,
                        cols_not_for_update=_omitted_cols,
                    )
#                     print(upsert_query)
                    _db_cred = {
                        "host": self.dbHostIP,
                        "database": self.dbName,
                        "user": self.dbUser,
                        "password": self.dbPswd,
                        "port": self.dbPort,
                    }
                    upsert_stats = _data.coalesce(self.partitions).rdd.mapPartitions(
                        lambda dataframe_partition: dataWorkLoads.batch_and_upsert(
                            dataframe_partition=dataframe_partition,
                            sql=upsert_query,
                            database_credentials=_db_cred,
                            batch_size=_batch_size,
                        )
                    )
                    ''' confirm upsert returned stats '''
                    if not isinstance(upsert_stats.collect(),list) \
                        or len(upsert_stats.collect())<=0:
                        raise RuntimeError("upsert failed, returned %s count object" 
                                           % type(upsert_stats))
                    logger.debug("%s summing %d upsert stats", 
                                 __s_fn_id__, len(upsert_stats.collect()))
                    total_recs_loaded = 0
                    for counter in upsert_stats.collect():
                        total_recs_loaded += counter

                else:
                    ''' invalid DB type '''
                    raise RuntimeError("TBD %s dbType upsert; only works for postgresql", self.dbType)

                # logger.info("%s Success saving %d rows of %d records in table %s in %s db complete!",
                #             __s_fn_id__, total_recs_loaded, _data.count(),
                #             self._dbSchema.upper()+"."+db_table.upper(), self._dbName.upper())

            except Exception as err:
                logger.error("%s %s \n",__s_fn_id__, err)
                logger.debug(traceback.format_exc())
                print("[Error]"+__s_fn_id__, err)
                return None

            finally:
                logger.info("%s Success saving %d rows of %d records in table %s in %s db complete!",
                            __s_fn_id__, total_recs_loaded, _data.count(),
                            self._dbSchema.upper()+"."+db_table.upper(), self._dbName.upper())
                return total_recs_loaded

        return upsert_wrapper

    @upsert_with_dbtype
    def upsert_sdf_to_table(
        self,
        save_sdf,       # any dtype data set (will be converted to Dataframe)
        db_name : str,  # optional name of the database; else use default
        db_table: str,  # name of table to be updated; excluding the schema name
        unique_keys:List[str], # list of columns to use in the where statement 
#         uspert_sql:str="",  # sql update statement
        **options) -> int:
        """
        Description:
            First option is to use the function to generate an update statement, for you,
            using the data columns and the table given; then apply the dynamically generated
            statement string to update the columns.
            Other option is to provid and an update sql statement that can be directly applied;
            e.g., UPDATE <tablename> SET 
            The options and and drivers will be set accordingly.
        Attributes:
            save_sdf (any) dtype that can be converted to a pyspark dataframe
            db_name (str) optional name of the database; else use default
            db_table (str) table name to update rows; excluding schema name
            uspert_sql (str) sql update statement to apply directly
        Returns"
            self._data (DataFrame) 
            cols_not_for_update_ (List) of column names to omit from update
            batch_size_ (int) number of rows to process in each batch
        """
        
        __s_fn_id__ = f"{self.__name__} function <upsert_sdf_to_table>"
        
        cols_not_for_update_=[]
        batch_size_ = 1000

        try:
#             print("Validating upsert attributes and parameters ...")
            self.data = save_sdf
            if self.data.count() <= 0:
                raise ValueError("No data to update in table")
            if len(options) > 0:
                self.session = options
            else:
                self.session = {}

            ''' TODO validate table exists '''
            
            ''' if created audit columns don't exist add them '''
            listColumns=self._data.columns
            if "modified_dt" not in listColumns:
                self._data = self._data.withColumn("modified_dt", F.current_timestamp())
            else:
                self._data = self._data.withColumn('modified_dt',
                                                   F.when(F.col('modified_dt').isNull(),
                                                          F.current_timestamp()
                                                         ).otherwise(F.col('modified_dt')))
            if "modified_by" not in listColumns:
                self._data = self._data.withColumn("modified_by", F.lit(self.dbUser))
            else:
                self._data = self._data.withColumn('modified_by',
                                                   F.when(F.col('modified_by').isNull(),
                                                          F.lit(self.dbUser)
                                                         ).otherwise(F.col('modified_by')))
            _s_mod_proc = "_".join([self.__app__,self.__module__,self.__package__,
                                    self.__name__,__s_fn_id__])
            if "modified_proc" not in listColumns:
                self._data = self._data.withColumn("modified_proc", F.lit(_s_mod_proc))
            else:
                self._data = self._data.withColumn('modified_proc',
                                                   F.when(F.col('modified_proc').isNull(),
                                                          F.lit(_s_mod_proc)
                                                         ).otherwise(F.col('modified_proc')))

            ''' TODO: add code to accept options() to manage schema specific
                authentication and access to tables '''

            ''' set all option parameters '''
            if "DBTYPE" in options.keys():
                self.dbType = options['DBTYPE']
            ''' optional db_name is given then set property, else use default '''
            if db_name is not None and "".join(db_name.split())!="":
                self.dbName=db_name
            if "PARTITIONS" in options.keys():
                self.partitions = options['PARTITIONS']
            if "OMITCOLS" in options.keys():
                cols_not_for_update_ = options['OMITCOLS']
            if "BATCHSIZE" in options.keys():
                batch_size_ = options['BATCHSIZE']
            if "numPartitions" not in options.keys():
                options['numPartitions'] = self.partitions
            if "user" not in options.keys():
                options['user'] = self.dbUser
            if "password" not in options.keys():
                options['password'] = self.dbPswd
#             if "driver" not in options.keys():
#                 options['driver'] = self.dbDriver

            logger.debug("%s Validation complete with %d rows to process, "+\
                         "%d columns to omit in update, and "+\
                         "applying %d batch size", 
                         __s_fn_id__,self._data.count(), len(cols_not_for_update_), batch_size_)
            print("Wait a moment, writing data to %s %s database ..." 
                  %(self.dbType, self.dbName))
            ''' use query else use partition column '''

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._data, cols_not_for_update_, batch_size_


    ''' - Functions --- SPARK POSTGRES BATCH UPSERT ---

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    @staticmethod
    def get_pgsql_conn(
        host: str,     # host name of database instance
        database: str, # name of the database to connect to.
        user: str,     # user name with permissions to access schema and tables.
        password: str, # password for authenticating the user name.
        port: str):    # postgres port to connect; e.g. 5432.
        """
        Description:
            makes us of psycopg2 connection to a postgres database.
        Attributes:
            host (str) host name of database instance.
            database (str) name of the database to connect to.
            user (str) user name with permissions to access schema and tables.
            password (str) password for authenticating the user name.
            port (str) postgres port to connect; e.g. 5432.
        Return (connect) database connection.
        """
        __s_fn_id__ = f"{dataWorkLoads.__name__} @staticmethod <upsert_sdf_to_table>"

        try:
            conn = connect(
                host=host, database=database,
                user=user, password=password,
                port=port
            )

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return conn

    @staticmethod
    def batch_and_upsert(dataframe_partition: Iterable,
                         sql: str,
                         database_credentials: dict,
                         batch_size: int = 1000):
        """
        Description:
            Batch the input dataframe_partition as per batch_size and upsert to postgres
            using psycopg2 execute values.
        Attributes:
            dataframe_partition (DataFrame/Iterable) Pyspark DataFrame partition or any iterable.
            sql (str): query to insert/upsert the spark dataframe partition to postgres.
            database_credentials (Dict) postgres database credentials.
            Example: database_credentials = {
                    host: <host>,
                    database: <database>,
                    user: <user>,
                    password: <password>,
                    port: <port>
                }
            batch_size (int) size of batch per round trip to database.
        Return (int) with total records processed.
        """

        __s_fn_id__ = f"{dataWorkLoads.__name__} @staticmethod <batch_and_upsert>"

        conn, cur = None, None
        counter = 0
        batch = []
        
        try:
            for record in dataframe_partition:

                counter += 1
                batch.append(list(record))

                if not conn:
                    conn = dataWorkLoads.get_pgsql_conn(**database_credentials)
                    cur = conn.cursor()
                    logger.warning("%s PSQL connection set with %s and connection %s",
                                   __s_fn_id__,type(cur),str(conn))

                if counter % batch_size == 0:
                    execute_values(
                        cur=cur, sql=sql,
                        argslist=batch,
                        page_size=batch_size
                    )
                    conn.commit()
                    logger.debug("%s executed and commited batch size %d counter %d",
                                 __s_fn_id__,batch_size,counter)
                    batch = []

            if batch:
                execute_values(
                    cur=cur, sql=sql,
                    argslist=batch,
                    page_size=batch_size
                )
                conn.commit()
                logger.debug("%s executed and commited batch size %d",
                             __s_fn_id__,batch_size)

            if cur:
                cur.close()
            if conn:
                conn.close()

            yield counter

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

            yield None

    @staticmethod
    def build_pgsql_query(
        cols: List[str],
        table_name: str,
        unique_key: List[str],
        cols_not_for_update: List[str] = None
    ) -> str:
        """
        Description:
            Builds postgres upsert query using input arguments.
            Example : build_pgsql_query(
                ['col1', 'col2', 'col3', 'col4'],
                "my_table",
                ['col1'],
                ['col2']
            ) ->
            INSERT INTO my_table (col1, col2, col3, col4) VALUES %s  
            ON CONFLICT (col1) DO UPDATE SET (col3, col4) = (EXCLUDED.col3, EXCLUDED.col4) ;
        Attributes:
            cols (List) the postgres table columns required in the insert part of the query.
            table_name (str) the postgres table name; must contain the schema name as a prefix.
            unique_key (List) unique_key of the postgres table for checking unique constraint
                violations.
            cols_not_for_update (List) columns in cols which are not required in the update
                part of upsert query.
        Returns (str) Upsert query as per input arguments.
        """

        __s_fn_id__ = f"{dataWorkLoads.__name__} @staticmethod <build_pgsql_query>"

        insert_query=""
        on_conflict_clause=""

        try:
            cols_str = ', '.join(cols)

            insert_query = """ INSERT INTO %s (%s) VALUES %%s """ % (
                table_name, cols_str
            )

            if cols_not_for_update is not None:
                cols_not_for_update.extend(unique_key)
            else:
                cols_not_for_update = [col for col in unique_key]

            unique_key_str = ', '.join(unique_key)

            ''' construct update columns removing the ommited columns '''
            update_cols = [col for col in cols if col not in cols_not_for_update]
            update_cols_str = ', '.join(update_cols)
            ''' add prefix EXCLUDED to the cols to be updated '''
            update_cols_with_excluded_markers = [f'EXCLUDED.{col}' for col in update_cols]
            update_cols_with_excluded_markers_str = ', '.join(update_cols_with_excluded_markers)
            ''' constrict the on conflict insertging the keys, cols, and excludes '''
            on_conflict_clause = """ ON CONFLICT (%s) DO UPDATE SET (%s) = (%s) ;""" % (
                unique_key_str,
                update_cols_str,
                update_cols_with_excluded_markers_str
            )
#             print(insert_query + on_conflict_clause)
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return insert_query + on_conflict_clause



    ''' - Functions --- DB TABLE INFO ---

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    def get_db_table_info(
        self,
        db_type : str = None,
        db_name : str = None,
        db_schema:str = None,
        **options
    ) -> DataFrame:
        """
        Description:
            fetches the table information, such as catelog, schema, name of all or specific
                tables from the specified database and schema
                Necessary to validate the table exisits before performing an CRUD actions
        Attributes :
            db_name (str) database name is mandatory
            kwargs (dict)
                DBSCHEMA (str) if not defined will list tables for all schemas
        returns :
            table_info_sdf_ (DataFrame)
        """

        __s_fn_id__ = f"{self.__name__} function <get_db_table_info>"

        ''' returning sdf column name '''
        __comm_tbl_name_col__="table_name"
        __comm_schema_col__ = "schema_names"   
        __def_pg_db_info_tbl__= "information_schema.tables"
        __def_pg_schema_col__ = "table_schema"

        try:
            ''' validate database name '''
            if db_type is not None and "".join(db_type.split())!="":
                self.dbType = db_type
            if db_name is not None and "".join(db_name.split())!="":
                self.dbName = db_name
            if db_schema is not None and "".join(db_schema.split())!="":
                self.dbSchema = db_schema
            if self.dbType == 'postgresql':
                ''' set the partitions '''
                if "PARTITIONS" in options.keys():
                    self.partitions = options['PARTITIONS']
                if "FORMAT" in options.keys():
                    self.rwFormat = options['FORMAT']
                if "url" not in options.keys():
                    options['url'] = self.dbConnURL
                if "numPartitions" not in options.keys():
                    options['numPartitions'] = self.partitions
                if "user" not in options.keys():
                    options['user'] = self.dbUser
                if "password" not in options.keys():
                    options['password'] = self.dbPswd
                if "driver" not in options.keys():
                    options['driver'] = self.dbDriver
                if "dbtable" not in options.keys() or "".join(options['dbtable'])=="":
                    options['dbtable'] = __def_pg_db_info_tbl__

                table_info_sdf_ = None
                table_info_sdf_ = self.session.read\
                    .format(self.rwFormat)\
                    .options(**options)\
                    .load()
                if not isinstance(table_info_sdf_,DataFrame):
                    raise RuntimeError("Failed reading table info from %s %s returned empty %s" 
                                       % (self._dbType.upper(), self.dbName.upper(),
                                          type(table_info_sdf_)))
                ''' rename columns to common name '''
                table_info_sdf_ = table_info_sdf_\
                                    .withColumnRenamed("table_name",__comm_tbl_name_col__)\
                                    .withColumnRenamed("table_schema",__comm_schema_col__)\
#                 logger.debug("%s returned %d rows and %d columns for %s %s",
#                              __s_fn_id__, table_info_sdf_.count(), len(table_info_sdf_.columns), 
#                              self._dbType.upper(), self.dbName.upper())
            elif self.dbType == 'bigquery':
                pass
            else:
                raise AttributeError("Undefined dbType; something went wrong")
            
            logger.debug("%s returned %d rows and %d columns for %s %s",
                         __s_fn_id__, table_info_sdf_.count(), len(table_info_sdf_.columns), 
                         self._dbType.upper(), self.dbName.upper())

            ''' filter by Schema name '''
            if self._dbSchema:
                table_info_sdf_ = table_info_sdf_.filter(
                    F.col(__comm_schema_col__).isin([self._dbSchema])
                )
                logger.debug("%s filter for schema %s returned %d rows and %d columns",
                             __s_fn_id__, self._dbSchema.upper(), 
                             table_info_sdf_.count(), len(table_info_sdf_.columns))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return table_info_sdf_

    ''' - Functions --- GET TABLE SCHEMA ---

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    def get_table_schema(
        self,
        tbl_name : str = None,
        db_schema:str = None,
        **options
    ) -> DataFrame:
        """
        Description:
            fetches the table information, such as catelog, schema, name of all or specific
                tables from the specified database and schema
                Necessary to validate the table exisits before performing an CRUD actions
        Attributes :
            db_name (str) database name is mandatory
            kwargs (dict)
                DBSCHEMA (str) if not defined will list tables for all schemas
        returns :
            table_info_sdf_ (DataFrame)
        """

        __s_fn_id__ = f"{self.__name__} function <get_table_schema>"
        
        __def_pg_db_info_tbl__= "information_schema.tables"
        __def_pg_schema_col__ = "table_schema"

        try:
            ''' validate database name '''
            if tbl_name is None or "".join(tbl_name.split())=="":
                raise AttributeError("Unspecified empty %s" % type(tbl_name))
            if db_schema is not None and "".join(db_schema.split())!="":
                self.dbSchema = db_schema
            if self.dbType == 'postgresql':
                ''' set the partitions '''
                if "PARTITIONS" in options.keys():
                    self.partitions = options['PARTITIONS']
                if "FORMAT" in options.keys():
                    self.rwFormat = options['FORMAT']
                if "url" not in options.keys():
                    options['url'] = self.dbConnURL
                if "numPartitions" not in options.keys():
                    options['numPartitions'] = self.partitions
                if "user" not in options.keys():
                    options['user'] = self.dbUser
                if "password" not in options.keys():
                    options['password'] = self.dbPswd
                if "driver" not in options.keys():
                    options['driver'] = self.dbDriver
#                 if "dbtable" not in options.keys() or "".join(options['dbtable'])=="":
#                     options['dbtable'] = __def_pg_db_info_tbl__

                _query = f"SELECT * FROM {self.dbSchema}.{tbl_name} LIMIT 1"
                tbl_schema_ = None
                tbl_schema_ = self.read_data_from_table(
                    select=_query,
                    **options
                ).schema
                _tbl_fields_lst = [field.name for field in tbl_schema_.fields]
                if len(_tbl_fields_lst)<=0:
                    raise ValueError("Unable to recover field names from table %s, returned empty %s" 
                                     % (tbl_name, type(_tbl_fields_lst)))
                logger.debug("%s %s fields list: %s", __s_fn_id__, tbl_name, _tbl_fields_lst)


        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return tbl_schema_


    ''' - Functions --- SPARK POSTGRES TABLE PK NEXTVAL ---

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    def get_table_pk_nextval(
        self,
        tbl_name:str=None,
        pk_attr :str=None,
        **options
    ) -> DataFrame:
        """
        Description:
            fetches the table next pk value
        Attributes :
            tbl_name (str) table name is mandatory
            pk_attr (str) attribute name of serial primary key is mandatory
            options (dict) - unnecesary
        returns :
            pk_sdf_ (DataFrame) nextval column
        Exceptions:
            * Validate the table is not empty
            * Validate the pk attribute not empty
        """

        __s_fn_id__ = f"{self.__name__} function <get_table_pk_nextval>"


        try:
            ''' validate database name '''
            if tbl_name is None or "".join(tbl_name.split())=="":
                raise AttributeError("Undefined tbl_name %s; specify a proper table name" 
                                     % type(tbl_name))
            if pk_attr is None or "".join(pk_attr.split())=="":
                raise AttributeError("Undefined pk_attr %s; specify a proper primary key column" 
                                     % type(pk_attr))
            if self.dbType == 'postgresql':
                ''' set the partitions '''
                if "PARTITIONS" in options.keys():
                    self.partitions = options['PARTITIONS']
                if "FORMAT" in options.keys():
                    self.rwFormat = options['FORMAT']
                if "url" not in options.keys():
                    options['url'] = self.dbConnURL
                if "numPartitions" not in options.keys():
                    options['numPartitions'] = self.partitions
                if "user" not in options.keys():
                    options['user'] = self.dbUser
                if "password" not in options.keys():
                    options['password'] = self.dbPswd
                if "driver" not in options.keys():
                    options['driver'] = self.dbDriver
                ''' pre query '''
                _query = "SELECT nextval(pg_get_serial_sequence"
                _query += f"('{self._dbSchema}.{tbl_name}','{pk_attr}'))"
                options['query'] = _query
                
                pk_sdf_ = self.session.read\
                    .format(self.rwFormat)\
                    .options(**options)\
                    .load()
                if not isinstance(pk_sdf_,DataFrame):
                    raise RuntimeError("Failed reading %s table %s %s next value pk returned empty %s" 
                                       % (f"{self._dbSchema}.{tbl_name}", self._dbType.upper(), 
                                          self.dbName.upper(), type(pk_sdf_)))
                logger.debug("%s returned %d rows and %d columns for %s table in %s %s",
                             __s_fn_id__, pk_sdf_.count(), len(pk_sdf_.columns), 
                             f"{self._dbSchema}.{tbl_name}",
                             self._dbType.upper(), self.dbName.upper())


        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return pk_sdf_

    ''' - Functions --- SPARK POSTGRES TABLE PK NEXTVAL ---

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    def set_table_pk_lastval(
        self,
        tbl_name:str=None,
        pk_attr :str=None,
        set_val :int=None,
        **options
    ) -> DataFrame:
        """
        Description:
            set the table next pk value
        Attributes :
            tbl_name (str) table name is mandatory
            pk_attr (str) attribute name of serial primary key is mandatory
            options (dict) - unnecesary
        returns :
            pk_sdf_ (DataFrame) nextval column
        Exceptions:
            * Validate the table is not empty
            * Validate the pk attribute not empty
        """

        __s_fn_id__ = f"{self.__name__} function <set_table_pk_lastval>"


        try:
            ''' validate database name '''
            if tbl_name is None or "".join(tbl_name.split())=="":
                raise AttributeError("Undefined tbl_name %s; specify a proper table name" 
                                     % type(tbl_name))
            if pk_attr is None or "".join(pk_attr.split())=="":
                raise AttributeError("Undefined pk_attr %s; specify a proper primary key column" 
                                     % type(pk_attr))
            if self.dbType == 'postgresql':
                ''' set the partitions '''
                if "PARTITIONS" in options.keys():
                    self.partitions = options['PARTITIONS']
                if "FORMAT" in options.keys():
                    self.rwFormat = options['FORMAT']
                if "url" not in options.keys():
                    options['url'] = self.dbConnURL
                if "numPartitions" not in options.keys():
                    options['numPartitions'] = self.partitions
                if "user" not in options.keys():
                    options['user'] = self.dbUser
                if "password" not in options.keys():
                    options['password'] = self.dbPswd
                if "driver" not in options.keys():
                    options['driver'] = self.dbDriver

#                 ''' decide on value to set with '''
#                 _query = f"SELECT max({pk_attr}) as max_val FROM {self._dbSchema}.{tbl_name}"
#                 options['query'] = _query
#                 ''' get the value '''
#                 pk_deq_sdf_ = self.session.read\
#                     .format(self.rwFormat)\
#                     .options(**options)\
#                     .load()
#                 if not isinstance(pk_deq_sdf_,DataFrame) or pk_deq_sdf_.count()<1:
#                     raise RuntimeError("Failed reading %s table %s %s next value pk returned empty %s" 
#                                        % (f"{self._dbSchema}.{tbl_name}", self._dbType.upper(), 
#                                           self.dbName.upper(), type(pk_deq_sdf_)))
#                 _new_val = pk_deq_sdf_.select('max_val').collect()[0][0]
#                 if not isinstance(_new_val,int):
#                     _new_val=1
#                 if not isinstance(set_val, int) or set_val <= _new_val:
#                     set_val = _new_val
#                     logger.warning("%s invalid %s primary key value; setting to default %d",
#                                    __s_fn_id__, pk_attr.upper(), set_val)
# #                 ''' set the value in sequence table '''
# #                 _query = f"SELECT setval('{self._dbSchema}.{tbl_name}_{pk_attr}_seq', {set_val}, true)"
                ''' another way to set_val of pk_attr of a table '''
                _query = "SELECT pg_catalog.setval(pg_get_serial_sequence("
                _query += f"'{self._dbSchema}.{tbl_name}','{pk_attr}'),"
                _query += f"(SELECT MAX({pk_attr}) FROM {self._dbSchema}.{tbl_name})+1)"
                options['query'] = _query
                ''' get the value '''
                pk_deq_sdf_ = self.session.read\
                    .format(self.rwFormat)\
                    .options(**options)\
                    .load()
                if not isinstance(pk_deq_sdf_,DataFrame) or pk_deq_sdf_.count()<1:
                    raise RuntimeError("Failed setting %s table %s %s new value pk returned empty %s" 
                                       % (f"{self._dbSchema}.{tbl_name}", self._dbType.upper(), 
                                          self.dbName.upper(), type(pk_deq_sdf_)))
                set_val = pk_deq_sdf_.select('setval').collect()[0][0]
                logger.debug("%s Successfully set %s primary key nextval to %d %s table in %s %s",
                             __s_fn_id__, pk_attr.upper(), set_val, 
                             f"{self._dbSchema}.{tbl_name}".upper(),
                             self._dbType.upper(), self.dbName.upper())


        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return set_val
