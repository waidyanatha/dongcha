#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from flask import Flask, request
app = Flask(__name__)

# __actions__ = ["INFO", "CREATE", "READ", "UPDATE", "DEACTIVATE" ]
# __methods__ = ["GET", "POST", "DELETE", "PUT"]
# __desc__ = "Flask instantiated rezaware module: "

@app.route('/')
def index():
    """
    Description:
        default api provides no functionality and can be used to test the api
    Attributes :
        None
    Returns :
        ret_dict (dict) - welcome message and list of endpoints and parameter
            descriptions
    """
    ret_dict = {
        "data" : ["Welcome to REZAWARE API controller", 
                  "You have successfully accessed rezaware api controller",
        {
            "endpoints" : [
                {"etl/loader" :
                 [
                         {'/nosql':'CRUD mongodb collections and documents with apache spark'},
                         {'/rdbm' :'CRUD postgresql and bigquery database tables with apache spark'},
                         {'/file' :'CRUD local folder, AWS S3, and GCS buckets with apache spark'}
                 ]
                }
            ]},
            {"methods" : ["GET", "POST", "DELETE", "PUT"]},
            {"actions" : [
                {"INFO" : 'retrieve info about a resource: class, method, and properties'}, 
                {"CREATE":'create a data object or insert data in a table, collection, etc'}, 
                {"READ" : 'read data from memory, structured, or unstructure data stores'}, 
                {"UPDATE":'update data in memory, unstructured, or structured data store'},
                {"DEACTIVATE":'permanently diable an an object or data (instead of deleting)'}
            ]}
                 ],
        "error" : None,
        "status": 200
    }
    return ret_dict 


@app.route('/etl/loader/rdbm', methods=["GET", "POST", "DELETE", "PUT"])
def etl_loader_rdbm():
    """
    Description: etl/loader/rdbm endpoint to CRUD postgresql & bigquery
        database tables using apache sparkRDBM dataWorkLoads class
    Attributes :
        package (str) - must be one of the following: nosql, rdbm, or file
    returns:
        
    """
    __s_fn_id__ = f"{__name__} function <etl_loader_rdbm>"

    ret_val = "Invalid method or action"
    params_ = {}

    try:
        
        clsExeRes = package_action(request=request, desc=__s_fn_id__)
        ret_val = clsExeRes.rdbm()
            
    except Exception as err:
#         logger.error("%s %s \n",__s_fn_id__, err)
#         logger.debug(traceback.format_exc())
        print("[Error]"+__s_fn_id__, err)
        
    return ret_val


__actions__ = ["INFO", "CREATE", "READ", "UPDATE", "DEACTIVATE" ]
__methods__ = ["GET", "POST", "DELETE", "PUT"]
__desc__ = "Flask instantiated rezaware module: "

class package_action(object):

    def __init__(
        self,
        request,
        desc:str="package action",
    ):
        """
        Description:
            parese request to confirms action and method are given and 
            assign to class properties 
        Attributes :
            request (object)
        Returns:
            None
        """

        __s_fn_id__ = f"{__name__} function <__init__>"

        try:
            self.__desc__ = desc   # description for instantiating the specific class instance
            ''' Validate method and action '''
            ''' METHOD '''
            if request.method.upper() not in __methods__:
                raise AttributeError("Unspecified method; must be one of %s" % (__methods__))
            self._method = request.method.upper()
            params = {k.lower():v.lower() for k,v in dict(request.args).items()}
            ''' ACTION '''
            if "action" not in params.keys() or params['action'].upper() not in __actions__:
                raise AttributeError("Unspecified action; must be one of %s" % (__actions__))
            self._action = params['action'].upper()   # resource function; e.g. info, read, write, 

            ''' set class properties '''
#             self._package= params['package'] # package name
            ''' DB TYPE '''
            self._db_type_lst = ['mongodb','postgresql', 'bigquery']
            self._dbtype = None
            if "dbtype" in params.keys() and params['dbtype'] in self._db_type_lst:
                self._dbtype = params['dbtype']
            ''' DRIVER '''
            self._dbdriver = None
            if "dbdriver" in params.keys() and "".join(params['dbdriver'].split())!="":
                self._dbdriver = params['dbdriver']
            ''' HOSTIP '''
            self._dbhostip = None
            if "dbhostip" in params.keys() and "".join(params['dbhostip'].split())!="":
                self._dbhostip = params['dbhostip']
            ''' PORT '''
            self._dbport = None
            if "dbport" in params.keys() and isinstance(params['dbport'],int):
                self._dbport = params['dbport']
            ''' NAME '''
            self._dbname = None
            if "dbname" in params.keys() and "".join(params['dbname'].split())!="":
                self._dbname = params['dbname']
            ''' SCHEMA '''
            self._dbschema = None
            if "dbschema" in params.keys() and "".join(params['dbschema'].split())!="":
                self._dbschema = params['dbschema']
            ''' AUTH '''
            self._dbauth = None
            if "dbauth" in params.keys() and "".join(params['dbauth'].split())!="":
                self._dbauth = params['dbauth']
            ''' TABLE '''
            self._tblname = None
            if "tblname" in params.keys() and "".join(params['tblname'].split())!="":
                self._tblname = params['tblname']
            ''' QUERY '''
            self._query = None
            if "query" in params.keys() and "".join(params['query'].split())!="":
                self._query = params['query'].replace("%20",' ')
            ''' USER '''
            self._dbuser = None
            if "dbuser" in params.keys() and "".join(params['dbuser'].split())!="":
                self._dbuser = params['dbuser']
            ''' PASSWORD '''
            self._dbpswd = None
            if "dbpswd" in params.keys() and "".join(params['dbpswd'].split())!="":
                self._dbpswd = params['dbpswd']
            ''' DATA '''
            self._data = None
            if request.get_json() is not None:
                self._data = request.get_json()

        except Exception as err:
    #         logger.error("%s %s \n",__s_fn_id__, err)
    #         logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None


    ''' Function: --- RDBM ---
        author : <nuwan.waidyanatha@rezgateway.com>
    '''
    def rdbm(self):
        """
        """

        from pyspark.sql import functions as F
        import json

        __s_fn_id__ = f"{__name__} function <rdbm>"

        ret_dict = {
            "data" : None,
            "status":None,
            "error": None
        }

        try:
            from modules.etl.loader import sparkRDBM as rdbm
            clsSDB = rdbm.dataWorkLoads(
                desc=self.__desc__,
                db_type = self._dbtype, # database type one of self._dbTypeList
                db_driver=self._dbdriver,
                db_hostIP=self._dbhostip,
                db_port = self._dbport,
                db_name = self._dbname,
                db_schema=self._dbschema,
                db_user = self._dbuser,
                db_pswd = self._dbpswd,
                spark_partitions=None,
                spark_format = None,
                spark_save_mode=None,
                spark_jar_dir = None,
            )

            if self._method=="GET":
                ''' INFO '''
                if self._action=='INFO':
                    sdf_ = clsSDB.get_db_table_info()
                    if sdf_ is None or sdf_.count()<=0:
                        raise RuntimeError("Failed retrieving %s for schema %s and db %s" 
                                           % (self._action, self._dbschema.upper(),
                                              self._dbname.upper()))
                    sdf_ = sdf_.select(F.col('table_schema'),F.col('table_name'))
                    
                elif self._action=='READ':
                    sdf_=clsSDB.read_data_from_table(
                        select=self._query,
                        db_table=self._tblname,
                        db_column="",
                        lower_bound=None,
                        upper_bound=None,
                    )

                else:
                    raise AttributeError("Unrecognized %s action for %s method" 
                                         % (self._action, self._method))
                if sdf_ is None or sdf_.count()<=0:
                    raise RuntimeError("%s method with %s action returned an empty %s object"
                                       % (self._method, self._action,str(type(sdf_)).upper()))
                ret_dict['data']=sdf_.toPandas().to_json(orient='table')

            elif self._method=="POST":
                _records= -1

                if self._action=='CREATE':
                    print(self._tblname, self._dbname)
#                     ret_dict['data'] = {"method":self._method, "action":self._action, "message":'TBD'}
                    if self._data is None or len(self._data)<=0:
                        raise AttributeError("Failed, %s table records in db %s; empty %s object" 
                                             % (self._tblname.upper(), self._dbname.upper(),
                                                str(type(self._data)).upper()))
                    ret_dict['data'] = self._data
                    _records= len(self._data)

                elif self._action=='UPDATE':
#                     ret_dict['data'] = {"method":self._method, "action":self._action, "message":'TBD'}
                    ret_dict['data'] = self._data

#                     _pk = ['ref_pk']
#                     _cols_not_for_update = ['created_dt','created_by','created_proc']
#                     _options={
#                         "BATCHSIZE":1000,   # batch size to partition the dtaframe
#                         "PARTITIONS":1,    # number of parallel clusters to run
#                         "OMITCOLS":_cols_not_for_update,    # columns to be excluded from update

#                     }
#                     _records=clsSDB.upsert_sdf_to_table(
#                         save_sdf=_upsert_sdf,
#                         db_name =_db_name,
#                         db_table=_tbl_name,
#                         unique_keys=_pk,
#                         **_options,
#                     )

                else:
                    raise AttributeError("Unrecognized %s action for % method" 
                                         % (self._action, self._method))
                if not isinstance(_records,int) or _records<0:
                    ret_dict['data'] = {"records processed":-1}
                    raise RuntimeError("Failed %s method with %s action" 
                                       % (self._method, self._action))

            else:
                raise AttributeError("Unrecognized method %s; something went wrong" % self._method)

            ret_dict['status']=200

        except AttributeError as a_err:
            ret_dict['error']=f"{__s_fn_id__} : {a_err}"
            ret_dict['status']=400   # Bad Request Error

        except RuntimeError as r_err:
            ret_dict['error']=f"{__s_fn_id__} : {r_err}"
            ret_dict['status']=404   # Not Found Error

        except Exception as err:
            print("[Error]"+__s_fn_id__, err)
            ret_dict['error']=f"{__s_fn_id__} : {err}"
            ret_dict['status']=500   # Internal Server Error

        return ret_dict


    def nosql(self):
        """
        """
        __s_fn_id__ = f"{__name__} function <nosql>"

        ret_dict = {
            "data" : None,
            "status":None,
            "error": None
        }

        try:
            from modules.etl.loader import sparkNoSQL as nosql
            clsNoSQL = nosql.dataWorkLoads(
                desc=self.__desc__,
                db_type = self._dbtype, # database type one of self._dbTypeList
                db_hostIP=None,
                db_port = None,
                db_name = None,
                db_authSource = None,
                db_collection = None,
                db_user = None,
                db_pswd = None,
                spark_partitions=None,
                spark_format = None,
                spark_master = None,
            )

            ret_dict['data'] = self._package
            ret_dict['status']=200

        except Exception as err:
    #         logger.error("%s %s \n",__s_fn_id__, err)
    #         logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            ret_dict['error']=f"{__s_fn_id__} : {err}"

        return ret_dict

