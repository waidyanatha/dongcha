#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Function --- POPULATE UTIL REFER ---
    author: <nuwan.waidyanatha@rezgateway.com>
'''
def pop_util_ref(
    proj_dir= None,
    app:str = "rezaware",
    cfile:str="app.cfg",
):
    """
    Description:
        Configure the rezaware, wrangler, mining, and visuals apps
        If the apps (i.e., folders) don't exist, then it the process will create them
    Attributes:
        app (str) the name of the app wrangler, rezaware, mining, or visuals
    Returns:
        conf_files (list) absolute path of all .cfg and .ini files 
    """
    __s_fn_id__ = f"{__name__} function <build_app_struct>"
    __def_app_dir_list__ = ['dags','data','db','logs','modules','notebooks','tests']

    try:
        if proj_dir is None or "".join(proj_dir.split())=="":
            proj_dir = os.path.abspath(os.pardir)
        if app.lower() not in __apps_list__:
            raise AttributeError("Invalid app name %s,must be one of %s"
                                 %(app.upper(),str(__apps_list__)))
        logger.debug("%s %s project directory is %s",__s_fn_id__,app.upper(),proj_dir)
        ''' confirm app folder and config files exist '''
        _app_path = os.path.join(proj_dir,app.lower())
        if not os.path.isdir(_app_path):
            os.makedirs(_app_path)
            logger.warning("%s app dir unaivalable; created new %s dir: %s",
                         __s_fn_id__,app.upper(),_app_path)
        if not os.path.exists(os.path.join(_app_path,cfile)):
            ''' create default cfg file '''
            shutil.copy(os.path.join(proj_dir,"rezaware","000_defaults",app.lower(),cfile.lower()),
                        os.path.join(_app_path,cfile.lower()))
            logger.debug("%s copyied default %s to %s",__s_fn_id__,cfile.upper(),_app_path)

        ''' setup default app director structure '''
        for _app_dir in __def_app_dir_list__:
            if not os.path.isdir(os.path.join(_app_path,_app_dir)):
                os.makedirs(os.path.join(_app_path,_app_dir))
                logger.debug("%s added empty dir %s to %s",__s_fn_id__,_app_dir.upper(),_app_path)

        ''' execute rezaware to setup module/package ini files '''

        conf_files=app
    
    except Exception as err:
        logger.error("%s %s \n",__s_fn_id__, err)
        logger.debug(traceback.format_exc())
        print("[Error]"+__s_fn_id__, err)

    return conf_files

''' Function --- LOGGER ---
    author: <nuwan.waidyanatha@rezgateway.com>
'''
def get_logger_handle(
    cwd:str=None,
    folder_path:str="logs",
    file_name:str="app.log",
):
    """
    Description:
        Invokes a logging instance for the setup process
    Attribute:
        NA
    Returns:
        logger (logger) object
    """
    __s_fn_id__ = f"{__name__} function <get_logger_handle>"

    __def_log_form__ = '[%(levelname)s] - %(asctime)s - %(name)s - %(message)s'
    
    try:
        ''' validate function inputs '''
        if cwd is None or "".join(cwd.split())=="":
            cwd=os.path.dirname(__file__)
        if folder_path is None or "".join(folder_path.split())=="":
            folder_path="logs"
        if file_name is None or "".join(file_name.split())=="":
            file_name="app.log"

        ''' formulate file path '''
        if not os.path.isdir(os.path.join(cwd,folder_path)):
            os.makedirs(os.path.join(cwd,folder_path))
        _log_fpath=os.path.join(cwd,folder_path,file_name)
        if not os.path.exists(_log_fpath):
            with open(_log_fpath, 'w') as fp:
                pass

        ''' invoke logging instance '''
        logger = logging.getLogger("SETUP")
        logger.setLevel("DEBUG")
        if (logger.hasHandlers()):
            logger.handlers.clear()
        fh = logging.FileHandler(_log_fpath,'w+')
        fh.setLevel("DEBUG")
        formatter = logging.Formatter(__def_log_form__)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    except Exception as err:
        logger.error("%s %s \n",__s_fn_id__, err)
        logger.debug(traceback.format_exc())
        print("[Error]"+__s_fn_id__, err)

    return logger

''' Function --- MAIN --- 
    author: <nuwan.waidyanatha@rezgateway.com>
'''
def main():
    """
    Description:
        The main function will execute all steps for setting up a new project or
        updating the configuration of an existing project
    Attributes:
        NA
    Returns
        NA
    Exception(s):
    """

    __s_fn_id__ = f"{__name__} function <main>"

    global config
    global logger
    global rez

    print("Starting setup and configure project")

    try:
        cwd=os.path.dirname(__file__)
        sys.path.insert(1,cwd)
        ''' innitialize the logger '''
        logger = get_logger_handle(
            cwd=cwd,
            folder_path="logs",
            file_name = "app.log",
        )
        ''' set a new logger section '''
        logger.info('########################################################')
        logger.info("Initializing Main in %s",__name__)

#         from rezaware import Logger as logs
#         utils = __import__('000_utils')
        from utils import Config as conf
        ''' get args from command line '''
        parser = argparse.ArgumentParser(description="setup main controller arg and flags")
        parser.add_argument("--apps", type=str)
        parser.add_argument("--database", type=str)
        parser.add_argument("--storemode", type=str)
        parser.add_argument("--storeroot", type=str)
        parser.add_argument('--module', type=str)
        d = vars(parser.parse_args())

        ''' process if list of APPS are given '''
        if "apps" in d.keys() and d["apps"] is not None and "".join(d["apps"].split())!="":
            _app_set = [s.strip() for s in d["apps"].split(",")]
            logger.debug("%s apps to configure %s",__s_fn_id__,str(_app_set))
        else:
            _app_set=__apps_list__
            logger.warning("%s unrecognized app list, reverting to default %s",
                           __s_fn_id__,_app_set)

        ''' process if list of MODULES are given '''
        if "module" in d.keys() and d["module"] is not None and "".join(d["module"].split())!="":
            _module_set = [s.strip() for s in d["module"].split(",")]
            logger.debug("%s apps to configure %s",__s_fn_id__,str(_app_set))
        else:
            _app_set=__apps_list__
            logger.warning("%s unrecognized app list, reverting to default %s",
                           __s_fn_id__,_app_set)

        ''' get the parent folder '''
#         parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
        parent_dir=os.path.abspath(os.pardir)
        ''' find all the CSV files in app module default folders '''
        for _app in _app_set:
            try:
                ''' get app modules from app.cfg file '''

                ''' find the CSV files in app/module/defaul for util_ref'''
                
                ''' read the individual files and union into a dataframe '''
                
                ''' upsert the data in util_refer table '''

                _app_def_file_lst = read_util_ref(
                    proj_dir=parent_dir,
                    app = _app,
                    cfile=__conf_fname__,
                )
                logger.debug("%s initiated %s structure",__s_fn_id__,_app.upper())

                ''' add ini files for all the modules '''
                if "with_ini_files" in d.keys() and d["with_ini_files"]:
                    _app_ini_list = conf.create_ini_files(
                        reza_cwd=parent_dir,
                        app_name=_app,
                        app_path=os.path.join(parent_dir,_app),
                        conf_file=__conf_fname__,
                    )
                    if _app_ini_list is None or len(_app_ini_list)<=0:
                        raise RuntimeError("create_ini_files function return empty %s file list"
                                           % type(_app_ini_list))
                    logger.debug("%s generated files: %s",__s_fn_id__,str(_app_ini_list))

            except Exception as _app_err:
                logger.warning("%s app %s had errors %s",__s_fn_id__, _app.upper(),_app_err)
                print("** [WARNING] skipping %s had errors %s"
                      % (_app.upper(),_app_err))
            
        print("Setup process is Done!")
        
    except Exception as err:
        logger.error("%s %s \n",__s_fn_id__, err)
        logger.debug(traceback.format_exc())
        print("[Error]"+__s_fn_id__, err)

    return None

''' EXECUTION CONTROLLER '''
if __name__ == "__main__":

    ''' Initialize with default environment variables '''
    __name__ = "setup"
    __module__ = ""
    __package__ = ""
    __app__ = "rezaware"
    __ini_fname__ = "app.ini"
    __conf_fname__ = "app.cfg"
    __apps_list__ = ['wrangler','visuals','mining']
    __def_storemode__ = ['local-fs','aws-s3-bucket','google-storage']

    ''' Load necessary and sufficient python librairies that are used throughout the class'''
    try:
        ''' standard python packages '''
        import os
        import sys
        import logging
        import traceback
        import configparser
        import functools
        import argparse
        import shutil
        from datetime import datetime, date, timedelta

        print("All packages in %s %s %s %s imported successfully!"
              % (__app__,__module__,__package__,__name__))

        main()

    except Exception as e:
        print("Some software packages in {0} didn't load\n{1}".format(__name__,e))
