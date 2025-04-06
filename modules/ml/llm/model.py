#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "model"
__package__="llm"
__module__ ="ml"
__app__ = "rezaware"
__ini_fname__ = "app.ini"
__conf_fname__= "app.cfg"

''' Load necessary and sufficient python librairies that are used throughout the class'''
try:
    import os
    import sys
    import configparser    
    import logging
    import traceback
    import functools
#     import findspark
#     findspark.init()
#     from pyspark.sql import functions as F
# #     from pyspark.sql.functions import lit, current_timestamp
#     from pyspark.sql import DataFrame
#     from google.cloud import storage   # handles GCS reads and writes
#     import pandas as pd
#     import numpy as np
    import json
    from dotenv import load_dotenv
    load_dotenv()

    import litellm
    ''' LANGCHAIN '''
    from langchain_groq import ChatGroq
    from langchain.chat_models import ChatOllama

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))

'''
    CLASS establishes LLMs to use in Agentic processes
'''
class llmWorkLoads():
    ''' Function --- INIT ---
    
        author: <nuwan.waidyanatha@rezgateway.com>

    '''
    def __init__(
        self, 
        desc : str=None,   # identifier for the instances
        provider :str ="ollama",
        llm_name :str ="llama2-70b-chat",
        temperature:float=0.0,
        max_tokens : int =100,
        max_retries: int =0,
        base_url : str = "http://127.0.0.1:11434",
        **kwargs,
    ):
        """
        Description:
        Attributes:
        Returns:
            None
        """

        self.__name__ = __name__
        self.__package__ = __package__
        self.__module__ = __module__
        self.__app__ = __app__
        self.__ini_fname__ = __ini_fname__
        self.__conf_fname__ = __conf_fname__
        if isinstance(desc,str) and "".join(desc.split())!="":
            self.__desc__ = desc
        else:
            self.__desc__ = " ".join([self.__app__, self.__module__, self.__package__, self.__name__])

        ''' Initialize property var to hold the data '''
        self._provider=provider
        self._provList=[
            'anthropic', # Anthropic llm
            'azure',  # microsoft llm
            'gemini', # google llm
            'groq',   # groc llms
            'huggingface', # Huggingface
            'mistral',# mistral llm
            'ollama', # ollama local llm
            'openai'
        ]
        self._starCoder = llm_name
        self._starCoderList = [
            "llama-3.3-70b-versatile",
            "gemma:2b"
        ]
        self._temperature=temperature
        self._maxTokens = max_tokens
        self._maxReTries= max_retries
        self._baseURL = base_url

        ''' initiate to load app.cfg data '''
        global logger  # inherits the utils logger class
        global pkgConf # inherits package app.ini config data

        __s_fn_id__ = f"{self.__name__} function <__init__>"

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


            logger.info("%s Connection complete! ready to load data.",__s_fn_id__)
            logger.debug("%s initialization for %s module package %s %s done.\nStart workloads: %s."
                         %(self.__app__.upper(),
                           self.__module__.upper(),
                           self.__package__.upper(),
                           self.__name__.upper(),
                           self.__desc__.upper()))

            print("%s Class initialization complete" % self.__name__)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None

    ''' Function --- CLASS PROPERTY SETTER & GETTER ---

            author: <nuwan.waidyanatha@rezgateway.com>
    '''
    ''' --- PROVIDER --- '''
    @property
    def provider(self):

        __s_fn_id__ = f"{self.__name__} function <@property provider>"

        try:
            ''' validate provider value '''
            if self._provider not in self._provList:
                raise AttributeError("Invalid class property provider, must be %s" 
                                     % ", ".join(self._provList))
                
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._provList

    @provider.setter
    def provider(self,provider:str) -> str:

        __s_fn_id__ = f"{self.__name__} function <@provider.setter>"

        try:
            ''' validate provider value '''
            if provider.lower() not in self._provList:
                raise AttributeError("Invalid class property provider, must be %s" 
                                     % ", ".join(self._provList))

            self._provider = provider.lower()

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._provider

    ''' --- STARCODER --- '''
    @property
    def starCoder(self):

        __s_fn_id__ = f"{self.__name__} function <@property starCoder>"

        try:
            ''' validate llm name value '''
            if self._starCoder not in self._starCoderList:
                raise AttributeError("Invalid class property starCoder, %s must be one of %s" 
                                     % (type(self._llmName), ", ".join(self._starCoderList)))
                
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._starCoder

    @starCoder.setter
    def starCoder(self,star_coder:str):

        __s_fn_id__ = f"{self.__name__} function <@starCoder.setter>"

        try:
            ''' validate llm name value '''
            if star_coder.lower() not in self._starCoderList:
                raise AttributeError("Invalid class property star_coder, %s must be one of %s" 
                                     % (type(star_coder), ", ".join(self._starCoderList)))

            self._starCoder = llm_name.lower()

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._starCoder

    ''' --- TEMPERATURE --- '''
    @property
    def temperature(self):

        __s_fn_id__ = f"{self.__name__} function <@property temperature>"

        try:
            ''' validate temperature value '''
            if not isinstance(self._temperature, float) or not (0.0<=self._temperature<=1.0):
                raise AttributeError("Invalid class property temperature, %s" % type(self._temperature))
                
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._temperature

    @temperature.setter
    def temperature(self,temperature:float) -> float:

        __s_fn_id__ = f"{self.__name__} function <@temperature.setter>"

        try:
            ''' validate property value '''
            if not isinstance(temperature, float) or not (0.0<=temperature<=1.0):
                raise AttributeError("Invalid class property 0.0<=temperature<=1.0; must be a float")

            self._temperature = temperature

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._temperature

    # def patch_litellm(self):
    #     """
    #     """
    #     __s_fn_id__ = f"{self.__name__} function <patch_litellm>"

    #     try:
    #         original_get_llm_provider = litellm.get_llm_provider
    #         logger.debug("%s original_get_llm_provider = %s", 
    #                      __s_fn_id__, original_get_llm_provider)
    
    #         def custom_get_llm_provider(model, *args, **kwargs):
    #             _model = "/".join([self._provider.lower(), self._starCoder])
    #             if model == _model:
    #                 logger.debug("%s creating model: %s", __s_fn_id__, _model)
    #                 # return ChatOllama(
    #                 #     model=self._starCoder,
    #                 #     temperature=self._temperature,
    #                 #     max_tokens=self._maxTokens,
    #                 #     max_retries=self._maxReTries,
    #                 #     base_url=self._baseURL,
    #                 # )
    #                 ollama_model_ = ChatOllama(
    #                     model=_model, #self._starCoder,
    #                     temperature=self._temperature,
    #                     max_tokens=self._maxTokens,
    #                     max_retries=self._maxReTries,
    #                     base_url=self._baseURL,
    #                 )
    #                 if ollama_model_ is None:
    #                     raise RuntimeError("Failed to create model for:" % _model)
    #                 logger.debug("%s created ollama_model_= %s", __s_fn_id__, ollama_model_)
    #                 return ollama_model_
    #             origin_prov=None
    #             origin_prov = original_get_llm_provider(model, *args, **kwargs)
    #             if origin_prov is None:
    #                 raise ChildProcessError("Failed set origin_prov, returned" 
    #                                         % type(origin_prov))
    #             logger.debug("%s Set origin_prov = %s", __s_fn_id__, origin_prov)
    #             # return original_get_llm_provider(model, *args, **kwargs)
    #         litellm.get_llm_provider = custom_get_llm_provider
    #         if not litellm.get_llm_provider:
    #             raise ChildProcessError("Failed custom_get_llm_provider, returned %s" 
    #                                     % type(litellm.get_llm_provider))
    #         # Optionally, return something if needed

    #     except Exception as err:
    #         logger.error("%s %s \n",__s_fn_id__, err)
    #         logger.debug(traceback.format_exc())
    #         print("[Error]"+__s_fn_id__, err)
    #         return None
            
    #     finally:
    #         logger.debug("%s created %s", __s_fn_id__, self)
    #         return self

    def get(self):
        """
        """

        __s_fn_id__ = f"{self.__name__} function <get>"

        _ret_model = None

        try:
            _model = "/".join([self._provider.lower(), self._starCoder])
            if self._provider == "ollama":
                ''' running locally '''
                _ret_model=ChatOllama(
                    model=_model,#elf._starCoder,
                    temperature=self._temperature,
                    max_tokens=self._maxTokens,
                    max_retries=self._maxReTries,
                    base_url=self._baseURL,
                )
                from rezaware.modules.ml.llm import crewai_ollama_wrapper as wrapper
                # from rezaware.modules.ml.llm import crewai_wrapper_tool as tool
                _ret_model = wrapper.CrewAIOllamaWrapper(
                    ollama_model=_ret_model, 
                    model_name=_model
                )

                # original_get_llm_provider = litellm.get_llm_provider
                # _ret_model = ChatOllama(
                #     model=self._starCoder,
                #     temperature=self._temperature,
                #     max_tokens =self._maxTokens,
                #     max_retries=self._maxReTries,
                #     base_url=self._baseURL,
                #     )
                # # Monkey patch LiteLLM
                # litellm.get_llm_provider = custom_get_llm_provider
                # _ret_model=self.patch_litellm()

            elif self._provider == "groq":
                _model_name = "/".join([self._provider.lower(), self._starCoder])
                # _ret_model = ChatGroq(
                #     temperature=self._temperature,
                #     max_tokens =self._maxTokens,
                #     max_retries=self._maxReTries,
                #     model_name=_model_name,
                #     api_key = os.environ.get('GROQ_API_KEY')
                # )
                _ret_model=ChatGroq(
                    temperature=self._temperature,
                    max_tokens=self._maxTokens,
                    max_retries=self._maxReTries,
                    model_name=_model_name,#self._starCoder,
                    api_key=os.environ.get("GROQ_API_KEY")
                )
            else:
                raise RuntimeError(f"Provider {self._provider} not supported in this implementation.")

            ''' check return value '''
            if _ret_model is None:
                raise ChildProcessError("Failed to establish a mode, returned %s" % type(_ret_model))

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            return None

        finally:
            logger.debug("%s Succeeded in building model %s", __s_fn_id__, _ret_model)
            return _ret_model
