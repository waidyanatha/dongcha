#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "model"
__package__="llm"
__module__ ="ml"
__app__ = "dongcha"
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
    import json
    from dotenv import load_dotenv
    load_dotenv()

    import litellm
    ''' LANGCHAIN '''
    from langchain_community.llms.fake import FakeListLLM
    from langchain_groq import ChatGroq
    from langchain.chat_models import ChatOllama
    from openai import OpenAI

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
            'deepseek', 
            'gemini', # google llm
            'groq',   # groq llms
            'huggingface', # Huggingface
            'mistral',# mistral llm
            'ollama', # ollama local llm
            'openai',
            'fake', # FakeLLM and default
        ]
        self._starCoder = llm_name
        self._starCoderList = [
            "llama-3.3-70b-versatile", # groq
            "deepseek-chat", # deepseek
            "gemma:2b",  # ollama
            "gemma:7b" # ollama
            "gemma-7b-it", # groq
            "mistral-saba-24b" #deprecated: "mixtral-8x7b-32768", # groq
            "llama-3.1-8b-instant", # groq
            "test", # a dummy startcode for FakeLLM
        ]
        self._temperature=temperature
        self._maxTokens = max_tokens
        self._maxReTries= max_retries
        self._baseURL = base_url

        ''' initiate to load app.cfg data '''
        __s_fn_id__ = f"{self.__name__} function <__init__>"

        try:
            self.cwd=os.path.dirname(__file__)
            self.pkgConf = configparser.ConfigParser()
            self.pkgConf.read(os.path.join(self.cwd,__ini_fname__))

            self.projHome = self.pkgConf.get("CWDS","PROJECT")
            sys.path.insert(1,self.projHome)

            ''' initialize the logger '''
            from dongcha.utils import Logger as logs
            self.logger = logs.get_logger(
                cwd=self.projHome,
                app=self.__app__, 
                module=self.__module__,
                package=self.__package__,
                ini_file=self.__ini_fname__)
            ''' set a new logger section '''
            self.logger.info('########################################################')
            self.logger.info("%s %s",self.__name__,self.__package__)

            self.logger.debug("%s initialization for %s module package %s %s done.\nStart workloads: %s."
                         %(self.__app__.upper(),
                           self.__module__.upper(),
                           self.__package__.upper(),
                           self.__name__.upper(),
                           self.__desc__.upper()))

            print("%s Class initialization complete" % self.__name__)

        except Exception as err:
            self.logger.error("%s %s \n",__s_fn_id__, err)
            self.logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None


    def get(self):
        """
        """

        __s_fn_id__ = f"{self.__name__} function <{self.get.__name__}>"

        _ret_model = None

        try:
            ''' used for model identification '''
            _model = self._starCoder  # Use direct model name
            self.logger.debug("%s setting model as: %s for provider: %s", __s_fn_id__, _model, self.provider)
            if self.provider != "openai" and 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']

            if self.provider == "ollama":
                ''' running locally '''
                _ret_model=ChatOllama(
                    model=_model,
                    temperature=self.temperature,
                    # max_tokens parameter removed as it's not supported by ChatOllama
                    # max_retries parameter removed as it's not supported by ChatOllama
                    base_url=self._baseURL,
                )
                if not _ret_model:
                    raise ChildProcessError("Failed to set ChatOllama with model: received %s" 
                                            % type(_ret_model))

                # Import and use wrapper if available
                try:
                    from dongcha.modules.ml.llm import crewai_ollama_wrapper as wrapper
                    self.logger.debug("%s Wrapping ChatOllama with model_name: %s", __s_fn_id__, _model)
                    _ret_model = wrapper.CrewAIOllamaWrapper(
                        ollama_model=_ret_model, 
                        model_name=_model
                    )
                    self.logger.debug("%s Wrapper created successfully: %s", __s_fn_id__, _ret_model)
                except ImportError as ie:
                    self.logger.warning("%s Wrapper not available, using direct ChatOllama: %s", __s_fn_id__, ie)
                except Exception as we:
                    self.logger.error("%s Error creating wrapper: %s", __s_fn_id__, we)
                    self.logger.debug(traceback.format_exc())

            elif self.provider == "groq":
                _model = "/".join([self._provider, self._starCoder]) # need to prefix it with the provider name
                _ret_model=ChatGroq(
                    temperature=self.temperature,
                    max_tokens=self.maxTokens,
                    max_retries=self.maxReTries,
                    model_name=_model,
                    api_key=os.environ.get("GROQ_API_KEY")
                )
                
            elif self.provider == 'fake':
                _ret_model=FakeListLLM(
                    responses=[
                        "This is a test response 1",
                        "This is a test response 2"
                        ])

            elif self.provider == "deepseek":
                _ret_model = OpenAI(
                    api_key=os.environ.get("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com",  # Correct DeepSeek API endpoint
                )
                # Note: temperature, max_tokens, etc. are passed during completion calls, not init
                
            elif self.provider == "openai":
                _ret_model = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
                
            else:
                 raise NotImplementedError(f"Provider {self.provider} not supported yet")

            ''' check return value '''
            if _ret_model is None:
                raise RuntimeError("Failed to establish a model, returned %s" % type(_ret_model))

        except Exception as err:
            self.logger.error("%s %s \n",__s_fn_id__, err)
            self.logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            return None

        finally:
            self.logger.debug("%s Succeeded in building model %s", __s_fn_id__, type(_ret_model))
            return _ret_model

    ''' Function --- CLASS PROPERTY SETTER & GETTER ---
            author: <samana.thetha@gmail.com>
    '''
    ''' --- PROVIDER --- '''
    @property
    def provider(self):

        __s_fn_id__ = f"{self.__name__} function <@property provider>"

        try:
            ''' validate provider value '''
            if self._provider is None or self._provider.lower() not in self._provList:
                self._provider = "fake"
                self.logger.warning("%s Invalid provider set to default: %s, did you mean %s",
                                    __s_fn_id__, self._provider.upper(), ", ".join(self._provList))
                
        except Exception as err:
            self.logger.error("%s %s \n",__s_fn_id__, err)
            self.logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._provider

    @provider.setter
    def provider(self,provider:str) -> str:

        __s_fn_id__ = f"{self.__name__} function <@provider.setter>"

        try:
            ''' validate provider value '''
            if provider is None or provider.lower() not in self._provList:
                raise AttributeError("Invalid class property provider, must be %s" 
                                     % ", ".join(self._provList))
            self._provider = provider.lower()

        except Exception as err:
            self.logger.error("%s %s \n",__s_fn_id__, err)
            self.logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._provider

    ''' --- STARCODER --- '''
    @property
    def starCoder(self):

        __s_fn_id__ = f"{self.__name__} function <@property starCoder>"

        try:
            ''' validate llm name value '''
            if self._starCoder is None or self._starCoder.lower() not in self._starCoderList:
                self._starCoder = "test"
                self.logger.warning("%s Invalid starCoder %s set to default: %s or did you mean: %s",
                                    __s_fn_id__, self._starCoder, self._starCoder.upper(), 
                                        ", ".join(self._starCoderList))
                
        except Exception as err:
            self.logger.error("%s %s \n",__s_fn_id__, err)
            self.logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._starCoder

    @starCoder.setter
    def starCoder(self,star_coder:str):

        __s_fn_id__ = f"{self.__name__} function <@starCoder.setter>"

        try:
            ''' validate llm name value '''
            if star_coder is None or star_coder.lower() not in self._starCoderList:
                raise AttributeError("Invalid class property starCoder, %s must be one of %s" 
                                     % (star_coder, ", ".join(self._starCoderList)))
            else:
                self._starCoder = star_coder.lower()

        except Exception as err:
            self.logger.error("%s %s \n",__s_fn_id__, err)
            self.logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._starCoder

    ''' --- TEMPERATURE --- '''
    @property
    def temperature(self):

        __s_fn_id__ = f"{self.__name__} function <@property temperature>"

        try:
            ''' validate temperature value '''
            if not isinstance(self._temperature, (int, float)) or not (0.0<=self._temperature<=1.0):
                self._temperature = 0.1
                self.logger.warning("%s Invalid %s temperature set to: %0.2f; must be a float 0.0<=temperature<=1.0",
                                    __s_fn_id__, type(self._temperature), self._temperature)
                
        except Exception as err:
            self.logger.error("%s %s \n",__s_fn_id__, err)
            self.logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._temperature

    @temperature.setter
    def temperature(self,temperature:float) -> float:

        __s_fn_id__ = f"{self.__name__} function <@temperature.setter>"

        try:
            ''' validate property value '''
            if not isinstance(temperature, (int, float)) or not (0.0<=temperature<=1.0):
                raise AttributeError("Invalid property temperature, %s; must be a float 0.0<=temperature<=1.0"
                                     % type(temperature))

            self._temperature = float(temperature)

        except Exception as err:
            self.logger.error("%s %s \n",__s_fn_id__, err)
            self.logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._temperature

    ''' --- MAX TOKENS --- '''
    @property
    def maxTokens(self):

        __s_fn_id__ = f"{self.__name__} function <@property maxTokens>"
        __def_max_tokens__ = 300

        try:
            ''' validate max tokens value '''
            if not isinstance(self._maxTokens, int) or self._maxTokens<=0:
                self._maxTokens = __def_max_tokens__
                self.logger.warning("%s Invalid %s max_tokens set to: %d; must be an int >0",
                                    __s_fn_id__, type(self._maxTokens), self._maxTokens)
                
        except Exception as err:
            self.logger.error("%s %s \n",__s_fn_id__, err)
            self.logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._maxTokens

    @maxTokens.setter
    def maxTokens(self,max_tokens:int) -> int:

        __s_fn_id__ = f"{self.__name__} function <@maxTokens.setter>"

        try:
            ''' validate property value '''
            if not isinstance(max_tokens, int) or max_tokens<=0:
                raise AttributeError("Invalid property max_tokens, %s; must be an int > 0"
                                     % type(max_tokens))

            self._maxTokens = max_tokens

        except Exception as err:
            self.logger.error("%s %s \n",__s_fn_id__, err)
            self.logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._maxTokens

    ''' --- MAX RETRIES --- '''
    @property
    def maxReTries(self) -> int:

        __s_fn_id__ = f"{self.__name__} function <@property maxReTries>"
        __def_max_retries__ = 0

        try:
            ''' validate max retries value '''
            if not isinstance(self._maxReTries, int) or self._maxReTries<0:
                self._maxReTries = __def_max_retries__
                self.logger.warning("%s Invalid %s max retries set to: %d; must be an int >= 0",
                                    __s_fn_id__, type(self._maxReTries), self._maxReTries)
                
        except Exception as err:
            self.logger.error("%s %s \n",__s_fn_id__, err)
            self.logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._maxReTries

    @maxReTries.setter
    def maxReTries(self,max_retries:int) -> int:

        __s_fn_id__ = f"{self.__name__} function <@maxReTries.setter>"

        try:
            ''' validate property value '''
            if not isinstance(max_retries, int) or max_retries<0:
                raise AttributeError("Invalid property max_retries, %s; must be an int >= 0"
                                     % type(max_retries))

            self._maxReTries = max_retries

        except Exception as err:
            self.logger.error("%s %s \n",__s_fn_id__, err)
            self.logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return self._maxReTries


    def debug_info(self):
        """Debug method to show current configuration"""
        info = {
            'provider': self._provider,
            'starCoder': self._starCoder,
            'temperature': self._temperature,
            'maxTokens': self._maxTokens,
            'maxReTries': self._maxReTries,
            'baseURL': self._baseURL,
            'valid_providers': self._provList,
            'valid_models': self._starCoderList
        }
        print(f"Current LLM Config: {info}")
        return info
