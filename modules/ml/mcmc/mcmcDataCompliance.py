#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "mcmcDataCompliance"
__package__ = "mcmc"
__module__ = "ml"
__app__ = "rezaware"
__ini_fname__ = "app.ini"
__conf_fname__ = "app.cfg"

''' Load necessary and sufficient python librairies that are used throughout the class'''
try:
    import os
    import sys
    import configparser    
    import logging
    import traceback
    import functools
    import findspark
    findspark.init()
    from pyspark.sql import functions as F
    from pyspark.sql import DataFrame
    from pyspark.sql.types import *
    from pyspark.sql.window import Window
    from datetime import datetime, date, timedelta

    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from statsmodels.tsa.stattools import adfuller, acf
    from statsmodels.graphics.tsaplots import plot_acf
    from sklearn.neighbors import KernelDensity
    from scipy.signal import find_peaks
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    from mining.modules.budget.evaluation import __propAttr__ as attr

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))

'''
    CLASS configure the master property details, groups, reference, and geom entities

    Contributors:
        * nuwan.waidyanatha@rezgateway.com

    Resources:

'''

class evalWorkLoads(attr.properties):

    ''' Function --- INIT ---

            author: <samana.thetha@gmail.com>
    '''
    def __init__(
        self, 
        desc : str = None,
        # f_store_mode:str= None,   # set the file storage mode
        # f_store_root:str= None,   # set the root folder (bucket)
        **kwargs):
        """
        Decription:
            Initializes the features: class property attributes, app configurations, 
                logger function, data store directory paths, and global classes 
        Attributes:
            desc (str) identify the specific instantiation and purpose
        Returns:
            None
        """

        ''' instantiate property attributes '''
        super().__init__(desc="read write RDB property class init")

        self.__name__ = __name__
        self.__package__ = __package__
        self.__module__ = __module__
        self.__app__ = __app__
        self.__ini_fname__ = __ini_fname__
        self.__conf_fname__ = __conf_fname__
        if desc is None or "".join(desc.split())=="":
            self.__desc__ = " ".join([self.__app__,self.__module__,
                                      self.__package__,self.__name__])
        else:
            self.__desc__ = desc

        global pkgConf  # this package configparser class instance
        global appConf  # configparser class instance
        global logger   # rezaware logger class instance
        global clsFile  # etl loader sparkFile class instance

        __s_fn_id__ = f"{self.__name__} function <__init__>"

        try:
#             ''' instantiate property attributes '''
#             super().__init__(self.__desc__)

            self.cwd=os.path.dirname(__file__)
            pkgConf = configparser.ConfigParser()
            pkgConf.read(os.path.join(self.cwd,__ini_fname__))

            self.projHome = pkgConf.get("CWDS","PROJECT")
            sys.path.insert(1,self.projHome)

            ''' initialize the logger '''
            from rezaware.utils import Logger as logs
            logger = logs.get_logger(
                cwd=self.projHome,
                app=self.__app__, 
                module=self.__module__,
                package=self.__package__,
                ini_file=self.__ini_fname__)

            ''' set a new logger section '''
            logger.info('########################################################')
            logger.info("%s Class",self.__name__)
            logger.debug("%s initialization for %s module package %s %s done.\nStart workloads: %s."
                         %(self.__app__,
                           self.__module__,
                           self.__package__,
                           self.__name__,
                           self.__desc__))

            print("%s Class initialization complete" % self.__name__)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None

    ''' Function --- STATIONARITY ---

            author: <samana.thetha@gmail.com
    '''
    @staticmethod
    def stationarity(data, significance=0.05) -> dict:
        """
        Test for stationarity using Augmented Dickey-Fuller test.
        
        Parameters:
        -----------
        data : array-like
            Time series data to test
        significance : float
            Significance level for the test
            
        Returns:
        --------
        p_value_, comply_, explain_
            True if stationary, False otherwise
        """

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <stationarity>"

        return_dict_ = {}

        try:
            result_ = adfuller(data)
            if result_ is None or len(result_)<=0:
                raise ChildProcessError("Failed ADF Test process for %s" % type(data))
            # p_value = result_[1]
            return_dict_["check"] = "stationary"
            return_dict_["p_value"] = result_[1]
            # print(f"ADF Test p-value: {p_value}")
            
            if return_dict_["p_value"] < significance:
                return_dict_["explained"] ="✓ Data appears to be stationary"
                return_dict_["comply"] = True
            else:
                return_dict_["explained"] ="✗ Data does not appear to be stationary"
                return_dict_["comply"] = False

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        finally:
            logger.debug("%s ADF Test completed with results: %s", __s_fn_id__, str(return_dict_))
            return return_dict_
    
    
    ''' Function --- DATA VOLUME ---

            author: <samana.thetha@gmail.com
    '''
    @staticmethod
    def data_volume(data, min_samples=1000):
        """
        Check if there is sufficient data volume.
        
        Parameters:
        -----------
        data : array-like
            Data to test
        min_samples : int
            Minimum recommended sample size
            
        Returns:
        --------
        bool
            True if sufficient, False otherwise
        """

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <data_volume>"

        return_dict_ = {}

        try:    
            return_dict_["check"] = "data volume"
            return_dict_['samples'] = len(data)
            # print(f"Number of samples: {n_samples}")
            
            if return_dict_['samples'] >= min_samples:
                return_dict_["explained"] ="✓ Data volume is likely sufficient (>= "+\
                                                f"{min_samples} samples)"
                return_dict_["comply"] = True
                # print(f"✓ Data volume is likely sufficient (>= {min_samples} samples)")
                # return True
            else:
                return_dict_["explained"] =f"✗ Data volume may be insufficient < "+\
                                                f"{min_samples} samples)"
                return_dict_["comply"] = False
                # print(f"✗ Data volume may be insufficient (< {min_samples} samples)")
                # return False
        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        finally:
            logger.debug("%s Data volume Test completed with results: %s", __s_fn_id__, str(return_dict_))
            return return_dict_
    
    
    ''' Function --- REPRESENTATIVENESS ---

            author: <samana.thetha@gmail.com
    '''
    @staticmethod
    def representativeness(data, multivariate=False, n_bins=20):
        """
        Visualize the distribution to check for representativeness.
        This is somewhat subjective and depends on domain knowledge.
        
        Parameters:
        -----------
        data : array-like
            Data to test
        n_bins : int
            Number of bins for histogram
        """

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <representativeness>"

        return_dict_ = {}
        return_dict_["check"] = "representativeness"
        return_dict_["comply"] = None
        return_dict_["plot"] = None
        _explain_str = ""
        current_backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Agg backend doesn't display figures

        try:           
            data_clean = data[~np.isnan(data)].reshape(-1, 1)
            nan_count = len(data) - len(data_clean)
            nan_percentage = 100 * nan_count / len(data) if len(data) > 0 else 0

            if len(data_clean) == 0:
                _explain_str +="✗ All data values are NaN. Cannot perform representativeness check. "
                _explain_str +="\nempty_regions= 0, nan_percentage: 100.0, valid_data: FALSE "

                # Create an empty figure if all data is NaN
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "All data values are NaN",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14)
                ax.set_title('Data Distribution (No Valid Data)')
                return_dict_["data"] = None

            else:
                logger.warning("%s Removed %d NaN values for visualization", 
                               __s_fn_id__, (len(data) - len(data_clean)))
                ''' fill NaN with mean '''
                imputer = SimpleImputer(strategy='mean')
                return_dict_["data"] = imputer.fit_transform(data_clean)
    
                if multivariate:
                    # For multivariate data, we'll use PCA for visualization
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(return_dict_["data"])
                    pca = PCA(n_components=1)
                    return_dict_["data"] = pca.fit_transform(data_scaled).flatten()
                    # return_dict_["data"]= plot_data
                    _explain_str += "Note: Using PCA first component for representativeness visualization.\n"

                if not isinstance(return_dict_["data"],np.ndarray) or return_dict_["data"].shape[0]<=0:
                    raise ChildProcessError("Failed to impute or apply PCA returned %s" 
                                            % type(return_dict_["data"]))

                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot histogram
                ax.hist(return_dict_["data"], bins=n_bins, alpha=0.7, density=True)
                
                # Plot kernel density estimate
                x_grid = np.linspace(
                    min(return_dict_["data"]), 
                    max(return_dict_["data"]), 1000)
                kde = KernelDensity(bandwidth=0.5).fit(data_clean.reshape(-1, 1))
                log_dens = kde.score_samples(x_grid.reshape(-1, 1))
                ax.plot(x_grid, np.exp(log_dens), 'r-', label='KDE')
                
                ax.set_title('Data Distribution')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend()
                
                # Check for zeros in the histogram bins
                hist, bin_edges = np.histogram(return_dict_["data"], bins=n_bins)
                empty_bins = np.sum(hist == 0)
                
                _explain_str += f"empty_regions': {empty_bins},  nan_percentage: {nan_percentage},\n"
                _explain_str += f"valid_data': True, bin_edges': {bin_edges}.\n "
                return_dict_["data"]=hist
                
                # Add text annotations
                if empty_bins > 0:
                    ax.text(0.02, 0.98, f"Found {empty_bins} empty regions out of {n_bins} bins", 
                            transform=ax.transAxes, va='top', fontsize=10, color='red')
                
                if nan_percentage > 0:
                    ax.text(0.02, 0.94, f"{nan_percentage:.2f}% missing values (NaNs)", 
                            transform=ax.transAxes, va='top', fontsize=10, color='red')

        except Exception as err:
            return_dict_["explain"] = _explain_str
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            matplotlib.use(current_backend)
            return return_dict_

        finally:
            return_dict_["plot"] = fig
            _explain_str += "Note: Visual inspection is recommended to ensure representativeness. "
            return_dict_["explain"] = _explain_str
            logger.debug("%s Representativeness Test completed with %d data rows for plot", 
                         __s_fn_id__, len(return_dict_['data']))
            matplotlib.use(current_backend)
            return return_dict_
    
    
    ''' Function --- DIMENSIONALITY ---

            author: <samana.thetha@gmail.com
    '''
    @staticmethod
    def dimensionality(data, max_recommended_dim=20):
        """
        Check the dimensionality of the data and provide guidance.
        
        Parameters:
        -----------
        data : array-like
            Data to test, should be 2D (samples x features)
        max_recommended_dim : int
            Maximum recommended dimensionality for standard MCMC
            
        Returns:
        --------
        bool
            True if dimensionality is manageable, False otherwise
        """
        __s_fn_id__ = f"{dataWorkLoads.__name__} function <dimensionality>"

        return_dict_ = {}
        return_dict_["check"] = "dimensionality"
        return_dict_["comply"] = None
        _explain_str = ""

        try:
            if len(data.shape) == 1:
                n_dim = 1
            else:
                n_dim = data.shape[1]
            
            _explain_str += f"Data dimensionality: {n_dim}\n"
            
            if n_dim <= max_recommended_dim:
                _explain_str += f"✓ Dimensionality is manageable for standard MCMC. "
                return_dict_["comply"] = True
            else:
                _explain_str += f"✗ High dimensionality detected. Consider: "
                _explain_str += "Dimensionality reduction techniques, "
                _explain_str += "Specialized MCMC methods for high dimensions, "
                _explain_str += "Hamiltonian Monte Carlo."
                return_dict_["comply"] = False
    
        except Exception as err:
            return_dict_["explain"] = _explain_str
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            return return_dict_

        finally:
            return_dict_["explain"] = _explain_str
            logger.debug("%s Dimensionality Test completed with %s data rows for plot", 
                         __s_fn_id__, str(return_dict_))
            return return_dict_

    
    ''' Function --- CORRELATION STRUCTURE ---

            author: <samana.thetha@gmail.com
    '''
    @staticmethod
    def correlation_structure(data):
        """
        Analyze the correlation structure of the data.
        
        Parameters:
        -----------
        data : array-like
            Data to test, should be 2D (samples x features)
        """

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <correlation_structure>"

        return_dict_ = {}
        return_dict_["check"] = "correlation structure"
        return_dict_["comply"] = None
        return_dict_["plot"] = None
        _explain_str = ""

        current_backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Agg backend doesn't display figures

        try:
            if len(data.shape) == 1:
                return_dict_["data"]=data

                # For 1D data, show autocorrelation
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot_acf(return_dict_["data"], ax=ax, lags=30)
                ax.set_title('Autocorrelation Function')
                acf_values = acf(return_dict_["data"], nlags=30)
                significant_lags = np.sum(
                    np.abs(acf_values[1:]) > 1.96/np.sqrt(len(return_dict_["data"])))
                
                if significant_lags > 5:
                    return_dict_["comply"] = False
                    _explain_str += f"✗ Significant autocorrelation detected at {significant_lags} lags. "
                    _explain_str += f"High autocorrelation may slow MCMC convergence. "
                else:
                    print("✓ Autocorrelation appears manageable")
            else:
                # For multivariate data, show correlation matrix
                df = pd.DataFrame(data)
                return_dict_['data'] = df.corr()

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(return_dict_['data'], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                ax.set_title('Correlation Matrix')
                
                # Count high correlations
                high_corr_count = np.sum(np.abs(np.triu(return_dict_['data'].values, k=1)) > 0.7)
                
                if high_corr_count > 0:
                    return_dict_["comply"] = False
                    _explain_str += f"✗ Found {high_corr_count} pairs of highly correlated variables. "
                    _explain_str += "High correlation may slow MCMC convergence. "
                else:
                    return_dict_["comply"] = True
                    _explain_str += "✓ Correlation structure appears manageable"
    
        except Exception as err:
            return_dict_["explain"] = _explain_str
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            matplotlib.use(current_backend)
            return return_dict_

        finally:
            return_dict_["plot"] = fig
            _explain_str += "Note: Visual inspection is recommended to ensure representativeness. "
            return_dict_["explain"] = _explain_str
            logger.debug("%s Correlation Structure Test completed with %d data rows for plot", 
                         __s_fn_id__, len(return_dict_['data']))
            matplotlib.use(current_backend)
            return return_dict_
    
    ''' Function --- MULTIMODALITY ---

            author: <samana.thetha@gmail.com
    '''
    @staticmethod
    def multimodality(data, bandwidth=0.5, prominence=0.05):
        """
        Check if the distribution appears to be multimodal.
        
        Parameters:
        -----------
        data : array-like
            Data to test
        bandwidth : float
            Bandwidth for kernel density estimation
        prominence : float
            Minimum prominence of peaks to be considered
            
        Returns:
        --------
        bool
            True if multimodal, False otherwise
        """

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <multimodality>"

        return_dict_ = {}
        return_dict_["check"] = "multimodality"
        return_dict_["comply"] = None
        return_dict_["plot"] = None
        _explain_str = ""

        current_backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Agg backend doesn't display figures

        try:
            data_clean = data[~np.isnan(data)].reshape(-1, 1)
            nan_count = len(data) - len(data_clean)
            nan_percentage = 100 * nan_count / len(data) if len(data) > 0 else 0

            if len(data_clean) == 0:
                return_dict_["comply"] = False
                _explain_str +="✗ All data values are NaN. Cannot perform multimodality check. "
                _explain_str +="\nempty_regions= 0, nan_percentage: 100.0, valid_data: FALSE "

                # Create an empty figure if all data is NaN
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "All data values are NaN",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14)
                ax.set_title("Density Estimate NO Detected Modes")
                return_dict_["data"] = None
            else:
                logger.warning("%s Removed %d NaN values for visualization", 
                               __s_fn_id__, (len(data) - len(data_clean)))
                ''' fill NaN with mean '''
                imputer = SimpleImputer(strategy='mean')
                return_dict_["data"] = imputer.fit_transform(data_clean)
    
                # Create a KDE of the data
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot kernel density estimate
                x_grid = np.linspace(
                    np.min(return_dict_["data"]), 
                    np.max(return_dict_["data"]), 1000)
                kde = KernelDensity(bandwidth=bandwidth).fit(return_dict_["data"].reshape(-1, 1))
                log_dens = kde.score_samples(x_grid.reshape(-1, 1))
                density = np.exp(log_dens)

                # Find peaks in the density
                peaks, properties = find_peaks(density, prominence=prominence)
                
                ax.plot(x_grid, density)
                ax.plot(x_grid[peaks], density[peaks], 'ro')
                ax.set_title(f'Density Estimate with {len(peaks)} Detected Modes')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend()

                if len(peaks) > 1:
                    _explain_str += f"✗ Distribution appears to be multimodal with {len(peaks)} modes. "
                    _explain_str += "Multimodal distributions may require: "
                    _explain_str += "(i) Longer MCMC chains, "
                    _explain_str += "(ii) Multiple chains with different starting points, "
                    _explain_str += "(iii) Tempering methods."
                    return_dict_["comply"] = True
                else:
                    _explain_str += "✓ Distribution appears to be unimodal "
                    return_dict_["comply"] = False
    
        except Exception as err:
            return_dict_["explain"] = _explain_str
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            matplotlib.use(current_backend)
            return return_dict_

        finally:
            return_dict_["plot"] = fig
            return_dict_["explain"] = _explain_str
            logger.debug("%s Modality Test completed with %d data rows for plot", 
                         __s_fn_id__, len(return_dict_['data']))
            matplotlib.use(current_backend)
            return return_dict_
    
    ''' Function --- BOUNDARY CONDITION ---

            author: <samana.thetha@gmail.com
    '''
    @staticmethod
    def boundary_conditions(data):
        """
        Check for boundary conditions or constraints in the data.
        
        Parameters:
        -----------
        data : array-like
            Data to test
        """

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <boundary_conditions>"

        return_dict_ = {}
        return_dict_["check"] = "boundary conditions"
        return_dict_["comply"] = True
        return_dict_["plot"] = None
        _explain_str = ""

        try:
            ''' Check for common boundary types '''
            min_val = np.min(data)
            max_val = np.max(data)
            ''' Calculate distance from min/max to next closest points '''
            sorted_data = np.sort(data)
            min_gap = np.min(np.diff(sorted_data))
            ''' Check if data is close to 0, 1, or other common boundaries'''
            boundaries = {
                '0': 0,
                '1': 1,
                'Integer values': np.all(data == np.round(data))
            }
            _explain_str += "Potential boundary conditions: "
            _explain_str +=f"Data range: [{min_val}, {max_val}]. "
            
            for name, value in boundaries.items():
                if name == 'Integer values':
                    if value:
                        return_dict_["comply"]=False
                        _explain_str +=f"✗ Data appears to be discrete (integer-valued) "
                        _explain_str += "Discrete parameters require special MCMC approaches. "
                else:
                    if np.any(np.isclose(data, value, atol=min_gap)):
                        return_dict_["comply"]=False
                        _explain_str += f"✗ Data contains values very close to {name}. "
                        _explain_str += "Boundary constraints may require specialized MCMC. "
            
            ''' Check for potential truncation '''
            if np.isclose(min_val, max_val - (max_val - min_val) * 0.05, atol=min_gap) or \
               np.isclose(max_val, min_val + (max_val - min_val) * 0.05, atol=min_gap):
                return_dict_["comply"]=False
                _explain_str += "✗ Data may be truncated (high concentration at extremes). "
                _explain_str += "Truncated distributions require special handling. "

        except Exception as err:
            return_dict_["explain"] = _explain_str
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            return return_dict_

        finally:
            return_dict_["explain"] = _explain_str
            logger.debug("%s Boundary Conditions Test completed, %s", 
                         __s_fn_id__, str(return_dict_))
            return return_dict_

    
    ''' Function --- MIXING PROPERTY ---

            author: <samana.thetha@gmail.com
    '''
    @staticmethod
    def mixing_properties(data, lag=1):
        """
        For time series data, check properties that might affect mixing.
        
        Parameters:
        -----------
        data : array-like
            Time series data to test
        lag : int
            Lag for checking autocorrelation
        """

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <mixing_properties>"

        return_dict_ = {}
        return_dict_["check"] = "mixing properties"
        return_dict_["comply"] = None
        return_dict_["plot"] = None
        _explain_str = ""

        current_backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Agg backend doesn't display figures
            
        try:
            # current_backend = matplotlib.get_backend()
            # matplotlib.use('Agg')  # Agg backend doesn't display figures

            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate autocorrelation
            acf_1 = acf(data.reshape(-1), nlags=lag)[-1]
            
            # plt.figure(figsize=(10, 6))
            ax.scatter(data[:-lag], data[lag:], alpha=0.5)
            ax.set_title(f'Lag-{lag} Plot (r = {acf_1:.3f})')
            ax.set_xlabel(f'X(t)')
            ax.set_ylabel(f'X(t+{lag})')
            ax.grid(True)
            # plt.show()

            if abs(acf_1) > 0.7:
                return_dict_["comply"]=False
                _explain_str += f"✗ High lag-{lag} autocorrelation detected: {acf_1:.3f}. "
                _explain_str += "High autocorrelation may lead to poor mixing "
                _explain_str += "Consider: "
                _explain_str += "(i) Thinning your MCMC chain, "
                _explain_str += "(ii) Using longer burn-in periods, "
                _explain_str += "(iii) More efficient MCMC samplers (HMC, NUTS). "
            else:
                return_dict_["comply"]=True
                _explain_str += f"✓ Lag-{lag} autocorrelation is moderate: {acf_1:.3f}. "
                _explain_str += "Mixing properties look reasonable. "

        except Exception as err:
            return_dict_["explain"] = _explain_str
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            return return_dict_

        finally:
            return_dict_["plot"] = fig
            return_dict_["explain"] = _explain_str
            logger.debug("%s Mixing properties Test completed, %s", 
                         __s_fn_id__, str(return_dict_))
            return return_dict_

    
    ''' Function --- PRIOR KNOWLEDGE ---

            author: <samana.thetha@gmail.com
    '''
    @staticmethod
    def prior_knowledge(data):
        """
        Evaluate if the data aligns with common distributions
        to help with prior selection.
        
        Parameters:
        -----------
        data : array-like
            Data to test
        """

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <prior_knowledge>"

        return_dict_ = {}
        return_dict_["check"] = "prior knowledge"
        return_dict_["comply"] = None
        return_dict_["plot"] = None
        _explain_str = ""

        try:
            # Normalize data for distribution testing
            data_norm = ((data - np.mean(data)) / np.std(data)).reshape(-1)
            # List of distributions to test
            distributions = [
                ('Normal', stats.norm.cdf),
                ('Student-t 5df', lambda x: stats.t.cdf(x, df=5)),  # Using df=5 as an example
                ('Student-t 10df', lambda x: stats.t.cdf(x, df=10)), # Add multiple df values to test
                # ('Student-t', stats.t),
                ('Cauchy', stats.cauchy.cdf),
                ('Laplace', stats.laplace.cdf),
                ('Logistic', stats.logistic.cdf),
                # ('Poisson', lambda x: stats.poisson.cdf(x, mu=np.mean(x))), # Add multiple df values to test
            ]
            # Perform Kolmogorov-Smirnov tests
            results = []
            for name, dist in distributions:
                statistic, p_value = stats.kstest(data_norm, dist)
                results.append((name, p_value, statistic))
            # Sort by p-value (higher is better fit)
            results.sort(key=lambda x: x[1], reverse=True)

            _explain_str += "Distribution fitting results (higher p-value = better fit): "
            i=0
            for name, p_value, statistic in results:
                i+=1
                _explain_str += f"({i}) {name}: p-value = {p_value:.4f}, KS statistic = {statistic:.4f}. "
            
            # Suggest priors based on best fit
            best_dist = results[0][0]
            _explain_str += f"\nBased on your data, consider priors related to the {best_dist} distribution. "
            
            if results[0][1] < 0.05:
                return_dict_["comply"]=False
                _explain_str += "✗ No common distribution fits well (all p-values < 0.05). "
                _explain_str += "Consider non-parametric or mixture priors. "
            else:
                return_dict_["comply"]=True
                _explain_str += f"✓ {best_dist} distribution is a reasonable fit (p-value > 0.05) "
    
        except Exception as err:
            return_dict_["explain"] = _explain_str
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            return return_dict_

        finally:
            return_dict_["explain"] = _explain_str
            logger.debug("%s Prior knowledge Test completed, %s", 
                         __s_fn_id__, str(return_dict_))
            return return_dict_

    
    ''' Function --- NOISE CHARACTERISTICS ---

            author: <samana.thetha@gmail.com
    '''
    @staticmethod
    def noise_characteristics(data):
        """
        Analyze noise characteristics in the data.
        
        Parameters:
        -----------
        data : array-like
            Data to test
        """

        __s_fn_id__ = f"{dataWorkLoads.__name__} function <noise_characteristics>"

        return_dict_ = {}
        return_dict_["check"] = "noise characteristics"
        return_dict_["comply"] = None
        return_dict_["plot"] = None
        _explain_str = ""

        current_backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Agg backend doesn't display figures
            
        try:
            # Calculate differences to examine noise
            diffs = np.diff(data.reshape(-1))
            
            # Test for heteroskedasticity (changing variance)
            window_size = min(100, len(diffs) // 5)
            rolling_std = pd.Series(diffs).rolling(window=window_size).std().dropna()
            # Test for heteroskedasticity
            std_ratio = np.max(rolling_std) / np.min(rolling_std)

            if std_ratio > 2:
                return_dict_["comply"]=False
                _explain_str += f"✗ Heteroskedasticity detected (std ratio: {std_ratio:.2f}). "
                _explain_str += "Consider: "
                _explain_str += "(i) Transformation to stabilize variance, "
                _explain_str += "(ii) MCMC methods that account for varying noise. "
            else:
                return_dict_["comply"]=True
                _explain_str += f"✓ Noise appears homoskedastic (std ratio: {std_ratio:.2f}). "
            
            # Test for normality of noise
            _, p_value = stats.normaltest(diffs)
            
            if p_value < 0.05:
                return_dict_["comply"]=False
                _explain_str += f"✗ Noise is not normally distributed (p-value: {p_value:.4f}). "
                _explain_str += "  Consider MCMC likelihood functions that match your noise distribution. "
            else:
                return_dict_["comply"]=True
                _explain_str += f"✓ Noise appears normally distributed (p-value: {p_value:.4f}). "

            # Create a single figure with 4 subplots
            fig = plt.figure(figsize=(10, 12))  # Make the figure taller to accommodate 4 plots
            
            # Plot 1: Original data
            ax1 = fig.add_subplot(4, 1, 1)
            ax1.plot(data)
            ax1.set_title('Original Data')
            ax1.grid(True)
            
            # Plot 2: Differences
            ax2 = fig.add_subplot(4, 1, 2)
            ax2.plot(diffs)
            ax2.set_title('First Differences (Noise)')
            ax2.grid(True)
            
            # Plot 3: Rolling standard deviation
            ax3 = fig.add_subplot(4, 1, 3)
            ax3.plot(rolling_std)
            ax3.set_title('Rolling Standard Deviation of Noise')
            ax3.grid(True)
            
            # Plot 4: QQ plot to check noise normality
            ax4 = fig.add_subplot(4, 1, 4)
            stats.probplot(diffs, dist="norm", plot=ax4)
            ax4.set_title('Q-Q Plot of Noise')
            ax4.grid(True)
            
            # Adjust spacing between subplots
            plt.tight_layout()
            
            # Store the figure in the return dictionary
            # return_dict_["plot"] = fig
        except Exception as err:
            return_dict_["explain"] = _explain_str
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)
            return return_dict_

        finally:
            return_dict_["explain"]=_explain_str
            return_dict_["plot"] = fig
            # return_dict_["plot"] = fig
            logger.debug("%s Noise Characteristics Test completed, %s", 
                         __s_fn_id__, str(return_dict_))
            return return_dict_

    
    def run_checks(
        self,
        data : DataFrame = None, 
        checks:list = None,
        multivariate:bool=False,
        time_series :bool=True
    ) -> list:
        """
        Run all MCMC suitability checks on the data.
        
        Parameters:
        -----------
        data : array-like
            Data to test
        multivariate : bool
            Whether the data is multivariate (2D array)
        time_series : bool
            Whether the data should be treated as time series
        """

        __s_fn_id__ = f"{self.__name__} function <run_checks>"

        try:
            ''' validate checks '''
            self.checks = checks
            ''' validate and convert dataframe to array '''
            self.data = data
            ''' convert datetime to string '''
            
            data = np.asarray(self.data.toPandas())
            # Check for NaN values
            nan_mask = np.isnan(data)
            has_nans = np.any(nan_mask)
            if has_nans:
                print(f"Warning: Data contains NaN values. Handling them appropriately in each test.")
                logger.warning("%s Data contains NaN values. Handling them appropriately in each test."
                               , __s_fn_id__)
                
            # Ensure data is in the right format
            if multivariate and len(data.shape) == 1:
                data = data.reshape(-1, 1)
                logger.warning("%s Data was reshaped to 2D for multivariate analysis",__s_fn_id__)

            ''' Run checks '''
            results = []
            ''' 1. STATIONARITY '''
            if "stationarity" in checks and time_series:
                print("running STATIONARITY check ...")
                results.append(dataWorkLoads.stationarity(
                    data if len(data.shape) == 1 else data[:, 0],
                    multivariate,))
            elif "stationarity" in checks and not time_series:
                print("Skipped stationarity (not time series data)")
                logger.warning("Skipped stationarity (not time series data)")
            else:
                pass
            ''' 2. DATA VOLUME '''
            if "data volume" in checks:
                print("running DATA VOLUME check ...")
                results.append(dataWorkLoads.data_volume(data=data))
            ''' 3. REPRESENTATIVENESS '''
            if "representativeness" in checks:
                print("running REPRESENTATIVENESS check ...")
                results.append(
                    dataWorkLoads.representativeness(data=data, 
                    multivariate=multivariate))
            ''' 4. DIMENSIONALITY '''
            if "dimensionality" in checks:
                print("running DIMENSIONALITY check ...")
                results.append(
                    dataWorkLoads.dimensionality(data, max_recommended_dim=20))
            ''' 5. CORRELATION STRUCTURE '''
            if "correlation structure" in checks:
                print("running CORRELATION STRUCTURE check ...")
                results.append(
                    dataWorkLoads.correlation_structure(data))
            ''' 6. MULTIMODALITY '''
            if "multimodality" in checks:
                print("running MULTIMODALITY check ...")
                results.append(
                    dataWorkLoads.multimodality(data, bandwidth=0.5, prominence=0.05))
            ''' 7. BOUNDARY CONDITIONS '''
            if "boundary conditions" in checks:
                print("running BOUNDARY CONDITIONS check ...")
                results.append(
                    dataWorkLoads.boundary_conditions(data))
            ''' 8. MIXING PROPERTY '''
            if "mixing properties" in checks:
                print("running MIXING PROPERTY check ...")
                results.append(
                    dataWorkLoads.mixing_properties(data, lag=1))
            ''' 9. PRIOR KNOWLEDGE '''
            if "prior knowledge" in checks:
                print("running PRIOR KNOWLEDGE check ...")
                results.append(
                    dataWorkLoads.prior_knowledge(data))
            ''' 10. NOISE CHARACTERISTICS '''
            if "noise characteristics" in checks:
                print("running NOISE CHARACTERISTICS check ...")
                results.append(
                    dataWorkLoads.noise_characteristics(data))

            
            # print("\n==== 6. MULTIMODALITY CHECK ====")
            # if multivariate:
            #     print("For multivariate data, checking first PCA component")
            #     check_multimodality(data_pca)
            # else:
            #     check_multimodality(data)
            
            # print("\n==== 7. BOUNDARY CONDITIONS CHECK ====")
            # if multivariate:
            #     print("Checking each dimension separately:")
            #     for i in range(data.shape[1]):
            #         print(f"\nDimension {i+1}:")
            #         check_boundary_conditions(data[:, i])
            # else:
            #     check_boundary_conditions(data)
            
            # print("\n==== 8. MIXING PROPERTIES CHECK ====")
            # if time_series:
            #     if multivariate:
            #         print("Checking first dimension for mixing properties:")
            #         check_mixing_properties(data[:, 0])
            #     else:
            #         check_mixing_properties(data)
            # else:
            #     print("Skipped (not time series data)")
            
            # print("\n==== 9. PRIOR KNOWLEDGE CHECK ====")
            # if multivariate:
            #     print("Checking each dimension separately:")
            #     for i in range(min(3, data.shape[1])):  # Limit to first 3 dimensions
            #         print(f"\nDimension {i+1}:")
            #         check_prior_knowledge(data[:, i])
            # else:
            #     check_prior_knowledge(data)
            
            # print("\n==== 10. NOISE CHARACTERISTICS CHECK ====")
            # if time_series:
            #     if multivariate:
            #         print("Checking first dimension for noise characteristics:")
            #         check_noise_characteristics(data[:, 0])
            #     else:
            #         check_noise_characteristics(data)
            # else:
            #     print("Skipped (not time series data)")
            print("\nFinished running checks!")

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        finally:
            logger.debug("%s completed %d checks to return results", 
                         __s_fn_id__, len(results))
            return results
