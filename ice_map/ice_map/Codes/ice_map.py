import numpy as np
import pandas as pd
from scipy import stats
import os
import subprocess
from pyextremes import EVA
from pykrige.ok import OrdinaryKriging
from IPython.display import clear_output
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from SALib.analyze import sobol
from SALib.sample import saltelli
import alphashape
from shapely.geometry import Polygon, Point
from copy import deepcopy
import torch
from torch import nn
from torch import optim
import optuna
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings
import geopandas as gpd
from tqdm.notebook import tqdm as bar
from time import time, sleep


def check_for_versions():
    """
    This function is basically responsible for checking your Python directory
    and see whether you have the required version of packages, which are necessary
    to run this code or not. If you don't have it, it will be download and install them all.
    """
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
    clear_output()
    return None

class dataset():
    def __init__(self):
        """
        This is the basic class which forms the directory for reading cleaned dataset.
        Before running the code it is necessary to make it.
        """
        directory = os.getcwd()
        loc = directory.find('Codes')
        codes_dir = directory
        data_dir = directory[:-5]
        data_dir = os.path.join(data_dir, "Data")
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"The folder data does not exist in the current directory.")
        
        env_data_dir = os.path.join(data_dir, "processed_clean.csv")
        self.env_data_dir = env_data_dir

    def read_df(self):
        """
        This function reads the retrieved dataset from ASOS website. 
        **** THIS FUNCTION DOESN'T DO ANY PREPROCESSING SUCH AS DEALING WITH NULL VALUES OR etc. ****
        """
        self.df = pd.read_csv(self.env_data_dir,
                              parse_dates=['valid'],
                              date_format= '%Y-%m-%d %H:%M:%S',
                             low_memory = False,
                             index_col = None)
        # Sometimes, the pandas add an index column
        if self.df.shape[1]>8:
            self.df = self.df.iloc[:, 1:]
            
    def process_df(self, verbose = 0):
        """
        This function perfoms some adjustments in the imported data.
        verbose controls the level of output:
        0. prints nothing
        1. prints only first 5 rows.
        2. prints all table.
        """
        df = self.df.copy()
        # Excluding records without precipitation
        df = df.loc[df.p01m > 0]
        
        #adjustment of records with negative elevation
        df.loc[df.elevation <=0, 'elevation'] = 0
        
        #unit conversion
        df.sped *= 0.44704 #mph to m/sec
        
        #Adjustment of base height of anonemeter - based height of ASOS anemometer is 27ft.
        df.sped *= (32.8084/33)**(1/7)
        
        # Adjustment of records without wind speed
        df.loc[df.sped == 0, 'sped'] = 0.01
        
        # Applying restrictions for ambient temperature, wind speed, and precipitatoin range during freezing rain events
        # Limits:
        # Ambient temperature is in range in [-20, 1]
        # wind speed less than 20 m/sec.
        # cumulative precipitation is less than 1 cm.
        
        df = df.loc[(df.tmpc<=1)&
                     (df.tmpc>=-20)&
                     (df.p01m <= 10)&
                     (df.sped <= 20)]
        self.df = df
        if verbose == 2:
            display(self.df)
        elif verbose == 1:
            display(self.df.head(n=5))
        elif verbose == 0:
            pass
        else:
            raise ValueError(f'verbose can be [0, 1, 2] not {verbose}')


    def ice_cal(self, ice_model = 1, plot = False):
        """
        It computes total hourly ice accretion based on empirical models
        ice_model: int
           Values
                  1: jones (CRREL)
                  2: Goodwin
                  3: Chaine and Castonguay
                  4: Sanders (ILR)
        plot: bool
            plots hourly ice accretion and wind speed across the map.
                             
        """
        self.ice_model = ice_model
        if ice_model == 1:
            #Calculation of ice accretion using Jones model
            # ASsumptions:
            #       1. Precipitaion is freezing rain and there is no sleet.
            #       2. The cable is strong enough to tolerate ice's weight during accretion process.
            #       3. Ice type is glaze with a density of 0.9 gr/cm3.
            
            # liquid water content
            # This is Best model (1956). It is used as Professor Jones states this model is better than
            # Marshall equation.
            W = 0.067*self.df.p01m**0.846
            
            ro_ice = 0.9
            ro_drop = 1.0
            Req = 1/(np.pi* ro_ice)*np.sqrt((self.df.p01m*ro_drop)**2+
                                            (3.6*self.df.sped*W)**2)
            self.df['ice'] = Req
            
        elif ice_model == 2:
            #Calculation of ice accretion using Goodwin model
            # ASsumptions:
            #       1. All the drops collected freeze on the wire.
            #       2. The cable is strong enough to tolerate ice's weight during accretion process.
            #       3. Ice type is glaze with a density of 0.9 gr/cm3.
            #       4. The drop is pure liquid and there is no sign of sleet.
            

            # lambda function for computing the fall speed of drop (Rogers and Yau 1989)
            f = lambda x: 8*x if x<0.6 else (201*(x/1000)**0.5 if 0.6<=x<= 2.0 else np.NaN)
            
            # Vectorization of the lambda function for element-wise comparison
            f = np.vectorize(f)
            
            # Marshall-Palmer drop size distribution
            r0 = 1.835/(4.1*self.df.p01m.values**-0.21)

            # hourly mean wind speed
            Vd = f(r0)
            
            # depth of liquid precipitation (Hg).
            # As the period of assessment is considered to be one hourly, Hg can be 
            # assumed same as the rate of precipitation (mm/hr * 1hr = mm)
            
            Hg = self.df.p01m.copy()            
            # Density of water drop (gr/cm3).
            ro_w = 1.0
            # Density of accreted ice.
            ro_ice = 0.9
            Req = (ro_w * Hg)/(ro_ice * np.pi)*np.sqrt(1+(self.df.sped/Vd)**2)
            self.df['ice'] = Req
            
        elif ice_model == 3:
            #Calculation of ice accretion using Chaine and Castonguay model.
            # computing correction factor 
            k = []
            for row in range(self.df.shape[0]):
                # Sutherland formula
                
                # convert Celsius to Kelvin
                tmpc_kelvin = self.df.iloc[row, 5] + 273.15
                #Sutherland constant!
                S = 110.4e3 
                # viscosity of air:
                mu = 1.81e-5*(tmpc_kelvin/273.15)**1.5*(S/(tmpc_kelvin + S))
                
                ro_i = 1 
                # density of pure liquid
                r_drop = (15e-6)*1000/2 # radius of drop (d = 15 micrometer)! 
                speed_in_cm_per_second = self.df.iloc[row, 7]
                lambda_s = 2/9*ro_i*r_drop**2/mu*speed_in_cm_per_second
                r_rod = 2.5 # 25 milimeter
                ki = lambda_s/r_rod #doi:10.1016/b978-0-08-009362-8.50021-1  
                k.append(ki)
                
            k = np.array(k)
            #Calculation of ice accretion using Chaine and Castonguay model
            # ASsumptions:
            #       1. Precipitaion is freezing rain and there is no sleet.
            #       2. The cable is strong enough to tolerate ice's weight during accretion process.
            #       3. Ice type is glaze with a density of 0.9 gr/cm3.
            #       4. All impinging drops freeze on the wire.
            #       5. Radius of iced rod is assumed to be 25 mm, and there is no ice layer on the
            #         rod at the beginning of ice accretion period. So, ΔR = R_t1 - R_t0 = R_t1 - 0
            
            
            # The amount of ice accretion on a vertical surface
            Tv = 4.4316*self.df.sped*(self.df.p01m/25.4)**0.88
            R = 25
            # The amount of ice accretion on a horizontal surface
            Th = self.df.p01m
            
            Req = np.sqrt(k*R/2*np.sqrt(Tv**2 + Th**2) + R**2) - R
            self.df['ice'] = Req
        elif ice_model == 4:
            #Calculation of ice accretion using Sanders.
            #unit conversion
            # mm/h to in/hr
            p01m = self.df.p01m*0.0393701
            # m/sec to kt
            sped = self.df.sped*1.94384
            # computing Wet bulb temperature based on ambient temperature and relative humidity (Stull approximation)
            # 10.1175/JAMC-D-11-0143.1
            T = self.df.tmpc
            RH = self.df.relh
            term1 = T*np.arctan(0.151977*np.sqrt(RH + 8.313659))
            term2 = np.arctan(T + RH)
            term3 = np.arctan(RH - 1.676331)
            term4 = 0.00391838 * RH**(3/2) * np.arctan(0.023101 * RH)
            term5 = 4.686035
            Tw = term1 + term2 - term3 + term4 - term5

            ILR_p = 0.1395*(p01m**-0.541)
            ILR_tw = -0.0071*(Tw**3) - 0.1039*(Tw**2) -0.3904*Tw+ 0.5545
            ILR_v = 0.0014*(sped**2) + 0.0027*sped + 0.7574
            
            ILR_values = np.zeros_like(Tw)
            cond1 = Tw > -0.35
            cond2 = np.logical_and(Tw <= -0.35, sped > 12)
            cond3 = np.logical_and(Tw <= -0.35, sped <= 12)
            
            ILR_values[cond1] = (0.70 * ILR_p[cond1]) * (0.29 * ILR_tw[cond1]) * (0.01 * ILR_v[cond1])
            ILR_values[cond2] = (0.73 * ILR_p[cond2]) * (0.01 * ILR_tw[cond2]) * (0.26 * ILR_v[cond2])
            ILR_values[cond3] = (0.79 * ILR_p[cond3]) * (0.20 * ILR_tw[cond3]) * (0.01 * ILR_v[cond3])
            
            # ice accumulation on an elevated horizontal surface
            Ti = ILR_values*self.df.p01m
            #ice in radial form
            Req = 0.394*Ti
            # inch to mm
            Req *= 25.4
            self.df['ice'] = Req

        if plot:
            self.plot_ice_wind()
            
    def plot_ice_wind(self):
        # Maximum Ice Accretion for Each Station
        max_precipitation_per_station = self.df.groupby(['lon', 'lat'])['ice'].max().reset_index()
        if self.ice_model == 1:
            ice_model_name = 'CRREL'
        elif self.ice_model == 2:
            ice_model_name = 'Goodwin'
        elif self.ice_model == 3:
            ice_model_name = 'Chaine'
        elif self.ice_model == 4:
            ice_model_name = 'Sanders'
        Title = 'Maximum Observed Hourly Precipitation (mm) per Station [' + ice_model_name +']'
        # Plot Maximum Precipitation for Each Station
        fig_max_precipitation = px.scatter_mapbox(
            max_precipitation_per_station,
            lat='lat',
            lon='lon',
            size='ice',
            color='ice',
            size_max=20,
            zoom=2,
            mapbox_style='carto-positron',
            title=Title,
            color_continuous_scale="greens"  
        )
        fig_max_precipitation.update_layout(margin=dict(l=0, r=0, t=40, b=0),
                                           height=600,  
                                            width=1_100,
                                           title_x = 0.45) 
        
        
        # Show the plot
        fig_max_precipitation.show()
        
        # Maximum Wind Speed for Each Station
        max_wind_speed_per_station = self.df.groupby(['lon', 'lat'])['sped'].max().reset_index()
        
        # Plot Maximum Wind Speed for Each Station
        fig_max_wind_speed = px.scatter_mapbox(
            max_wind_speed_per_station,
            lat='lat',
            lon='lon',
            size='sped',
            color='sped',
            size_max=20,
            zoom=2,
            mapbox_style='carto-positron',
            title='Maximum Reported Hourly Wind Speed(m/sec) per Station'.title(),
                color_continuous_scale="blues"
        )
        fig_max_wind_speed.update_layout(margin=dict(l=0, r=0, t=40, b=0),
                                           height=600,  
                                            width=1_100,
                                            title_x = 0.45)    
        
        # Show the plot
        fig_max_wind_speed.show()

    def Ex_Va_An(self, plot = True, n_sample =10, n_year = 50):
        unique_stations = self.df.station.unique()
        num_stations = len(unique_stations)
        assert n_year>0, f'return period ({n_year})<0'
        if not isinstance(n_year, int):
            n_year = int(n_year)
        self.n_year = n_year
        """
        Performs EVA (BM) for the dataset
        The input of the function is the processed environmental file which was read before.
        The output of the function will be the maximum ice accretion with ith return period and 
        the fitted distributions.
        """
        # unique stations
        output = np.array([])
        dists = []
        dists_params = []
        for counter_rows, station in bar(enumerate(unique_stations), total=num_stations):
            
            station_data = self.df.loc[self.df.station == station].copy()
            
            # exclusion of stations with only one observation.
            if station_data.shape[0]>1:
                # extraction of longitude, latitude, and elevation of the station.
                lon, lat,elev = station_data.iloc[0,2], station_data.iloc[0,3], station_data.iloc[0,4]
            
                # EVA for each station
                try:
                
                    station_data = station_data[['valid','ice']]
    
                    station_data.set_index('valid',
                                           inplace = True)
                    
                    station_data = station_data.squeeze()
                    
                    model = EVA(station_data)
                    
                    model.get_extremes(
                        method="BM",
                        extremes_type="high",
                        block_size="365.2425D",
                        errors="ignore",
                    )
                
                    # Get max values for different return periods (50 and 500 yrs).
                    model.fit_model()
    
                    summary = model.get_summary(
                    return_period=[n_year],
                    alpha=0.95,
                    n_samples=n_sample)
                    summary = summary.iloc[:, 0]
                    ice_50 = summary[n_year]
                    
                    model_name = model.distribution.name
                    # ignoring of ilogical vlaues (maximum ice accretion in ice storm 1998 was 80 mm)
                    # so we chose 10 cm as the upper limit.
                    if ice_50 <=100:
                        Temp = np.empty(shape = [1, 4])
                        Temp[0, 0] = lon
                        Temp[0, 1] = lat
                        Temp[0, 2] = elev
                        Temp[0, 3] = ice_50
                        model_instance = getattr(stats, model_name)
                        model_instance = model_instance(model.distribution.mle_parameters.values())
    
                        dists.append(model.distribution.name)
                        dists_params.append(model.distribution.mle_parameters.values())
                        
                        if (counter_rows == 0)|(output.shape[0] == 0):
                            output = Temp.copy()
                        elif counter_rows >0:
                            
                            output = np.concatenate([output, Temp],
                                                axis=0)
                except (ZeroDivisionError, TypeError) as e:
                    print(e)
                    pass
            
        output = pd.DataFrame(output,
                              columns=['lon',
                                       'lat',
                                       'elev',
                                       'ice_50'])
        
        self.extreme = output
        self.extreme_ice_dist = dists
        self.extreme_ice_dist_params = dists_params
        clear_output()
        if plot:
            if self.ice_model == 1:
                model_name = 'jones'
            elif self.ice_model == 2:
                model_name = 'Goodwin'
            elif self.ice_model == 3:
                model_name = 'Chaine'
            elif self.ice_model == 4:
                model_name = 'Sanders'
            else:
                model_name = 'Unknown'
                
            title = f'{self.n_year:.0f}-year return period extreme ice [' +model_name +']'
            fig_ice = px.scatter_mapbox(output,
                                             lat='lat',
                                             lon='lon',
                                             size='ice_50',
                                             color='ice_50',
                                             color_continuous_scale='Viridis',  # Add color scale
                                             mapbox_style='open-street-map',
                                             title=title.title(),
                                             size_max=20,
                                             zoom=3,
                                             width=1_100,
                                             height=800)
            
            fig_ice.update_layout(mapbox=dict(center={'lat': 60, 'lon': -95},
                                                   zoom=3),
                                                   title_x=0.5)
            
            # Display the plot
            fig_ice.show()
    def inter_visualize(self, estimates, uncertainties, title, lon_min, lon_max, lat_min, lat_max, lon_stations, lat_stations):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,5))
        
        # Use a colormap for better visualization
        im1 = ax1.imshow(estimates, 
                       extent=[lon_min, lon_max,
                               lat_min, lat_max],
                       origin='lower',
                       cmap='GnBu',
                        vmin = 0,
                        aspect='auto')
        
        _ = ax1.scatter(lon_stations,
                        lat_stations,
                        marker = 'x',
                        c = 'k',
                       s = 1,
                       label = 'Main stations')
        
        # Add colorbar with adjusted size
        cbar = plt.colorbar(im1, 
                            ax=ax1,  
                            fraction=0.02, 
                            pad=0.05)
        cbar.ax.tick_params(labelsize=8)
        # Set axis labels
        ax1.set_xlabel('Longitude',
                      font = 'times new roman',
                      fontsize = 10)
        ax1.set_ylabel('Latitude',
                      font = 'times new roman',
                      fontsize = 10)
        
        # Add a title
        ax1.set_title('Estimations',
                      font = 'times new roman',
                      fontsize = 15)
        ax1.tick_params(axis='both',
                        which='both', 
                        labelsize=8)
        im2 = ax2.imshow(uncertainties**0.5, 
                       extent=[lon_min, lon_max,
                               lat_min, lat_max],
                       origin='lower',
                       cmap='GnBu',
                        vmin = 0,
                        aspect='auto')
        
        _ = ax2.scatter(lon_stations,
                        lat_stations,
                        marker = 'x',
                        c = 'k',
                       s = 1,
                       label = 'Main stations')
        
        # Add colorbar with adjusted size
        cbar = plt.colorbar(im2, 
                            ax=ax2,
                            fraction=0.02, 
                            pad=0.05)
        
        cbar.ax.tick_params(labelsize=8)
        # Set axis labels
        
        ax2.set_xlabel('Longitude',
                      font = 'times new roman',
                      fontsize = 10)
        ax2.set_ylabel('Latitude',
                      font = 'times new roman',
                      fontsize = 10)
        
        ax2.set_title('Uncertainties',
                      font = 'times new roman',
                      fontsize = 15)
        
        ax2.tick_params(axis='both',
                        which='both', 
                        labelsize=8)
    
        variogram_name = 'Variogram = [' + title.capitalize() + ']'
        plt.suptitle(variogram_name,
                     fontname='times new roman',
                     fontsize=20,
                    c= 'g')
        _ = plt.show()

    def spatial_inter2D(self, variogram = None, mesh_size = 0.1, plot = False):
        # extracting data corresponding the stations
        lat_stations, lon_stations, ice_50yr = self.extreme.lat, self.extreme.lon, np.round(self.extreme.ice_50,1)

        #forming the rectangular area
        lat_min, lat_max = self.extreme.lat.min(), self.extreme.lat.max()
        lon_min, lon_max = self.extreme.lon.min(), self.extreme.lon.max()

        self.grid_y = np.arange(lat_min,
                                lat_max,
                                step = mesh_size)

        self.grid_x = np.arange(lon_min,
                                lon_max,
                                step = mesh_size)

        # if the user doesn't know what's the best variogram, the code checks all variograms and plots the results. So, he can determine the best one.
        if variogram == None:
            list_of_variograms = ['gaussian',
                  'exponential',
                  'spherical',
                  'linear',
                  'power']
            for variogram in list_of_variograms:
                # forming the Kriging model
                OK = OrdinaryKriging(
                    lon_stations,
                    lat_stations,
                    ice_50yr,
                    variogram_model=variogram,
                    verbose=False,
                    enable_plotting=False,
                )
                
                # Perform Ordinary Kriging
                estimates, uncertainties = OK.execute("grid", self.grid_x, self.grid_y)
                self.inter_visualize(estimates.data,
                                     uncertainties,
                                     variogram,
                                     lon_min,
                                     lon_max,
                                     lat_min,
                                     lat_max,
                                     lon_stations,
                                     lat_stations)
        else:
            OK = OrdinaryKriging(
                    lon_stations,
                    lat_stations,
                    ice_50yr,
                    variogram_model=variogram.lower(),
                    verbose=False,
                    enable_plotting=False,
                )
            estimates, uncertainties = OK.execute("grid", self.grid_x, self.grid_y)
            self.estimates = estimates
            if plot:
                self.inter_visualize(estimates.data,
                                     uncertainties,
                                     variogram,
                                     lon_min,
                                     lon_max,
                                     lat_min,
                                     lat_max,
                                     lon_stations,
                                     lat_stations)
                
    def resid_model(self, neglect_resid=True):
        lat_min, lat_max = self.extreme.lat.min(), self.extreme.lat.max()
        lon_min, lon_max = self.extreme.lon.min(), self.extreme.lon.max()
        if not neglect_resid:
                # compute and print the residuals
                print('  ***computing residuals in stations***\n---------------------------------------------'.title())
                print(f"{'Observations':^15}{'Estimated':^15}{'Residual':^15}")
                print(f"{'------------':^15}{'---------':^15}{'--------':^15}")
                residuals = []
                target = pred = []
                for row in range(self.extreme.shape[0]):
                    lon_ind = np.argmin(np.abs(np.subtract(self.extreme.iloc[row, 0], self.grid_x)))
                    lat_ind = np.argmin(np.abs(np.subtract(self.extreme.iloc[row, 1], self.grid_y)))
                    target.append(self.extreme.iloc[row, 3])
                    pred.append(self.estimates[lat_ind, lon_ind])
                    resd_station = self.extreme.iloc[row, 3] - self.estimates[lat_ind, lon_ind]
        
                    # Print values in three columns with fixed width and centered alignment
                    print(f'{self.extreme.iloc[row, 3]:^15.2f}{self.estimates[lat_ind, lon_ind]:^15.2f}{resd_station:^15.2f}')
                
                    residuals.append(resd_station)
                self.mse_resd = mse(target, pred)
                self.mae_resd = mae(target, pred)
            
            
                df_res = deepcopy(self.extreme.iloc[:, :3])
                df_res['residuals'] = residuals
        
                #sensitivity analysis
                # developing a predictive model
                X = df_res.iloc[:, :-1].to_numpy()
                y = df_res.iloc[:, -1].to_numpy()
                scaler_X = MinMaxScaler()
                X_scaled = scaler_X.fit_transform(X)
                scaler_y = MinMaxScaler()
                y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled,
                                                                    test_size=0.15, 
                                                                    random_state=123)
                model = tf.keras.Sequential([
                tf.keras.layers.Dense(8, activation='linear', input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
                model.fit(X_train, y_train,
                          epochs=100,
                          batch_size=32,
                          validation_split=0.15,
                          verbose=0)
                # now, the model is fitted and is ready for performing sensitivity analysis.
                # Calculation of Sobol index
                np.random.seed(123)
                problem = {
                    'num_vars': 3,
                    'names': ['lon', 'lat', 'elev'],
                    'bounds': [[df_res['lon'].min(), df_res['lon'].max()],
                       [df_res['lat'].min(), df_res['lat'].max()],
                       [df_res['elev'].min(), df_res['elev'].max()]]
                        }
                # sampling
                param_values = saltelli.sample(problem, 1_024)
                # prediction
                Xi = scaler_X.transform(param_values)
                pred = model.predict(Xi,
                                     verbose= 0)
                pred = scaler_y.inverse_transform(pred.reshape([-1, 1]))
                Si = sobol.analyze(problem, pred.reshape(-1, ),
                           print_to_console=False)
                variable_names = problem['names']   
                sleep(2)
                clear_output()        
                print(f'*****     Spatial Interpolation    *****')
                print(f'     MSE: {self.mse_resd:.2f}')
                print(f'     MAE: {self.mae_resd:.2f}\n\n')
                print(f"     MAX: {df_res['residuals'].max():.2f}")
                print(f"     MIN: {df_res['residuals'].min():.2f}")
                print(f'***** Sensitivity Analysis Results *****')
                print(f'       total Sobol index')
                # Print each value from arr along with the corresponding value from arr1
                for name, value in zip(variable_names, Si['ST']):
                    print(f'\t{name:<4}: {value:.6f}')
        
                #keep columns where their corresponding Sobol index is higher than 1%.
                threshold = 0.01
                columns_to_keep = Si['ST']>=threshold
                X_remained = X[:, columns_to_keep]
                
                
                if X_remained.shape[1]<1:
                    raise ValueError(f'Fitting cannot proceed as its shape: {X_scaled.shape}')
                scaler_Xi = MinMaxScaler()
                X_remained_scaled = scaler_Xi.fit_transform(X_remained)
                print(f'shape of Input: {X_remained_scaled.shape}')
                print(f'****************************************')
                
                print(f'fitting an ANN to the residuals: * ', end = '')
                np.random.seed(123)        
                X_train, X_val, y_train, y_val = train_test_split(X_remained_scaled, y,
                                                                  test_size=0.15,
                                                                  random_state=123)
        
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        
                class NeuralNetwork(nn.Module):
                    def __init__(self, input_dim, hidden_dim1):
                        super(NeuralNetwork, self).__init__()
                        self.fc1 = nn.Linear(input_dim, hidden_dim1)
                        self.fc2 = nn.Linear(hidden_dim1, 1)
                        
                    def forward(self, x):
                        x = torch.relu(self.fc1(x))
                        x = self.fc2(x)
                        return x
                def objective(trial):
                    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
                    hidden_dim1 = trial.suggest_int("hidden_dim1", 4, 64, log=True)
                    
                    model = NeuralNetwork(input_dim=X_remained_scaled.shape[1], hidden_dim1=hidden_dim1)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.MSELoss()
                    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.unsqueeze(1))
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                    
                    epochs = 300
                    for epoch in range(epochs):
                        model.train()
                        for batch_x, batch_y in train_loader:
                            optimizer.zero_grad()
                            outputs = model(batch_x)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                            
                            # Add PyTorchLightningPruningCallback to perform pruning
                            trial.report(loss.item(), step=epoch)
                            if trial.should_prune():
                                raise optuna.TrialPruned()
                        
                    # Validation loss
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor.unsqueeze(1))
                        
                    return val_loss.item()
                
                study = optuna.create_study(direction="minimize")
                warnings.filterwarnings('ignore')
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                study.optimize(objective, n_trials=50, n_jobs=-1)
                
                # Get the best hyperparameters
                best_params = study.best_params
                best_lr = best_params["lr"]
                best_hidden_dim1 = best_params["hidden_dim1"]
                print(' * ', end = '')
                sleep(1)
                
                # fitting the optimal network
                train_losses = []
                val_losses = []
                
                
                best_model = NeuralNetwork(input_dim=X_remained_scaled.shape[1], hidden_dim1=best_hidden_dim1)
                best_optimizer = optim.Adam(best_model.parameters(), lr=best_lr)
                criterion = nn.MSELoss()
                
                epochs = 300
                for epoch in range(epochs):
                    best_optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = best_model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor.unsqueeze(1))
                    
                    # Backward pass and optimization
                    loss.backward()
                    best_optimizer.step()
                    
                    # Store training loss
                    train_losses.append(loss.item())
                    
                    # Validation loss
                    with torch.no_grad():
                        val_outputs = best_model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor.unsqueeze(1))
                        val_losses.append(val_loss.item())
                print(' * ', end= '')
                sleep(1)
                print(f'\n****************************************')
            
                print(f'Computing residuals across map:  * ', end = '')
                # forming inputs for copmuting residuals across the map
                row_size, col_size = self.estimates.shape
                Inputs_resd = np.array([])
                if Si['ST'][0]>threshold:
                    Inputs_resd = np.tile(self.grid_x, row_size).flatten()
                    Inputs_resd = Inputs_resd.reshape(-1, 1)
                if Si['ST'][1]>threshold:
                    if Inputs_resd.shape[0] == 0:
                        Inputs_resd = np.tile(self.grid_y, col_size).flatten()
                    else:
                        Inputs_resd = np.column_stack([Inputs_resd,
                                                      np.tile(self.grid_y, col_size).flatten()])
                        
                if Si['ST'][-1]>threshold:
                    elevations = self.extreme.elev
                    variogram = 'spherical'
                    OK = OrdinaryKriging(
                        self.extreme.lon,
                        self.extreme.lat,
                        elevations,
                        variogram_model=variogram,
                        verbose=False,
                        enable_plotting=False,
                    )
   
                    elev_network, _ = OK.execute("grid", self.grid_x, self.grid_y)
                    if Inputs_resd.shape[0] ==0:
                        Inputs_resd = elev_network.flatten().reshape(-1, 1)
                    else:
                        Inputs_resd = np.column_stack([Inputs_resd, elev_network.flatten()])

                if len(Inputs_resd.shape) == 1:
                    Inputs_resd = Inputs_resd.reshape(-1, 1)
                Inputs_resd_scaled = scaler_Xi.transform(Inputs_resd)
                Inputs_resd_scaled = torch.Tensor(Inputs_resd_scaled)
                predictions = best_model(Inputs_resd_scaled)
                predictions = predictions.detach().numpy()
                predictions = predictions.reshape([row_size, col_size])
                adjusted_ice_map = self.estimates.reshape([row_size, col_size]) + predictions
        else:
            adjusted_ice_map = self.estimates
        # replacing negative values with zero
        adjusted_ice_map[adjusted_ice_map < 0] = 1e-3
        print(' * ', end ='')
        coordinations_file_name = "ca_processed_coordinations.csv"
        coordinations_dir  = os.path.join(self.data_dir, coordinations_file_name)
        if os.path.exists(coordinations_dir):
            df_original_lonlat = pd.read_csv(coordinations_dir)
        else:
            raise ValueError(f'The file {coordinations_file_name} is not in this dir: {coordinations_dir}')
        if df_original_lonlat.shape[1]>2:
            df_original_lonlat = df_original_lonlat.iloc[:, 1:]

        points = deepcopy(df_original_lonlat.to_numpy())
        # Compute the alpha shape
        alpha_value = 0.2  
        alpha_shape = alphashape.alphashape(points, alpha=alpha_value)
        # Extract the boundary of the alpha shape
        boundary = np.array(alpha_shape.exterior.coords)
        grid_x_mesh, grid_y_mesh = np.meshgrid(self.grid_x, self.grid_y)
        mesh = np.column_stack([grid_x_mesh.flatten(), grid_y_mesh.flatten()])
        # Convert alpha shape boundary to a Shapely Polygon
        polygon = Polygon(np.array(alpha_shape.exterior.coords))
        points_inside = []
        points_outside = []
        for point in mesh:
            shapely_point = Point(point)
            if polygon.contains(shapely_point):
                points_inside.append(point)
            else:
                points_outside.append(point)
        ice_map = adjusted_ice_map.data.flatten()
        stations_coordination = np.column_stack([grid_x_mesh.flatten(), grid_y_mesh.flatten()])
        ice_output = pd.DataFrame(np.column_stack([stations_coordination, ice_map]),
                                                  columns = ['lon', 'lat', 'ice'])
        print(' * ', end='')
        indices = []  # To store indices where points are found
        for lon, lat in points_inside:
            # Check if the point exists in the ice_output dataframe
            result = ice_output[(ice_output['lon'] == lon) & (ice_output['lat'] == lat)]
            if not result.empty:
                indices.extend(result.index.tolist())  # Add the indices to the list
        ice_filtered_data = deepcopy(ice_output)
        # Create a boolean mask to identify rows not in the 'indices' array
        mask = ~np.in1d(np.arange(ice_filtered_data.shape[0]), indices)
        # Set points outside of the border to zero using the mask
        ice_filtered_data.iloc[mask, -1] = 0
        df_adjusted = ice_filtered_data.loc[ice_filtered_data.ice != 0]
        print(f'\n****************************************')
        
        # mask1 = df_adjusted.ice>20
        # mask2 = np.logical_and(df_adjusted.ice>=15, df_adjusted.ice<=20)
        # mask3 = np.logical_and(df_adjusted.ice>=10, df_adjusted.ice<15)
        # mask4 = np.logical_and(df_adjusted.ice>=8, df_adjusted.ice<10)
        # mask5 = np.logical_and(df_adjusted.ice>=6, df_adjusted.ice<8)
        # mask6 = np.logical_and(df_adjusted.ice>=4, df_adjusted.ice<6)
        # mask7 = np.logical_and(df_adjusted.ice>=2, df_adjusted.ice<4)
        # mask8 = np.logical_and(df_adjusted.ice>=0, df_adjusted.ice<2)

        # values = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        # color1 = np.tile(values[7], df_adjusted.shape)
        # color2 = np.tile(values[6], df_adjusted.shape)
        # color3 = np.tile(values[5], df_adjusted.shape)
        # color4 = np.tile(values[4], df_adjusted.shape)
        # color5 = np.tile(values[3], df_adjusted.shape)
        # color6 = np.tile(values[2], df_adjusted.shape)
        # color7 = np.tile(values[1], df_adjusted.shape)
        # color8 = np.tile(values[0], df_adjusted.shape)

        # df_adjusted[mask1] = color1[mask1]
        # df_adjusted[mask2] = color2[mask2]
        # df_adjusted[mask3] = color3[mask3]
        # df_adjusted[mask4] = color4[mask4]
        # df_adjusted[mask5] = color5[mask5]
        # df_adjusted[mask6] = color6[mask6]
        # df_adjusted[mask7] = color7[mask7]
        # df_adjusted[mask8] = color8[mask8]        
        
        # cmap = {
        #     'A': 'dodgerblue',
        #     'B': 'cadetblue',
        #     'C': 'darkkhaki',
        #     'D': 'yellowgreen',
        #     'E': 'peachpuff',
        #     'F': 'orange',
        #     'G': 'orangered',
        #     'H': 'red'}
        # cmap = list(cmap.values())
        # gdf = gpd.GeoDataFrame(
        #     df_adjusted, 
        #     geometry=gpd.points_from_xy(df_adjusted.lon, df_adjusted.lat)
        # )
        cmap = 'hsv'
        
        if self.ice_model == 1:
            ice_model_name = 'CRREL'
        elif self.ice_model == 2:
            ice_model_name = 'Goodwin'
        elif self.ice_model == 3:
            ice_model_name = 'Chaine'
        elif self.ice_model == 4:
            ice_model_name = 'Sanders'
        Title = f'{self.n_year:.0f}-Year Ice Map Based on {ice_model_name}'
        from matplotlib.colors import ListedColormap

        # Original Spectral colormap
        # cmap_spectral = plt.cm.get_cmap('Spectral')
        
        fig = px.scatter_mapbox(df_adjusted, 
                                lat='lat', 
                                lon='lon', 
                                color="ice",
                                color_continuous_scale='rdylbu',
                                mapbox_style="open-street-map",
                                title = Title,
                                zoom=3,
                                opacity=0.1)  
        
        # Update the layout to adjust the map appearance and add a title
  
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            width=800,
            height=400
        )
        fig.update_coloraxes(colorbar=dict(thickness=20, ticklen=1))
        # Show the plot
        fig.show()



        