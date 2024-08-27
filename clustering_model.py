"""
Classes that build the data pipeline for the clustering algorithm and includes the data preprocessing functions as 
class "DataPreprocessing()" and the trained clustering model as class "Clustering()" .
The classes include: 
- Data Preprocessing
    - Encoding strings
    - Handling missing values and outliers
    - Feature engineering
    - Data Normalisation
    - Dimensionality Reduction
- Clustering model
    - KMeans
    - GMM
"""

# Imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib as mpl
from matplotlib.colors import Normalize


class ClusteringAlgorithm:
    """
    Class that includes the child classes "DataPreprocessing()" and "Clustering()".
    """

    # Init common variables to inherit to and share between child classes
    def __init__(self, path):
        self.model = "Not_selected"  # "KMeans", "GMM"
        self.mode = "Initialisation"
        self.processed_data = pd.DataFrame()  # Initiate empty Dataframe
        self.normalised_data = pd.DataFrame()  # Initiate empty Dataframe
        self.pca_data = pd.DataFrame()
        self.raw_data = pd.DataFrame()  # Initiate empty Dataframe
        self.models = {"scaler": [], "pca": [], "clustering": []}
        # Set path variable to load the raw database
        self.path = path

    def __str__(self):
        return print(f"Currently in {self.mode} mode.")

    # Return path given to the class
    def get_path(self):
        return self.path


class DataPreprocessing(ClusteringAlgorithm):
    """
    Class that includes methods to preprocess the full database consisting of raw 
    telemetry data from the combined alerts and trends files, exported from the GENIUS IoT.
    These preprocessing steps are required and include feature engineering steps according to 
    the SotA presented in chapter 2.3.1

    Input:
    - Full database
    Output:
    - Preprocessed database, raw database, scaled database
    """

    # Init common and class specific parameters
    def __init__(self, path, comb):
        # Initialise parameters from "ClusteringAlgorithm" class
        super().__init__(path)
        self.mode = "Preprocessing"
        # Step of the algorithm method
        self.step = 0
        # Path to pump data in json format
        self.pump_data_path = "C:/Users/a00546973/ONEVIRTUALOFFICE/Genius - General/GENIUS Data Lead/Specs/Standardisation/encoding_data.json"
        self.pump_data = {}  # initialise empty Dataframe
        self.comb = comb

    def __str__(self):
        return print(f"Currently in {self.mode} mode.")

    # Step 1: Data Loading
    def data_loading(self):
        # Data loading
        self.step = 1
        # Load the data as dataframe using pandas read_csv method and temporary save it as class variable.
        self.raw_data = pd.read_csv(self.path, low_memory=False)

    # Step 2: Preparing of dataset
    def prepare_data(self):
        """
        The raw database features multiple columns that need to be excluded from because they do not add any information.
        These columns will be dropped. In addition to that, the method converts the "Timestamp" parameter to datetime format, 
        calculates the delta of time "dT" and creates a new "Weekday" parameter.
        Input:
        - self.raw_data: Loaded dataframe 
        Output:
        - self.preprocessed_df: Preprocessed database after step 2
        """

        self.step = 2 # Set step to 2
        try:
            # Load raw data base from class variable
            df = self.raw_data
            unnamed_cols = []
            # Drop unwanted columns
            for col in df.columns:
                # Drop column if the column is an object type since the further progressing requires the data to be float or integer.
                # Drop columns that include no information (if mean == 0 -> no variance of information)
                # If the column is unlabeled --> These result from reindexing during the combination of dataframes
                if (df[col].dtype == object or col == "" or df[col].mean() == 0) and (
                    col != "TimeStamp" and col != "Serial_No" and col != "Pump Type"
                ):
                    df.drop(columns=col, inplace=True)
                else:
                    if col == "Unnamed: 0.1":
                        df.drop(columns=["Unnamed: 0.1"], inplace=True)
                    elif col == "Unnamed: 0":
                        df.drop(columns=["Unnamed: 0"], inplace=True)
                    elif col == "index":
                        df.drop(columns=col, inplace=True)
                    else:
                        continue        
            df = df.drop(unnamed_cols, axis=1)

            # Fill missing values with 0
            df = df.fillna(0)

            # Convert Timestamp to datetime format to calculate time delta
            df["TimeStamp"] = pd.to_datetime(
                df["TimeStamp"], format="%Y-%m-%d %H:%M:%S"
            )

            # Shift time per data slice from pump
            for sn in df["Serial_No"].unique():
                df.loc[df["Serial_No"] == sn, "shifted_time"] = df.loc[
                    df["Serial_No"] == sn, "TimeStamp"
                ].shift(periods=-1)

            # Convert shifted time to datetime format
            df["shifted_time"] = pd.to_datetime(
                df["shifted_time"], format="%Y-%m-%d %H:%M:%S"
            )
            
            # Calculate delta of time between timestamps per pump    
            df["dT"] = df["shifted_time"] - df["TimeStamp"]
            # Fill values to avoid errors
            zero_timedelta = np.timedelta64(0, 's')
            df["dT"].fillna(zero_timedelta, inplace=True)
            # Apply pandas total_seconds() method as lambda function to convert the datetime delta into a int in seconds
            df["dT"] = df["dT"].apply(lambda x: df["dT"].fillna(zero_timedelta, inplace=True) if x == 0 else int(x.total_seconds()))
            
            # Set unrealistic parameters to 0 if negative delta and 1h if too high positive
            df.loc[df["dT"] < 0, "dT"] = df.loc[df["dT"] < 0, "dT"] * -1
            df.loc[df["dT"] > 10000, "dT"] = 3600
           
            # Create a new optional parameter weekday using the datetime module
            df["Weekday"] = df["TimeStamp"].apply(lambda x: x.weekday())
           
            # Save the preprocessed database as class variable
            self.processed_data = df

        except KeyError as err:
            print(f"An KeyError occured in {self.step}\nFrom {err}")

    # Step 3: Encode strings to ordinal values
    def encode_data(self):
        """
        Method encodes object type data that features valuable information into numerical values. This step is required to reuse this
        information in the further data analysis and the unsupervised learning algorithms since the algorithms require numerical 
        parameters as input. The method consists of three sub-steps that update the preprocessed database.
        1. Encode the pump type
        2. Encode the serial number
        3. Encode the alerts
        
        Input:
        - self.preprocessed_df: Preprocessed database from step 2
        - self.pump_data_path: Encoding json that was developed as part of the thesis.
        
        Output:
        - self.preprocessed_df: Updated preprocessed database
        
        """
        
        # Set algorithm step 
        self.step = 3
        # Load json file with references for pump alerts, serial no's and pump models
        with open(self.pump_data_path, "r") as file:
            self.pump_data = json.load(file)

        # Definition of method to encode pump type
        def encode_pumptype(self):
            """
            Method to assign each datapoint the pumptype as encoded parameter.
            The method uses the reference from the "pump_types" in the "encoding_data.json".
            Input:
            - self.preprocessed_df: Preprocessed database from step 2
            - self.pump_data_path: Encoding json that was developed as part of the thesis.
            
            Output:
            - self.preprocessed_df: Updated preprocessed database
            """
            
            self.step = 3.1 # Set step
            try:
                # Extract needed part from json and temporarily save as dataframe
                pump_types_data = self.pump_data["pump_types"]
                pump_types = pd.DataFrame(pump_types_data)
                # Encode pump type by applying pandas merge() function and drop not needed columns
                self.processed_data = pd.merge(
                    self.processed_data,
                    pump_types,
                    left_on="Pump Type",
                    right_on="pump_type",
                    how="left",
                )
                self.processed_data.drop(columns=["pump_type"], inplace=True)

            # Error handling
            except KeyError as err:
                print(f"An KeyError occured in {self.step}\nFrom {err}")

        # Definition of method to encode serial no
        def encode_serial_no(self):
            """
            Method to assign each datapoint the serial number as encoded parameter.
            The method uses the reference from the "pumps" in the "encoding_data.json".
            Input:
            - self.preprocessed_df: Preprocessed database from step 3.1
            - self.pump_data_path: Encoding json that was developed as part of the thesis.
            
            Output:
            - self.preprocessed_df: Updated preprocessed database
            """
            
            self.step = 3.2 # Set step
           # try:
                # Extract needed part from json
            pumps_data = self.pump_data["pumps"][0]
            pumps = pd.DataFrame(pumps_data)

            # Extract serial no and pump type from new data for each unique pump in the database
            for sn in self.processed_data["Serial_No"].unique():
                
                if "Pump Type" in self.processed_data.columns:
                    pump_type = self.processed_data.loc[
                        (self.processed_data["Serial_No"] == sn), "Pump Type"
                    ].values[0]
                else:
                    continue

                # Get highest integer in json
                current_highest_SnCode = max(
                    int(item["SnCode"]) for item in pumps_data
                )
                # Check if the serial no is already existing in the json
                if any(item["Serial_No"] == sn for item in pumps_data):
                    # If existing: Encode serial no with corresponding SnCode from json
                    self.processed_data = pd.merge(
                        self.processed_data,
                        pumps,
                        left_on="Serial_No",
                        right_on="Serial_No",
                        how="left",
                    )

                # If serial no not yet existing: Save serial no with pumptype and new SnCode in json
                else:
                    # Prepare data to append to json
                    data_to_fill = {
                        "SnCode": current_highest_SnCode + 1,
                        "Serial_No": sn,
                        "pump_type": pump_type,
                        "Usage": 0,
                        "VacuumType": 2,
                    }

                    # Append data and save json
                    self.pump_data["pumps"][0].append(data_to_fill)
                    pumps_data = self.pump_data["pumps"][0]
                    pumps = pd.DataFrame(pumps_data)

                    self.processed_data = pd.merge(
                        self.processed_data,
                        pumps,
                        left_on="Serial_No",
                        right_on="Serial_No",
                        how="left",
                    )
                    
                # Delete not needed columns if they are in the columns of the database
                if "Pump Type" in self.processed_data.columns:
                    self.processed_data.drop(columns=["Pump Type"], inplace=True)
                elif "pump_type" in self.processed_data.columns:
                    self.processed_data.drop(columns=["pump_type"], inplace=True)
                else:
                    pass
                
                # Column results of merging of the dataframe with the temporary dataframe and therefore needs to be dropped        
                if "SnCode_y" in self.processed_data.columns:
                    self.processed_data["SnCode"] = self.processed_data["SnCode_y"]
                    self.processed_data.drop(
                        columns=["SnCode_x", "SnCode_y"], inplace=True
                    )
                else:
                    pass
            
            self.processed_data.dropna(subset=["SnCode"], inplace=True)
            
            # Save updated json            
            with open(self.pump_data_path, "w") as file:
                json.dump(self.pump_data, file, indent=4)

            # Drop object type columns of pump type if they are in the columns
            if "Pump Type" in self.processed_data.columns:
                self.processed_data.drop(
                    columns=["Pump Type", "pump_type"], inplace=True
                )
            else:
                pass

            #except KeyError as err:
            #    print(f"An KeyError occured in {self.step}\nFrom {err}")

        # Definition of method to encode alerts
        def encode_alerts(self):
            """
            Method encodes the alerts as parameter based on their severity level that is defined in the json.
            The method uses the reference from the "alerts" in the "encoding_data.json" where each alert is defined 
            to a severity level of 0-4. In addition, the method restructures the columns of the database so that
            the order of the column is always the same after applying this method.
            Here 0 = "No Alert" and 4 = "Critical Alarm"
            Input:
            - self.preprocessed_df: Preprocessed database from step 3.2
            - self.pump_data_path: Encoding json that was developed as part of the thesis.
            
            Output:
            - self.preprocessed_df: Updated preprocessed database
            """
            
            self.step = 3.3 # Set step
            # Set processed dataframe as temporary copy
            temp = self.processed_data.copy()

            try:
                # Extract relevant part of the encoding json
                alerts_data = self.pump_data["alerts"]
                alerts = pd.DataFrame(alerts_data)

                keys = alerts["alert"].values
                values = alerts["code"].values
                
                # Create encoding dictionary    
                alert_dict = dict(zip(keys, values))
                alerts_to_encode = alert_dict.keys()

                # Iterate through the unique alerts
                for alert in alerts_to_encode:
                    # Map severity level to alert
                    if alert in temp.columns:
                        temp[alert] = temp[alert].map({0: 0, 1: alert_dict[alert]})

                    else:
                        continue
                
                # Define the order of the basic columns            
                basic_cols = [
                    "Current",
                    "Power",
                    "MotorSpeed",
                    "P_In",
                    "P_Out",
                    "T_Out",
                    "TotalEnergyConsumption",
                    "TotalCO2Impact",
                    "RunningHours",
                    "SnCode",
                    "PumpCode",
                    "dT",
                    "Usage",
                    "Reopen count",
                    'shifted_time',
                    'Weekday', 
                    'pump_type',
                    'TimeStamp',
                    'Alert condition',
                ]
                alert_cols = [col for col in temp.columns if col not in basic_cols]
        
                # Rearrange the columns order
                temp = temp[basic_cols + alert_cols]

                # Set alert to maximum encoded value (based on json) --> crital alerts
                temp.loc[:, ["Alert"]] = temp[alert_cols].max(axis=1, numeric_only=True)
                
                # Save updated database as class variable processed_data     
                self.processed_data = temp

            except KeyError as err:
                print(f"An KeyError occured in {self.step}\nFrom {err}")

        # Execute child methods
        encode_pumptype(self)
        encode_serial_no(self)
        # Reload data to encode with new SnCode
        with open(self.pump_data_path, "r") as file:
            self.pump_data = json.load(file)
        encode_alerts(self)
        
    # Step 4: Calculate deltas of MS, TotalEnergyConsumption, TotalCO2Impact, and RunningHours
    def calc_delta(self):
        """Calculate delta of TotalCO2Impact and TotalEnergyConsumption and assigns each pump a PressureRange
        based on the mean of the pressure when the pump is running.

        The function iterates through the different SnCodes to calculate the delta for each parameter.
        Parameters can also be multiplied by weights to impact PCA factor combinations
        Input:
        - self.processed_data: Processed database from step 3
            
        Output:
        - self.processed_data: Updated processed dataframe
        """
        self.step = 4.1

        temp_df = self.processed_data
        sn_keys = pd.unique(temp_df["SnCode"])

        for code in sn_keys:
            print(code)
            # Calculate "TargetPressure", delta "dE", delta "dCO2", and "dP"
            temp_df.loc[temp_df["SnCode"] == code, "TargetPressure"] = (
                temp_df.loc[
                    ((temp_df["P_In"] < 600) & (temp_df["MotorSpeed"]>600) & (temp_df["SnCode"] == code)), "P_In"
                ]
                .mean()
                .round(2)
            )
            
            # Calculate dP as distance to Targetpressure
            temp_df.loc[temp_df["SnCode"] == code, "dP"] = (
                abs(temp_df["TargetPressure"] - temp_df["P_In"])
            )
            
            # Apply pandas diff method to calculate delta of parameters between timestamps
            temp_df.loc[temp_df["SnCode"] == code, "dE"] = temp_df.loc[
                temp_df["SnCode"] == code, "TotalEnergyConsumption"
            ].diff(
                periods=-1
            )  # periods=-1 or 1, depends on if the dataset is in ascending or descending order
            temp_df.loc[temp_df["SnCode"] == code, "dCO2"] = temp_df.loc[
                temp_df["SnCode"] == code, "TotalCO2Impact"
            ].diff(periods=-1)

        self.step = 4.2 # Set step
        # Assign pressure range according to calculated mean using apply method
        temp_df["PressureRange"] = temp_df["TargetPressure"].apply(
            self.assign_Pressure_Range
        )

        self.step = 4.3 # Set step

        # TotalEnergyConsumption, TotalCO2Impact, and RunningHours are calculated parameters that are the sums of
        # the respective parameter and can therefore only be positive values. Set values to 0 if they are negative
        # Fill missing values
        temp_df.loc[temp_df["dE"] < 0, "dE"] = 0
        temp_df.drop(index=temp_df[temp_df["dE"] > 100].index, inplace=True)
        temp_df.loc[temp_df["dCO2"] < 0, "dCO2"] = 0
        temp_df = temp_df.fillna(0)

        # Handle outliers before scaling of data to avoid bias. 
        # Limits were selected based on the amount of occurences with greater values than the 
        # respective threshhold.
        temp_df.loc[temp_df["dT"] > 2000, "dT"] = temp_df.loc[
                temp_df["dT"] < 2000, "dT"
            ].max()
        temp_df.loc[temp_df["dE"] > 5, "dE"] = 5
        temp_df.loc[temp_df["dCO2"] > 10, "dCO2"] = 10
        
        # Save temp_df as processed data in the object
        self.processed_data = temp_df

    # assign pressure range according to pre defined buckets
    def assign_Pressure_Range(self, p_t):
        """
        Assign pressure range using the calculated mean of the "TargetPressure" parameter. This function is used in an apply function
        where it passes the parameter as p_t to this function. Here, p_t gets compared to the different pressure ranges. The corresponding
        Pressure range is assigned if the condition is fulfilled.
        The Pressure ranges were defined with Adria Sala Romera, a Service Development Engineer Product Support.
        Input:
        - p_t: Parameter value of "TargetPressure" computed in step 4.1

        Returns:
        - int: Pressure range code according to the predefined pressure ranges 
        """

        # Pressure range 1
        if p_t < 50:
            return 1

        # Pressure range 2
        elif 50 <= p_t < 75:
            return 2

        # Pressure range 3
        elif 75 <= p_t < 125:
            return 3

        # Pressure range 4
        elif 125 <= p_t < 175:
            return 4

        # Pressure range 5
        elif 175 <= p_t < 225:
            return 5

        # Pressure range 6
        elif 225 <= p_t < 275:
            return 6

        # Pressure range 7
        elif 275 <= p_t < 325:
            return 7

        # Pressure range 8
        elif 325 <= p_t <= 400:
            return 8

        # Pressure range 9
        elif p_t > 400:
            return 9
   
    # Step 5: Handle null values that aren't reasonable (P_In, P_Out, T_Out)
    def handling_nulls(self):
        """
        This method handles null values for pressure and temperature that are not reasonable and 
        standardises the order of columns in the dataset. 
        The pressure can not physically reach 0. If such values occurs it is most likely due to a 
        sensor error or due to a accuracy fault. In case of the temperature parameter, zeros are only
        appearing after the creation of database and therefore artificial outliers. 
        Input:
        - self.processed_data: Prepared database from step 4
        
        Return:
        - self.processed_data: Processed database that only includes the parameters of interest.
        """
        # Handling nulls in the dataset
        self.step = 5
        df = self.processed_data
        try:
            # Iterate through the parameters of each pump
            for num in pd.unique(df["SnCode"]):
                print(num)
                # If outlet pressure == 0 -> set pressure to mean of remaining parameter values
                df.loc[(df["SnCode"] == num) & (df["P_Out"] == 0), "P_Out"] = df.loc[
                    ((df["SnCode"] == num) & (df["P_Out"] > 0)), "P_Out"
                ].mean()

                # If inlet pressure == 0 -> set pressure to TargetPressure 
                df.loc[(df["SnCode"] == num) & (df["P_In"] == 0), "P_In"] = df.loc[
                    ((df["SnCode"] == num) & (df["P_In"] > 0)), "TargetPressure"
                ].mean()
                print(df.loc[
                    ((df["SnCode"] == num) & (df["P_In"] > 0)), "TargetPressure"
                ].mean())
                # If outlet temperature == 0 --> set temperature to mean of remaining parameter values
                df["T_Out"].fillna(0, inplace=True)
                df.loc[(df["SnCode"] == num) & (df["T_Out"] == 0), "T_Out"] = df.loc[
                    ((df["SnCode"] == num) & (df["T_Out"] > 0)), "T_Out"
                ].mean()

            # Rearrange column order of dataset and extract only the needed parameters.
            # This standardisation improves the quality and robustness of the dataset and the algorithms,
            # since the developed scripts do not need to be changed if these data preprocessing steps
            # are used.
            names = [
                "TimeStamp",
                "SnCode",
                "Current",
                "Power",
                "MotorSpeed",
                "P_In",
                "P_Out",
                "T_Out",
                "dP",
                "dE",
                "dT",
                "dCO2",
                "PumpCode",
                "Alert",
                "Reopen count",
                "PressureRange",
                "RunningHours",
                "TotalEnergyConsumption",
                "TotalCO2Impact"
            ] 
            # Create cut database that only includes the parameters of interest.
            df_cut = df[names]
            
            # Add SnCode to raw database    
            self.raw_data["SnCode"] = df["SnCode"]
            # Save preprocessed data
            self.processed_data = df_cut

        except KeyError as err:
            print(f"An KeyError occured in {self.step}\nFrom {err}")

    # Step 6: Normalise the data
    def data_normalisation(self, path1, path2):
        """
        Method normalises the processed database to prepare the data for dimensionality reduction methods
        and for the training / testing of unsupervised learning algorithms which require normalised data with values
        between 0 and 1. Detailed reasons for scaling are explained in chapter 2.3.1.3.
        Input:
        - self.processed_data: Processed database from step 5
        
        Return: 
        - self.scaled_data: Scaled and normalised dataframe
        """
        
        self.step = 6 # Set step
        # Data normalisation process
        temp_data = self.processed_data.copy()
        temp_data.drop(columns=["TimeStamp","PumpCode","SnCode"], inplace=True)  # drop timestamp

        # Select scaler for dataset if only one pumptype is selected
        # Further condition can be added if scalers for other pump types are trained
        
        # Load Robust- and MinMax-Scaler file that are trained for 1 pump code (GHS2002)
    
        scaler_path1 = path1
        robust_scaler = joblib.load(scaler_path1)
        scaler_path2 = path2
        minmax_scaler = joblib.load(scaler_path2)

        self.models["scaler"] = minmax_scaler

        try:
            self.step = 6.1 # Set step
                   
            # Delete Serial_No and PumpCode column if in dataset
            if "Serial_No" in temp_data.columns:
                temp_data.drop(columns="Serial_No", inplace=True)
            if "PumpCode" in temp_data.columns:
                temp_data.drop(columns="PumpCode", inplace=True)
            columns = temp_data.columns
           
            # Apply MinMaxScaler to normalised data
            robust_data = robust_scaler.transform(temp_data)
            scaled_data = minmax_scaler.transform(robust_data)
            self.normalised_data = pd.DataFrame(scaled_data, columns=columns)
            
            # Save scaled data without labels as class variable
            self.scaled_data = scaled_data

        except KeyError as err:
            print(f"An KeyError occured in {self.step}\nFrom {err}")

    # Step 7: Apply dimensionality reduction with pretrained PCA
    def dimensionality_reduction(self, pca_path):
        """
        Apply dimensionality reduction to the preprocessed and scaled database as last preprocessing step. 
        This step is required to avoid the curse of dimensionality during the data analysis, to enable the visualisation
        of the data, and to reduce the time for training and testing the unsupervised learning algorithms. 
        Moreover, dimensionality reduction methods are part of unsupervised learning methods and might result in
        intersting insights. The SotA on dimensionality reduction methods is in detail explained in chapter 2.3.1.4
        """
        
        self.step = 7 # Set step
        # Load trained pca model with joblib
        pca_model = joblib.load(pca_path)
        self.models["pca"] = pca_model
        # Load normalised instead of scaled data, since the PCA was trained with labels. 
        temp_data = self.normalised_data
        parameter_comb = {
        "0":["P_In","P_Out","T_Out","Current","MotorSpeed","Power"],
        "1":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","RunningHours","TotalCO2Impact","TotalEnergyConsumption"],
        "2":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dT","dE","dCO2"],
        "3":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dT","dE","dCO2","Alert","Reopen count"],
        "4":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dT","dE","dCO2","Alert","Reopen count","PressureRange"],
        "5":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dT","dE","dCO2","Alert","Reopen count","dP"],
        "6":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dT","dE","Alert","Reopen count"],
        "7":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dT","dE","Alert","Reopen count","PressureRange"],
        "8":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dT","dE","Alert","Reopen count","dP"],
        "9":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dE","Alert","Reopen count","dP"],
        }
        
        columns = parameter_comb[str(self.comb)]
        temp_data = temp_data[columns]
        
        try:
            # Apply PCA to normalised data and create the dimension reduced dataframe.
            temp_data = pca_model.transform(temp_data)
            pcs_df = pd.DataFrame(
                temp_data, columns=["pca0", "pca1", "pca2", "pca3"]
            )
            # Save PCA reduced dataframe as class variable
            self.pca_data = pcs_df

        except KeyError as err:
            print(f"Keyerror occured in {err}")

    # Execute steps 1-7
    def execute_preprocessing(self, path1, path2, pca_path):
        # Calling all preprocessing methods. This method can be implemented in a automated data preparation flow. 
        # Step 1
        DataPreprocessing.data_loading(self)

        # Step 2
        DataPreprocessing.prepare_data(self)

        # Step 3
        DataPreprocessing.encode_data(self)

        # Step 4
        DataPreprocessing.calc_delta(self)

        # Step 5
        DataPreprocessing.handling_nulls(self)

        # Step 6
        DataPreprocessing.data_normalisation(self, path1, path2)

        # Step 7
        DataPreprocessing.dimensionality_reduction(self, pca_path)

        print(f"{self.step} steps executed.")
        
    
class Clustering(ClusteringAlgorithm):
    
    def __init__(self, path, run, pca_data, processed_data):
        super().__init__(path)
        self.clustering_data = pd.DataFrame()
        self.run = run
        self.path = path
        self.processed_data = processed_data
        self.pca_data = pca_data
        
    def load_model(self):
        # Load pretrained clustering model
        self.models["clustering"] = joblib.load(self.path)
        return print("Model loaded...")        
    
    def get_score(self):
        test_score = self.models["clustering"].score_samples(self.pca_data.values)
        self.clustering_data["score"] = test_score

        for k in self.clustering_data["Cluster"].unique():
            #for alert in df["Alert"].unique():
            self.clustering_data.loc[( self.clustering_data["Cluster"] == k), "score_mean"] = self.clustering_data.loc[
                ( self.clustering_data["Cluster"] == k),"score"].mean() #& (df["Alert"] == alert)
        metric = self.clustering_data[["Cluster","score_mean"]].value_counts() 
        
        return  self.clustering_data, metric
    
    def apply_clustering(self):
        # Apply clustering to validation dataset
        self.clustering_data = self.processed_data
        self.clustering_data[self.pca_data.columns] = self.pca_data
        data = self.pca_data.values
        
        self.clustering_data["Cluster"] = self.models["clustering"].predict(data)
      
        self.clustering_data, metric = self.get_score()
        print(metric)
        # Plot validation dataset
        clusters = 7
        cmap = plt.cm.get_cmap("turbo", clusters)
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(projection="3d")

        x = self.pca_data.iloc[:,0]
        y = self.pca_data.iloc[:,1]
        z = self.pca_data.iloc[:,2]
        ax.set_xlabel("PC_0")
        ax.set_ylabel("PC_1")
        ax.set_zlabel("PC_2")
        
        # Plot the data points with colors representing the clusters
        p = ax.scatter(x,y,z, c=self.clustering_data["Cluster"] ,s=1, alpha=0.2 ,cmap=cmap)
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        plt.title("GMM with PCA reduced data")
        ax.set_position([left, bottom, width, height])
        plt.colorbar(p, ticks=range(clusters))
        plt.show()
        return print("Clustering applied...")
        
    def return_results(self):
        # Evaluate results
        dist = pd.DataFrame(columns=["Validate"])
        dist["Validate"] = self.clustering_data["Cluster"].value_counts() / len(self.clustering_data)
        dist = (dist*100).round(2)
        
        parameter_comb = {
        "0":["P_In","P_Out","T_Out","Current","MotorSpeed","Power"],
        "1":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","RunningHours","TotalCO2Impact","TotalEnergyConsumption"],
        "2":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dT","dE","dCO2"],
        "3":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dT","dE","dCO2","Alert","Reopen count"],
        "4":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dT","dE","dCO2","Alert","Reopen count","PressureRange"],
        "5":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dT","dE","dCO2","Alert","Reopen count","dP"],
        "6":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dT","dE","Alert","Reopen count"],
        "7":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dT","dE","Alert","Reopen count","PressureRange"],
        "8":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dT","dE","Alert","Reopen count","dP"],
        "9":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","dE","Alert","Reopen count","dP"],
        }
        
        data = self.processed_data[parameter_comb["9"]] 
        data.loc[:,"SnCode"] = self.processed_data["SnCode"]
        data.loc[:,["pca0","pca1","pca2","pca3"]] = self.pca_data
        data.loc[:,"Cluster"] = self.clustering_data["Cluster"]
        
        def get_definition(data, name):
            # This code is used to iterate through the clusters of the dataset and get their statistics
            df_list = []
            for cluster in pd.unique(data[name]):
                metric = data.loc[data[name]==cluster].describe().round(2)
                metric.index = pd.MultiIndex.from_product([[cluster], metric.index])
                df_list.append(metric)
                
                result_df = pd.concat(df_list, axis=0)
                result_df = result_df.sort_index(axis=0)
            
            return result_df    

        # Calculate cluster metrics
        m_df = get_definition(data, "Cluster")
        path = "cluster_description_validation_"+str(self.run)+".xlsx"
        m_df.to_excel(path)
        return print(dist.head(20))
    
    def plot_cluster(self, pump, cluster, r1, r2):
        # Plot cluster distribution across process
        try:
            fig, axs = plt.subplots(9,1, figsize=(20,40))
            start_time = r1
            end_time = r2 
            data = self.clustering_data
            
            if start_time != "":
                filtered_df = data[(data['TimeStamp'] >= start_time) & (data['TimeStamp'] <= end_time) & (data['SnCode'] == pump)]
            else:
                filtered_df = data[data["SnCode"]==pump]     

            k = data[cluster].max()+1    
            cmap = mpl.cm.get_cmap("turbo", k)
        
            norm = Normalize(vmin=0, vmax=data[cluster].unique().max())
            
            sc1 = axs[0].scatter(filtered_df["TimeStamp"],filtered_df["Power"], c=filtered_df[cluster], cmap=cmap, s=3, norm=norm)
            axs[0].set_ylabel("Power [kW]")
            axs[0].set_title(f"Cluster Distribution of Parameters\nPower of pump {pump}")
            cbar1 = fig.colorbar(sc1, ax=axs[0])
            cbar1.set_label('Clusters')
        
            sc2 = axs[1].scatter(filtered_df["TimeStamp"],filtered_df["P_In"], c=filtered_df[cluster], cmap=cmap, s=3, norm=norm)
            axs[1].set_ylabel("P_In [mbar]")
            axs[1].set_title(f"P_In of pump {pump}")
            cbar2 = fig.colorbar(sc2, ax=axs[1])
            cbar2.set_label('Clusters')
            
            sc3 = axs[2].scatter(filtered_df["TimeStamp"],filtered_df["T_Out"], c=filtered_df[cluster], cmap=cmap, s=3, norm=norm)
            axs[2].set_ylabel("T_Out [°C]")
            axs[2].set_title(f"T_Out of pump {pump}")
            cbar3 = fig.colorbar(sc3, ax=axs[2])
            cbar3.set_label('Clusters')
            
            sc4 = axs[3].scatter(filtered_df["TimeStamp"],filtered_df["MotorSpeed"], c=filtered_df[cluster], cmap=cmap, s=3, norm=norm)
            axs[3].set_ylabel("MotorSpeed [rpm]")
            axs[3].set_title(f"MotorSpeed of pump {pump}")
            cbar4 = fig.colorbar(sc4, ax=axs[3])
            cbar4.set_label('Clusters')
            
            sc5 = axs[4].scatter(filtered_df["TimeStamp"],filtered_df["pca0"], c=filtered_df[cluster], cmap=cmap, s=3, norm=norm)
            #axs[4].plot(filtered_df["TimeStamp"],filtered_df["pca0_mean"], color="darkred",linestyle="-", alpha=0.5)
            axs[4].set_ylabel("Principle Component 0")
            axs[4].set_title(f"Principle Component 0 of pump {pump}")
            cbar5 = fig.colorbar(sc5, ax=axs[4])
            cbar5.set_label('Clusters')
            
            sc6 = axs[5].scatter(filtered_df["TimeStamp"],filtered_df["pca1"], c=filtered_df[cluster], cmap=cmap, s=3, norm=norm)
            #axs[5].plot(filtered_df["TimeStamp"],filtered_df["pca1_mean"], color="darkred",linestyle="-", alpha=0.5)
            axs[5].set_ylabel("Principle Component 1")
            axs[5].set_title(f"Principle Component 1 of pump {pump}")
            cbar6 = fig.colorbar(sc6, ax=axs[5])
            cbar6.set_label('Clusters')
            
            sc7 = axs[6].scatter(filtered_df["TimeStamp"],filtered_df["pca2"], c=filtered_df[cluster], cmap=cmap, s=3, norm=norm)
            #axs[6].plot(filtered_df["TimeStamp"],filtered_df["pca2_mean"], color="darkred",linestyle="-", alpha=0.5)
            axs[6].set_ylabel("Principle Component 2")
            axs[6].set_title(f"Principle Component 2 of pump {pump}")
            cbar7 = fig.colorbar(sc7, ax=axs[6])
            cbar7.set_label('Clusters')
            
            sc8 = axs[7].scatter(filtered_df["TimeStamp"],filtered_df["pca3"], c=filtered_df[cluster], cmap=cmap, s=3,norm=norm)
            #axs[7].plot(filtered_df["TimeStamp"],filtered_df["pca3_mean"], color="darkred",linestyle="-", alpha=0.5)
            axs[7].set_ylabel("Principle Component 3")
            axs[7].set_title(f"Principle Component 3 of pump {pump}")
            cbar8 = fig.colorbar(sc8, ax=axs[7])
            cbar8.set_label('Clusters')
            
            sc9 = axs[8].scatter(filtered_df["TimeStamp"],filtered_df["score"], c=filtered_df[cluster], cmap=cmap, s=3, norm=norm)
            #axs[7].plot(filtered_df["TimeStamp"],filtered_df["pca3_mean"], color="darkred",linestyle="-", alpha=0.5)
            axs[8].set_ylabel("GMM Score")
            axs[8].set_title(f"GMM Score of pump {pump} from {start_time} to {end_time}")
            cbar8 = fig.colorbar(sc9, ax=axs[8])
            cbar8.set_label('Clusters')
        
            
            path = "validation_plot_" + str(pump)
            plt.savefig(path)
            plt.show()    
            return print("Graphs plotted.")
                
        except KeyError as err:
            print(f"A KeyError occured: {err}")
        
    def execute_clustering(self):
        # Execute clustering
        self.load_model()
        
        self.apply_clustering()
        
        self.return_results()
        for pump in self.processed_data["SnCode"].unique():
            self.plot_cluster(pump, "Cluster","","")
        