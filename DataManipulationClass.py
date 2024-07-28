"""
This file includes the methods to generate datasets by combining the "Trends" and "Alerts" exports of 
pumps from the genius cloud in the Database class. The methods are implemented in the "simpleGUI.py" 
and can there be accessed by the user via a GUI. 
The methods require the alerts and trends export need to be saved like in these example paths:

Alerts file path: add_directory_location/GHS2002_API695269_Alerts.xlsx
Trends file path: add_directory_location/GHS2002VSD+_(API695269)-20230724-20230827.csv

Steps to generate databases from alerts and trends files:
1. select_files
    - opens dialogue to select Trends-File and Alerts file
2. load_data 
    1. calls prepTrendsData
        -> prepares Trends data
    2. calls prepAlertData
        -> prepares Alert data
    3. calls combineData
        -> combines prepared Trends and Alerts file to one database for the selected pump
        -> applies basic preprocessing: handling of missing values, interpolation of missing values, 
            dropping of columns that are not needed
        -> return as csv file

Third step is the combination of multiple prepared databases that are located in one directory        
3. combDatabases
    1. combine prepared databases to one FullDatabase
    2. save FullDatabase as csv

"""

# Imports
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import datetime
from datetime import datetime
import dateutil.parser
import os.path
import re
from scipy.interpolate import interp1d


# Definition of Database class
class Database:
    # Initialisation of parameters that are shared beetween methods of the database class
    def __init__(self):
        self.key = 0
        self.db_created = False
        self.files_selected = False
        self.file1 = ""
        self.file2 = ""
        self.file_list = []
        self.file_type = 1  # 1 = csv, 0 = excel
        self.database = pd.DataFrame()
        self.joined_DB = pd.DataFrame()

    # Method to select the Trends and Alert files
    def select_files(self, start):
        """
        Method to select the "Trends" and "Alerts" files.
        After Selection the files are saved as file1 = Trends and file 2 = Alerts.
        Input:
        - User input -> manual selection of alert and trend file
        Return:
        - Save file paths as self.file1, self.file2
        """

        # Tkinter GUI to take user input from filedialog
        if start == 1:
            root = tk.Tk()
            root.withdraw()  # Hide the root window
        else:
            print("Call Database.select_files(1) to start the program")

        # Filedialog to get trends file path
        file1 = filedialog.askopenfilename(title="Select 'Trends' file")
        if file1:
            # Filedialog to get alerts file path
            file2 = filedialog.askopenfilename(title="Select 'Alert' file")
            if file2:
                # Confirmation request if the correct files were selected
                confirm = tk.messagebox.askyesno(
                    title="Confirm Selections",
                    message=f"You have selected:\n\n{file1}\n{file2}\n\nProceed?",
                )

                # If confirmation was yes -> save files
                if confirm:
                    self.file1 = file1
                    self.file2 = file2
                    self.files_selected = True
                    textvar = f"Selected files:\n {file1}\n{file2}"
                    return textvar
                # If any filedialog was closed or deselected -> Quit current selection
                else:
                    textvar = "Quit..."
                    return textvar
            else:
                textvar = "Quit..."
                return textvar
        else:
            textvar = "Quit..."
            return textvar

    # Method that calls the preparation methods of the Trends and Alerts files a the combine method for both
    def load_data(self):
        """
        Method to call the methods to prepare the data and combine them.
        Input:
        - self.file1: Path of trends file
        - self.file2: Path of alerts file
        Return:
        - db: Combined data base that is saved as self.database and as csv
        """

        # Check if the required files were selected
        if self.files_selected == True:
            # Preparation of "Trends" data
            file1, serial_no, pump_type = Database.prepTrendsData(self, self.file1)

            # Preparation of "Alert" data
            file2 = Database.prepAlertData(self, self.file2)

            # Combine data
            db = Database.combineData(self, file1, file2, serial_no, pump_type)
            self.database = db

        # If files were not selecte call select files method again
        else:
            print("No files selected.\nPlease select files to load data.")
            Database.select_files(self, 1)

        return db  # ignore: type

    # Method to prepare the alerts data export for combination
    def prepAlertData(self, file):
        """
        Method to prepare the raw alert data. Prepared dataframe will be returned.
        Input:
        - file: Alert file path
        Returns:
        - temp_df: prepared alert data as pd.Dataframe
        """

        print("Preparing alerts file...")
        # Load exported alert excel from file path
        temp_df = pd.read_excel(file)

        # Define unnecessary columns for deletion
        if " " in temp_df.columns:
            drop_list_alerts = [
                " ",
                "Status",
                "Subsystem",
                "Total duration",
                "Closed at",
            ]
        else:
            drop_list_alerts = ["Status", "Subsystem", "Total duration", "Closed at"]

        # Select alerts from pump only since alerts from "Subsystem"=="Gateway"
        # consist only of connectivity issues and are not of interest for the analysis.
        # Decision to exclude the subsystem from Yun Shi (Data Intelligence )
        temp_df = temp_df[temp_df["Subsystem"] == "Pump"]

        # Apply pandas get_dummies method to one-hot encode all unique alerts that are in the alert list
        # Save the dummies as dataframe and concat to loaded data
        dummies = pd.get_dummies(temp_df["Alert condition"])
        temp_df = pd.concat([temp_df, dummies], axis=1)

        # Drop defined unnecessary columns
        temp_df = temp_df.drop(columns=drop_list_alerts)

        # Set timestamp columns to datetime format to ensure the format and to use them in the combination
        temp_df["Created at"] = pd.to_datetime(
            temp_df["Created at"], format="%d-%m-%Y %H:%M:%S"
        )
        temp_df["Resolved at"] = pd.to_datetime(
            temp_df["Resolved at"], format="%d-%m-%Y %H:%M:%S"
        )
        print("Alert file prepared and loaded.")
        return temp_df

    # Method to prepare the trends data export for combination
    def prepTrendsData(self, file):
        """
        Method to prepare the raw trends data. Prepared dataframe will be returned.
        Input:
        - file: Trends file path
        Return:
        - prepared trends data as pd.Dataframe
        """

        print("Preparing trends file...")
        # Load Trends data as dataframe from file path
        temp_df = pd.read_csv(file)

        # Extract the serial no and the pump type from the file path using the extract_SN 
        # method to add them as parameters
        serial_no, pump_type = self.extract_SN(file)

        # Extract column labels of trends parameters
        db_cols = temp_df.columns
        drop_list_MK5 = [
            "ElementInletPressure",
            "InletTemperature",
            "InletValve",
            "ActualFlow",
            "MixingValve0100",
        ]
        drop_list_GHS900 = ["ActualFlow"]

        # Renaming of GHS Controllor specific column names to ensure a standardised naming convention for data analysis
        if all(col in db_cols for col in ["ElementOutletPressure", "DevicePower"]):
            if "ElementInletPressure" in db_cols:
                temp_df = temp_df.drop(columns=drop_list_MK5)
            else:
                pass
            temp_df.rename(
                columns={
                    "InletPressure": "P_In",
                    "ElementOutletPressure": "P_Out",
                    "OutletTemperature": "T_Out",
                    "CompressorMotorSpeed": "MotorSpeed",
                    "DevicePower": "Power",
                },
                inplace=True,
            )
        # Renaming of GHS4600 specific columns
        elif all(col in db_cols for col in ["ElementOutletPressure", "OutputPower"]):
            temp_df.rename(
                columns={
                    "InletPressure": "P_In",
                    "ElementOutletPressure": "P_Out",
                    "OutletTemperature": "T_Out",
                    "CompressorMotorSpeed": "MotorSpeed",
                    "OutputPower": "Power",
                },
                inplace=True,
            )

        # Renaming of GHS1300 specific column names
        elif all(col in db_cols for col in ["PumpTemperature", "Drive1Motor"]):
            temp_df.rename(
                columns={
                    "Drive1Motor": "Current",
                    "InletPressure": "P_In",
                    "ExhaustPressure": "P_Out",
                    "PumpTemperature": "T_Out",
                    "CompressorMotorSpeed": "MotorSpeed",
                    "OutputPower": "Power",
                },
                inplace=True,
            )
        # Renaming of GHS900 specific column names
        elif "PumpTemperature" in db_cols:
            if "ActualFlow" in db_cols:
                temp_df = temp_df.drop(columns=drop_list_GHS900)
            else:
                pass
            temp_df.rename(
                columns={
                    "InletPressure": "P_In",
                    "ExhaustPressure": "P_Out",
                    "PumpTemperature": "T_Out",
                    "CompressorMotorSpeed": "MotorSpeed",
                    "OutputPower": "Power",
                },
                inplace=True,
            )

        # Renaming of GHS2002 specific column names
        else:
            temp_df.rename(
                columns={
                    "inletPressurePressure": "P_In",
                    "outletPressurePressure": "P_Out",
                    "outletTemperatureTemperature": "T_Out",
                    "mainInverterMotorCurrent": "Current",
                    "mainInverterActualSpeed": "MotorSpeed",
                    "powerIntegratorPower": "Power",
                },
                inplace=True,
            )

        # Add pump type and serial no as new columns to the dataframe
        temp_df["Pump Type"] = pump_type
        temp_df["Serial_No"] = serial_no

        # Convert MessageTimestamp to datetime format to be able to match them to the timerange of alerts
        temp_df["NewTimestamp"] = temp_df["MessageTimestamp"].apply(
            self.convert_datetime
        )
        temp_df["NewTimestamp"] = pd.to_datetime(temp_df["NewTimestamp"])

        # Drop old timestamp column, reset the index and sort the data by descending time order
        temp_df = temp_df.drop(columns=["MessageTimestamp"])
        temp_df.reset_index(inplace=True, drop=True)
        temp_df = temp_df.sort_values(by="NewTimestamp", ascending=False)
        temp_df = temp_df.reset_index(drop=True)

        print("Trends file prepared and loaded.")
        return temp_df, serial_no, pump_type

    # Method to extract the serial no and pump type from trends file path
    def extract_SN(self, path):
        """
        Method extracts serial no and pump type of the pump by slicing the path.
        Both variables will be used as new parameters.
        Input:
        - path: Path of the trends file
        Return:
        - SN_string: Serial no as str
        - pump_type: Pump type as str
        """

        print(f"Filename: {path}")
        filename = path
        match = re.search(r"\((.*?)\)", filename)
        if match:
            SN_string = match.group(1)
            print(SN_string)  # Output: API854014
        else:
            SN_string = ""

        # Extract pump type string based by searching the matching part of the path
        match = re.search(r"/([^+/]+)\+", filename)
        if match:
            pump_type = match.group(1)
            print(pump_type)  # Output: GHS...
        else:
            pump_type = ""

        return SN_string, pump_type

    # Method to combine the prepared dataframes from prepTrends- and prepAlertsData methods
    def combineData(self, df1, df2, serial_no, pump_type):
        """
        Method to combine specific columns from the prepared alerts and trends dataframes with regard to the timedate of the rows.
        Input:
        - df1: Prepared trends dataframe
        - df2: Prepared alerts dataframe
        - serial_no: Serial number of the device
        - pump_type: Type of pum

        Return:
        - database: Combined dataframe consisting of alerts and trends data
        """

        # Create combined dataframe with structure given by trends dataframe and alerts columns
        combined_df = df1.reindex(
            columns=df1.columns.union(df2.columns), fill_value=None
        )

        # Iterate over the alerts in the alerts dataframe to map the alerts to the time range where the alert occurs
        for index, row in df2.iterrows():
            # Defining start time, end time and alert condition from Alerts file
            start_date = row["Created at"]
            try:
                # Find the timestamp that is the closest to the start_date and set it as start time 
                # since the timestamps from alerts and trends do not necessarily match
                start_time = self.find_closest_time(combined_df, start_date)
            except ValueError:
                print("Skipped alert with open end.")
                continue

            end_date = row["Resolved at"]
            try:
                # Find the timestamp that is the closest to the end_date and set it as start time
                # since the timestamps from alerts and trends do not necessarily match
                end_time = self.find_closest_time(combined_df, end_date)
            except ValueError:
                print("Skipped alert with open end.")
                continue

            condition = row["Alert condition"]
            # If end time is not given or the problem wasn't solved yet, skip the row
            if pd.isnull(end_time):
                print("Skipped alert with open end")
                continue
            # Else check if the condition is given
            elif condition != (None or ""):
                # First fill condition of alert within specified time range with bool = 1
                combined_df.loc[
                    (combined_df["NewTimestamp"] >= start_time)
                    & (combined_df["NewTimestamp"] <= end_time),
                    condition,
                ] = 1
                # Then also set the row for Alert condition bool = 1
                combined_df.loc[
                    (combined_df["NewTimestamp"] >= start_time)
                    & (combined_df["NewTimestamp"] <= end_time),
                    "Alert condition",
                ] = 1
                
                # Add reopen count to dataframe
                combined_df.loc[
                    (combined_df["NewTimestamp"] >= start_time)
                    & (combined_df["NewTimestamp"] <= end_time),
                    "Reopen count",
                ] = row["Reopen count"]
                
                # Lastly copy Created at in specified time range
                combined_df.loc[
                    (combined_df["NewTimestamp"] >= start_time)
                    & (combined_df["NewTimestamp"] <= end_time),
                    "Created at",
                ] = row["Created at"]
            else:
                # Print info, if none of these conditions were met
                print("No condition met")
                breakpoint

        print("Status before trimDB")
        # Call trimDB method to preprocess combined dataframe
        database = self.trimDB(combined_df)

        # Saving created Database as csv file
        currentDateTime = datetime.now()  # type: ignore
        timestring = currentDateTime.strftime("%H%M%S")
        datestring = str(currentDateTime.date())

        # Save combined database with the specified path that includes the pump type, serial no,
        # and datetime
        path = os.path.join(
            "C:/Users/a00546973/Desktop/Database/"
            + pump_type
            + "_"
            + serial_no
            + "_DB_"
            + datestring
            + "_"
            + timestring
            + ".csv"
        )
        database.to_csv(path, sep=",", index_label="TimeStamp")
        self.db_created = True
        print("Database has been created as: " + path)

        return self.database

    # Method to find the closest timestamp to the "created_at" from the alerts file
    def find_closest_time(self, combined_df, date):
        """
        Method finds the timestamp in that is the closest to the "created_at" time from the specific alert.
        Here, the minimum absolute time difference is calculated to find the closest time stamp.
        Input:
        - combined_df: Combined dataframe
        - date: Date of interest for which the closes timestamp needs to be found

        Return:
        - closest_date: Closest date to the "Created_at" from the alert
        """
        # Calculate abs time difference for each timestamp to the date
        temp_df = combined_df
        temp_df["time_difference"] = (temp_df["NewTimestamp"] - date).abs()
        # Find minimum difference and set this as closest date
        closest_date = temp_df.loc[temp_df["time_difference"].idxmin(), "NewTimestamp"]

        return closest_date

    # Optional Method to resample the interval of the parameters in a dataset to a specified frequency
    def resampleDataset(self, db, serial_no):
        # Resample dataset to interval of 1 min
        interval = "60s"
        db = db
        list1 = ["NewTimeStamp", "Created at", "Closed at", "Serial_No"]

        # Set target datetime range as grid
        target_timestamps = pd.date_range(
            start=db.index.min(), end=db.index.max(), freq=interval
        )

        # Create new dictionary with target frequency of timestamps
        resampled_db = {}
        resampled_db["TimeStamp"] = target_timestamps

        # Append the data for each parameter with the new
        for col in db.columns:
            # Only append data for the parameters that are not in list1
            if col not in list1:
                # Interpolate parameter values
                interpolator = interp1d(db.index.astype(int), db[col], kind="nearest")
                resampled_values = interpolator(target_timestamps.astype(int))
                resampled_db[col] = resampled_values
            else:
                print(f"passed {col}")
                pass

        # Create resampled dataframe
        resampled_db["Serial_No"] = serial_no
        resampled_db = pd.DataFrame(
            resampled_db, index=resampled_db["TimeStamp"].values
        )
        resampled_db.drop(columns="TimeStamp", inplace=True)

        print("DataFrame resampled")

        return resampled_db

    # Method to convert datestrings into a datetime usable format. Method is used in the prepTrends method
    def convert_datetime(self, datestring):
        """
        Method to convert the datetime format of the IoT exports.
        Input:
        - datestring: Timestamp to be converted

        Return:
        - new_datetime_str: Converted timestamp as str
        """
        # Parse the datetime string using dateutil.parser.parse()
        ds = datestring
        datetime_obj = dateutil.parser.parse(ds)

        # Format the datetime object as a string in the desired format
        new_datetime_str = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
        return new_datetime_str

    # Method to preprocess the combined database consisting of trends and alerts data
    def trimDB(self, db):
        """Function to handle missing values within DataFrame

        Function applies different filling methods for each parameter to fill the missing values.
        Sensor-Parameters: Are filled by using the padding method. This fills the missin values until the next given value with the first value.
                           This helps to acknowledge the tolerances of the different parameters.
        Calculated-Parameters: Missing values for calculated values are treated by interpolating them using a linear function.
        Alert Condition: The alert condition value gets
        Input:
        - db: Combined database

        Return:
        - db_new: Preprocessed database
        """

        # Trim DB of not needed columns and convert alert conditions into bool parameters
        cols = ["Reason", "Open duration", "Priority", "Resolved at", "time_difference"]
        db_new = db.drop(columns=cols)

        # Drop duplicated timestamps for single pump
        db_new.drop_duplicates(
            subset="NewTimestamp", keep="first", inplace=True, ignore_index=True
        )
        db_new.set_index("NewTimestamp", drop=True, inplace=True)

        # Fill missing values of RunningHours linear
        db_new.loc[:, "RunningHours"].interpolate(
            method="linear", limit_direction="backward", inplace=True
        )

        # Interpolate missing values based on parameter
        db_new.loc[:, "T_Out"].interpolate(
            method="backfill", limit_direction="backward", inplace=True
        )

        # Fill missing values of TotalCO2Impact and TotalEnergyConsumption
        db_new.loc[:, "TotalCO2Impact"].interpolate(
            method="linear", limit_direction="backward", inplace=True
        )
        db_new.loc[:, "TotalEnergyConsumption"].interpolate(
            method="linear", limit_direction="backward", inplace=True
        )

        # Fill missing values of Current
        db_new.loc[:, "Current"].interpolate(
            method="backfill", limit_direction="backward", inplace=True
        )

        # Fill missing values of P_In
        db_new.loc[:, "P_In"].interpolate(
            method="backfill", limit_direction="backward", inplace=True
        )

        # Fill missing values of P_Out
        db_new.loc[:, "P_Out"].interpolate(
            method="backfill", limit_direction="backward", inplace=True
        )

        # Fill missing values of MotorSpeed and Power
        db_new.loc[:, "MotorSpeed"].interpolate(
            method="backfill", limit=None, limit_direction="backward", inplace=True
        )
        db_new.loc[:, "Power"].interpolate(
            method="backfill", limit=None, limit_direction="backward", inplace=True
        )

        # Padding missing values of Created at
        db_new.loc[:, "Created at"].interpolate(
            method="backfill", limit_direction="backward", inplace=True
        )
        db_new.loc[:, "Serial_No"].interpolate(
            method="backfill", limit_direction="backward", inplace=True
        )

        # Handle far outliers
        mask = list(db_new["T_Out"] > 150)
        db_new.loc[mask, "T_Out"] = 150

        mask = list(db_new["P_In"] > 1500)
        db_new.loc[mask, "P_In"] = 1500

        mask = list(db_new["P_Out"] < 500)
        db_new.loc[mask, "P_Out"] = 1013
        # Fill remaining Na values with 0
        db_new.fillna(0, inplace=True)

        # Optional: Resample dataset to interval of 1 min
        # db_new = self.resampleDataset(db_new, serial_no)
        return db_new

    # Method to combine multiple prepared databases
    def combDatabases(self, cont):
        """
        Method combines multiple prepared databases after the selection of the directory
        where the prepared databases are saved.
        Input:
        - cont: Integer = 0 to initialise while loop

        Return:
        - combined database as csv from self.saveDatabase method
        """

        # While loop that runs a select directory dialogue
        # Return dir = directory path and cont = 1 to end while loop
        while cont == 0:
            dir, cont = self.select_DB_files()

        # Iterating over dictionary and files that are saved in self.file_list if more than 1 file is in the list
        if len(self.file_list) > 1:
            for file in self.file_list:
                # Check if the current file is not the first file
                if file != self.file_list[0]:
                    try:
                        # Create path, load data from path as csv and combine 2 databases using concat method.
                        # Update temp_comb with combined database.
                        path = os.path.join(dir, file)
                        temp_df = pd.read_csv(path)
                        temp_comb = pd.concat([temp_comb, temp_df], axis=0)
                    except UnicodeDecodeError:
                        print(f"File {file} results into UnicodeDecodeError")

                # If only one file is selected initialse temp_comb dataframe
                else:
                    try:
                        print("First file loaded.\n Progressing...")
                        path = os.path.join(dir, file)
                        temp_comb = pd.read_csv(path)
                    except UnicodeDecodeError:
                        print(f"File {file} results into UnicodeDecodeError")

            # Save combined dataframe as self.joined_DB
            self.joined_DB = temp_comb

        # If only one file is in the directory save file with new path
        else:
            print("Only one file selected.")
            path = os.path.join(dir, self.file_list[0])
            self.joined_DB = pd.read_csv(path)

        # Save the database as csv
        text = self.saveDatabase(self.joined_DB)
        print("Database created.")
        return text

    # Method to select the directory in which the prepared databases are saved.
    def select_DB_files(self):
        """
        Method opens dialogue to navigate to the directory where the prepared databases
        are saved and generates list with file paths in the directory.
        Input:
        - Directory selection
        
        Return:
        - dir_path: Path of the selected directory
        - self.file_list: List of file paths in the selected directory
        - cont = 1 if selection was confirmed, 0 if selection was disregarded
        """

        # Open file dialogue to select directory
        dir_path = filedialog.askdirectory(title="Select directory")
        print(dir_path)

        # Save paths of files in the directory as list as self-variable from the class
        self.file_list = os.listdir(dir_path)
        # Dialogue to confirm the selection and set cont based on the input
        cont = messagebox.askyesno(
            title="Select Directory", message="Did you choose the correct directory?"
        )
        return dir_path, cont

    # Method to save the database
    def saveDatabase(self, database):
        """
        Method to save the combined FullDatabase as csv or excel.
        Input:
        - database: Combined database

        Return:
        - FullDatabase csv after call of the method create_file
        """

        # Message box to decide if database will be saved as csv or excel
        self.file_type = messagebox.askyesno(
            title="Save Database",
            message="How do you want to save the database?\nAs .csv (YES)\n As .xlsx",
        )

        # Generate string of current datae and time and save as strings
        currentDateTime = datetime.now()
        timestring = currentDateTime.strftime("%H%M%S")
        datestring = str(currentDateTime.date())

        # Create path where to save the FullDatabase as specified below
        path = os.path.join(
            "C:/Users/a00546973/Desktop/Database/FullDatabase_"
            + datestring
            + "_"
            + timestring
            + ".csv"
        )

        # Call create_file method to save the database
        self.create_file(database, path, self.file_type)
        text = f"\n\nDatabase saved in the following directory:\n{path}\n\n\n\n"

        return text

    # Method to generate a csv or excel file
    def create_file(self, df, path, filetype):
        """
        Method will save a given Dataframe at the the specified location (path).
        The filetype determines if the is to be saved as csv or excel.
        Input:
        - df: Database to be saved as csv or excel
        - path: Path where to save the file
        - filetype: 1 = csv or 0 = excel file

        Return:
        - Csv or excel file of database depending on filetype.
        """

        if filetype == 1:
            df.to_csv(path, sep=",")
        elif filetype == 0:
            df.to_excel(path)
        else:
            messagebox.showerror(title="Error", text="File format not supported.")
