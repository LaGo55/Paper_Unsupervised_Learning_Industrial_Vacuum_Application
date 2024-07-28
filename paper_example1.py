import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import seaborn as sn
import matplotlib.pyplot as plt
import joblib

# Load the preprocessed dataset. It is assumed that it is available as csv for the analysis. The import must be
# adjusted according to the available data format.

class PCA_example():
    
    def __init__(self, path_to_dataset) -> None:
        
        # Load the preprocessed dataset. It is assumed that it is available as csv for the analysis. The import must be
        # adjusted according to the available data format.
        self.dataset = pd.read_csv(path_to_dataset)
        print(self.dataset.head())
        self.pca_dataset = pd.DataFrame()
        self.parameter_comb = {
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
            "10":["P_In","P_Out","T_Out","Current","MotorSpeed","Power","RunningHours","dE","Alert","Reopen count","dP"]
        }
        #Define and initialise iteration 0
        self.iter = 0
        
        
    def __PCA__(self, k, err):
            
        # Perform PCA
        pca = PCA(n_components=k, svd_solver="full", tol=err, whiten=True) # whiten = True -> Correlation matrix, False = Covariance matrix
        self.pca_dataset = pca.fit_transform(self.dataset)
        data_pca_reconstructed = pca.inverse_transform(self.pca_dataset)
        
        return pca, data_pca_reconstructed   
    
    def __compDimReducMethods__(self, data, k, err, metrics):
        """
        The function performs 3 different dimensionality reduction methods (PCA, ICA, NMF) and 
        compares their performance based on performance indicators (MSE, R^2, EV).

        With Args:
        - data: Input data to perform dimensionality reduction on
        - k: Number of components
        - err: Permitted inaccuracy

        Outputs: 
        Array of number of metrics for given number of components k.    
        """
    
        metrics.loc[k-2, "Components"] = k
        
        # Perform PCA
        pca, data_pca_reconstructed = self.__PCA__(k, err)
        
        # Calculate PCA metrics
        metrics.loc[k-2,"PCA Variance"] = pca.explained_variance_ratio_.sum().round(2)
        metrics.loc[k-2,"PCA MSE"] = mean_squared_error(data, data_pca_reconstructed)
        metrics.loc[k-2,"PCA R2"] = r2_score(data, data_pca_reconstructed)
        metrics.loc[k-2,"PCA EV"] = explained_variance_score(data, data_pca_reconstructed)
        del data_pca_reconstructed
        
        #print(f"Iteration {self.iter}-{k}")

    def __dimReductionMethods__(self, data, comp, err):
        """
        The function performs 3 different dimensionality reduction methods (PCA, ICA, NMF) and 
        compares their performance based on performance indicators (MSE, R^2, EV).

        With Args:
        - data: Input data to perform dimensionality reduction on
        - comp: Maximum number of components / dimension to reduce the dataframe to
        - err: Permitted inaccuracy

        Outputs: 
        Array of number of components and corresponding performances of each method.    
        """
        k=2
        metrics = pd.DataFrame()
        
        while k <= comp:
            print("Iteration with: "+str(k)+"/"+str(comp)+ " Components")
            metrics = self.__compDimReducMethods__(data,k,err, metrics)
            k+=1
        
        return metrics
    
    
    def __plot_dim_red__(self, data):
        """
        Method calculates the metrics MSE, R2, and PCA variance for PCA.
        Returns a metrics excel sheet that summarises the results from the metrics and the according plots.
        
        Input:
        - data: Normalised data with specified parameter combination
        """
        # Calculate metrics
        metrics = self.__dimReductionMethods__(data, comp=data.shape[1], err=0.001)
        # Save metrics as Excel file 
        path = str(self.run)+"-"+str(self.iter)+"_dim_red_metrics.xlsx"
        metrics.to_excel(path, index=False)
        
        # Plot metric results
        plt.figure(figsize=(12, 12))
        comp = data.shape[1]
        metric_labels = ["MSE","R2","EV"]

        for i in range(1,len(metric_labels)+1):
            plt.subplot(2, 2, i)
            plt.plot(range(2,comp+1,1), metrics[f"PCA {metric_labels[i-1]}"].values, label='PCA',marker="o")
            #plt.plot(range(1,comp+1,i), metrics["UMAP MSE"].values, label='UMAP')
            plt.xlabel('Number of Components')
            plt.ylabel('Reconstruction Error')
            plt.title(f'Reconstruction Error Comparison ({metric_labels[i-1]}, Iteration {str(self.run)}-{str(self.iter)})')
            plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(range(2,comp+1,1), metrics["PCA Variance"].values, label='PCA', marker="o")
        plt.xlabel('Number of Components')
        plt.ylabel('Cummulative Variance')
        plt.title('PCA Cummulative Variance')
        plt.legend()
        plt.tight_layout()
        
        # Save metrics plot as png
        path = str(self.run) +"-"+str(self.iter)+"_dim_red_metrics"
        plt.savefig(path)
    
    
    def dim_red_results(self, k, data):
        """
        Method plots the reduced data from the first three components of the respective algorithm.
        Returns figure plot saved as png. 
        """
        
        # Define components k, err, and PCA solver
        k = k
        err = 0.001
        solver= "full"
        
        # Fit PCA and transform data
        pca = PCA(n_components=k, tol=err, whiten=True, svd_solver=solver)
        pca = pca.fit(data)
        # Save PCA model as pkl file
        path = "C:/Users/a00546973/Desktop/MasterGENIUS/Models/"+"pca_model_file_GHS2002_"+str(self.run)+"-"+str(self.iter)+".pkl"
        joblib.dump(pca,path)
        # Save PCA reduced data as csv file
        pcs = pca.transform(data)
        path = "C:/Users/a00546973/Desktop/MasterGENIUS/Dimensionality_Reduction/"+"PCA_data_"+str(self.run)+"-"+str(self.iter)+".csv"
        label = pca.get_feature_names_out()
        pcs_data = pd.DataFrame(pcs, columns=label)
        pcs_data.to_csv(path, index=False)
        # Save PCA component description as Excel file
        pcs_df = pd.DataFrame(pca.components_,columns = data.columns, index=label)
        path = "C:/Users/a00546973/Desktop/MasterGENIUS/Dimensionality_Reduction/"+"PCA_results_"+str(self.run)+"-"+str(self.iter)+".xlsx"
        pcs_df.to_excel(excel_writer=path)
        
        fig = plt.figure(figsize=(20,20))

        # PCA subplot
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(pcs[:,0], pcs[:,1], pcs[:,2], s=0.1,alpha=0.2)
        ax1.set_xlabel(pca.get_feature_names_out()[0])
        ax1.set_ylabel(pca.get_feature_names_out()[1])
        ax1.set_zlabel(pca.get_feature_names_out()[2])
        ax1.set_title('PCA reduced')
    
    def __dim_reduction__(self):
        """ 
        Function that executes the methods provided by evaluate_dimRed class in the correct order.
        The function iterates through the specified parameter combinations and thereby specifies the input for the methods.
        """
        
        # Iterate through parameter combinations
        for it in self.parameter_comb.keys():
            
            # Set iteration parameter to parameter combination key
            self.iter = it
            # Select parameter combinations with iteration key from parameter_comb dictionary
            params = self.parameter_comb[it]
            data = self.dataset[params]
            
            # Execute class methods
            print(f"Iteration {it}: \nCalculating metrics...")
            self.__plot_dim_red__(data)
            print(f"Iteration {it}: \nMetrics calculated.\nGenerate plots..")
            self.dim_red_results(4, data)
            print(f"Iteration {it}: \nPlot generated.")
        
        print(f"Dimensionality reduction of run {self.run} finished.")

# Initialise class
#path = "scaled_df_2.csv"
#pca = PCA_example(path)
#pca.__dim_reduction__()