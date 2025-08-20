import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Dashboard:        
    def __init__(self, df, max_plots=6):
        self.df= df.copy()
        self.max_plots = max_plots
    def remove_unused_axes(self, axes, used_count):
        for i in range(used_count, len(axes)):
            fig= axes[i].figure
            fig.delaxes(axes[i])
    def create_dashboard(self):
        if self.df.empty:
            print("DataFrame is empty. Please provide a valid DataFrame.")
            return
        print("Creating Multi-Panel Visualization Dashboard...")
        num_cols = len(self.df.columns)
        num_rows = int(num_cols**0.5)
        num_cols_grid= (num_cols + num_rows - 1) // num_rows
        fig, axes = plt.subplots(num_rows, num_cols_grid, figsize=(5*num_cols_grid, 5*num_rows), constrained_layout=True)
        if isinstance(axes, np.ndarray):
            axes= axes.flatten()
        else:
            axes= [axes]
        for i, col in enumerate(self.df.columns):
            if i>= len(axes):
                break
            ax= axes[i]
            if pd.api.types.is_numeric_dtype(self.df[col]):
                sns.histplot(data= self.df, x= col, ax=ax, kde=True)
                ax.set_title(f'Distribution of {col}')
            else: 
                sns.countplot(data= self.df, y= col, ax=ax)
                ax.set_title(f'Count Plot of {col}')
            if ax.get_legend():
                ax.get_legend().set_title('')
        self.remove_unused_axes(axes, num_cols)
        plt.suptitle('Multi-Panel Visualization Dashboard', fontsize=16)
        plt.show()

if __name__== "__main__":
    path_for_csv= '/Users/sivakumar/projects/ml_project/Dotkonnet_Workshop/Assignment2/wine+quality/winequality-red.csv'
    df= pd.read_csv(path_for_csv, sep=';')
    dashboard = Dashboard(df)
    dashboard.create_dashboard()
    print("Dashboard created successfully.")

