import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Configure logging to provide status updates
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EDAVisualizer:
    """
    A class to automate exploratory data analysis and visualization.
    It takes a dataset path, loads the data, and generates a series of
    statistical summaries and plots based on column data types.
    """

    def __init__(self, file_path, output_dir='eda_report'):
        """
        Initializes the EDAVisualizer with a file path and output directory.

        Args:
            file_path (str): Path to the dataset (e.g., 'data.csv').
            output_dir (str): Directory to save the generated plots.
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.df = None
        self.log = logging.getLogger(__name__)

    def _create_output_directory(self):
        """Creates the output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.log.info(f"Output directory '{self.output_dir}' created or already exists.")
        return True

    def load_data(self):
        """
        Loads the dataset from the specified file path.
        Handles CSV and Excel formats.
        """
        if not self._create_output_directory():
            return

        if self.file_path.endswith('.csv'):
            self.df = pd.read_csv(self.file_path)
        elif self.file_path.endswith(('.xls', '.xlsx')):
            self.df = pd.read_excel(self.file_path)
        else:
            self.log.error("Unsupported file format. Please provide a .csv or .xlsx file.")
            return

        self.log.info(f"Successfully loaded dataset from '{self.file_path}'.")
        self.log.info(f"Dataset shape: {self.df.shape}")

    def _summarize_data(self):
        """Generates and prints a statistical summary of the dataset."""
        if self.df is None:
            self.log.warning("No data to summarize. Please load a dataset first.")
            return

        self.log.info("\n" + "="*50)
        self.log.info("DATASET OVERVIEW")
        self.log.info("="*50 + "\n")
        self.log.info("First 5 rows:\n" + str(self.df.head()))
        self.log.info("\n" + "-"*50)
        self.log.info("Data types and missing values:\n")
        self.df.info()
        self.log.info("\n" + "-"*50)
        self.log.info("Descriptive statistics for numerical features:\n" + str(self.df.describe()))
        self.log.info("\n" + "="*50 + "\n")

    def _plot_numerical_features(self):
        """Generates plots for all numerical features."""
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        if numerical_cols.empty:
            self.log.info("No numerical features found for plotting.")
            return

        self.log.info("Generating plots for numerical features...")
        for col in numerical_cols:
            # Create a figure with two subplots side by side
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f"Numerical Analysis of '{col}'", fontsize=16)

            # Histogram
            sns.histplot(self.df[col].dropna(), kde=True, ax=axes[0])
            axes[0].set_title('Distribution (Histogram)')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Frequency')

            # Box plot
            sns.boxplot(x=self.df[col], ax=axes[1])
            axes[1].set_title('Outlier Detection (Box Plot)')
            axes[1].set_xlabel(col)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(self.output_dir, f'numerical_{col}.png'))
            plt.close()
        self.log.info(f"Saved plots for {len(numerical_cols)} numerical features to '{self.output_dir}'.")

    def _plot_categorical_features(self):
        """Generates plots for all categorical features."""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if categorical_cols.empty:
            self.log.info("No categorical features found for plotting.")
            return

        self.log.info("Generating plots for categorical features...")
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(y=self.df[col], order=self.df[col].value_counts().index)
            plt.title(f"Frequency of '{col}'")
            plt.xlabel('Count')
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'categorical_{col}.png'))
            plt.close()
        self.log.info(f"Saved plots for {len(categorical_cols)} categorical features to '{self.output_dir}'.")
    
    def _plot_correlation_matrix(self):
        """Generates a heatmap of the correlation matrix for numerical features."""
        numerical_df = self.df.select_dtypes(include=np.number)
        if numerical_df.empty or numerical_df.shape[1] < 2:
            self.log.info("Not enough numerical features (at least 2) to plot a correlation matrix.")
            return
            
        self.log.info("Generating correlation matrix heatmap...")
        plt.figure(figsize=(10, 8))
        correlation_matrix = numerical_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'))
        plt.close()
        self.log.info(f"Saved correlation matrix heatmap to '{self.output_dir}'.")

    def generate_eda_report(self):
        """
        Orchestrates the entire EDA report generation process.
        """
        self.load_data()
        if self.df is None:
            return

        self._summarize_data()
        self._plot_numerical_features()
        self._plot_categorical_features()
        self._plot_correlation_matrix()

        self.log.info(f"EDA report generation complete. Check the '{self.output_dir}' directory for plots.")


# --- Usage Example ---
if __name__ == '__main__':
    # 1. Instantiate and run the EDAVisualizer
    eda_tool = EDAVisualizer(file_path='/Users/sivakumar/Documents/SCALER/Datasets/tips.csv')
    eda_tool.generate_eda_report()
