import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataTransformationPipeline:
    def __init__(self,config):
        self.config = config
        self.log= logging.getLogger(__name__)

    def load_data(self, path):
        """Load data from a CSV file."""
        self.log.info(f"Loading data from {path}")
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.xlsx'):
            return pd.read_excel(path)
        else:
            self.log.error("Unsupported file format. Please provide a CSV or Excel file.")
            return None
        
    def _handle_missing_values(self, df):
        """Handles missing values based on the configuration."""
        if 'missing_values' in self.config:
            strategy = self.config['missing_values'].get('strategy', 'mean')
            self.log.info(f"Handling missing values using '{strategy}' strategy.")

            if strategy == 'mean':
                df = df.fillna(df.mean(numeric_only=True))
            elif strategy == 'median':
                df = df.fillna(df.median(numeric_only=True))
            elif strategy == 'constant':
                fill_value = self.config['missing_values'].get('fill_value', 0)
                df = df.fillna(fill_value)
            else:
                self.log.warning(f"Unknown missing value strategy: '{strategy}'. Skipping.")
        return df

    def _remove_outliers(self, df):
        """Detects and removes outliers using the IQR method."""
        if 'outliers' in self.config:
            strategy = self.config['outliers'].get('strategy')
            if strategy == 'iqr':
                self.log.info("Detecting and removing outliers using the IQR method.")
                Q1 = df.quantile(0.25, numeric_only=True)
                Q3 = df.quantile(0.75, numeric_only=True)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Filter out rows where any numerical feature is an outlier
                outlier_rows = (
                    (df.select_dtypes(include=np.number) < lower_bound) |
                    (df.select_dtypes(include=np.number) > upper_bound)
                ).any(axis=1)

                df = df[~outlier_rows]
                self.log.info(f"Removed {outlier_rows.sum()} outliers.")
            else:
                self.log.warning(f"Unknown outlier strategy: '{strategy}'. Skipping.")
        return df

    def _scale_features(self, df):
        """Scales numerical features based on the configuration."""
        if 'scaling' in self.config:
            strategy = self.config['scaling'].get('strategy', 'standard')
            self.log.info(f"Scaling features using '{strategy}' strategy.")
            
            numerical_cols = df.select_dtypes(include=np.number).columns
            if numerical_cols.empty:
                self.log.warning("No numerical columns found for scaling.")
                return df
            
            if strategy == 'standard':
                scaler = StandardScaler()
            elif strategy == 'minmax':
                scaler = MinMaxScaler()
            else:
                self.log.warning(f"Unknown scaling strategy: '{strategy}'. Skipping.")
                return df

            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df

    def fit_transform(self, df):
        """
        Applies the transformations to the DataFrame in a specified order.
        """
        self.log.info("Starting data transformation pipeline.")
        df_transformed = df.copy()

        # Define the order of operations
        steps = [
            self._handle_missing_values,
            self._remove_outliers,
            self._scale_features
        ]

        for step in steps:
            df_transformed = step(df_transformed)

        self.log.info("Pipeline completed successfully.")
        return df_transformed


if __name__=='__main__':

    path= '/Users/sivakumar/Documents/SCALER/Datasets/tips.csv'
    pipeline_config= {
        'missing_values': {'strategy': 'median'},
        'outliers': {'strategy': 'IQR'},
        'scaling': {'strategy': 'minmax'}
    }

    pipeline= DataTransformationPipeline(pipeline_config)

    df= pipeline.load_data(path)

    if df is not None:
        print("Data loaded successfully.")
        print(df.head())

        transformed_df= pipeline.fit_transform(df.copy())
        print("Data transformed successfully.")
        print(transformed_df.head())

