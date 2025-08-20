import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   

class AutomatedDataExploration:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.report = {}
    def satistic_summary(self):
        sats_summary= self.df.describe(include='all')
        self.report['statistic_summary'] = sats_summary
        return sats_summary
    def missing_values(self):
        missing_values= self.df.isnull().sum()
        missing_values= missing_values[missing_values > 0]
        self.report['missing_values']= missing_values
        return missing_values
    def duplicate_values(self):
        duplicate_values= self.df.duplicated().sum()
        self.report['duplicate_values'] = duplicate_values
        return duplicate_values
    def outliers(self):
        outliers= {}
        for col in self.df.select_dtypes(include=np.number).columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers[col] = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].shape[0]
            self.report['outliers'] = outliers
            return outliers        
    def recommenderreport(self):
        recommendations= []
        if self.report.get('missing_values', pd.Series()).sum()>0:
            recommendations.append("Consider handling missing values by imputation or removal.")
        if self.report.get('duplicate_values', 0) > 0:
            recommendations.append("Remove duplicate rows to ensure data integrity.")
        if self.report.get(v for v in self.report.get('outliers', {}).values()):
            recommendations.append("Investigate and handle outliers to improve model performance.")
        self.report['recommendations'] = recommendations
        return recommendations
    def generate_report(self):
        print("Generating Automated Data Exploration Report...")
        print('\n-----------Statical Summary-------------\n')
        print(self.satistic_summary())
        print('\n-----------Missing Values-------------\n')
        print(self.missing_values())
        print('\n-----------Duplicate Values-------------\n')
        print(self.duplicate_values())
        print('\n-----------Outliers-------------\n')
        print(self.outliers())
        print('\n-----------Recommendations-------------\n')
        for rec in self.recommenderreport():
            print(f'-- {rec}')
        return self.report

if __name__ == "__main__":    
    path_of_csv= '/Users/sivakumar/Documents/SCALER/Datasets/tips.csv'
    df= pd.read_csv(path_of_csv)
    eda = AutomatedDataExploration(df)
    report= eda.generate_report()