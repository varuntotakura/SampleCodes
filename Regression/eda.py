import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon, ttest_ind, spearmanr, pearsonr
from typing import Union, List, Tuple, Dict, Any
import os

class ExploratoryAnalysis:
    """
    A comprehensive class for performing exploratory data analysis on regression datasets.
    Handles multiple data types: numerical, categorical, boolean, and text.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the EDA class with a dataset.
        
        Args:
            data (pd.DataFrame): Input dataset for analysis
        """
        self.data = data
        self.plots_dir = 'plots'
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        
    def _get_column_type(self, column: str) -> str:
        """
        Determine the type of the column for appropriate analysis.
        
        Args:
            column (str): Column name to analyze
            
        Returns:
            str: Column type ('numeric', 'categorical', 'boolean', or 'text')
        """
        if self.data[column].dtype in ['int64', 'float64']:
            return 'numeric'
        elif self.data[column].dtype == 'bool':
            return 'boolean'
        elif self.data[column].nunique() < 10:  # Assuming small number of unique values is categorical
            return 'categorical'
        else:
            return 'text'
        
    def univariate_analysis(self, column: str) -> Dict[str, Any]:
        """
        Perform comprehensive univariate analysis based on column type without visualizations.

        Args:
            column (str): Column name to analyze

        Returns:
            Dict[str, Any]: Dictionary containing statistical measures
        """
        col_type = self._get_column_type(column)
        stats_dict = {}

        if col_type in ['numeric', 'boolean']:
            # Add normality test
            _, normality_p_value = stats.normaltest(self.data[column].dropna())
            stats_dict.update({
                'normality_test_p_value': normality_p_value,
                'is_normal': normality_p_value > 0.05
            })
            
            stats_dict.update({
                'count': self.data[column].count(),
                'mean': self.data[column].mean(),
                'median': self.data[column].median(),
                'mode': self.data[column].mode()[0] if not self.data[column].mode().empty else None,
                'min': self.data[column].min(),
                'max': self.data[column].max(),
                'std': self.data[column].std(),
                'variance': self.data[column].var(),
                'skewness': self.data[column].skew(),
                'kurtosis': self.data[column].kurtosis(),
                'q1': self.data[column].quantile(0.25),
                'q3': self.data[column].quantile(0.75),
                'iqr': self.data[column].quantile(0.75) - self.data[column].quantile(0.25),
                'missing_values': self.data[column].isnull().sum()
            })

        elif col_type in ['categorical', 'text']:
            value_counts = self.data[column].value_counts()
            proportions = self.data[column].value_counts(normalize=True)

            stats_dict.update({
                'count': self.data[column].count(),
                'unique_values': self.data[column].nunique(),
                'mode': self.data[column].mode()[0] if not self.data[column].mode().empty else None,
                'missing_values': self.data[column].isnull().sum(),
                'value_counts': value_counts.to_dict(),
                'proportions': proportions.to_dict()
            })

        return stats_dict
    
    def bivariate_analysis(self, x_column: str, y_column: str) -> Dict[str, Any]:
        """
        Perform bivariate analysis between two variables without visualizations.

        Args:
            x_column (str): First variable
            y_column (str): Second variable

        Returns:
            Dict[str, Any]: Dictionary containing analysis results
        """
        x_type = self._get_column_type(x_column)
        y_type = self._get_column_type(y_column)

        results = {}

        if x_type == 'numeric' and y_type == 'numeric':
            pearson_corr, p_pearson = pearsonr(self.data[x_column].dropna(), self.data[y_column].dropna())
            spearman_corr, p_spearman = spearmanr(self.data[x_column], self.data[y_column])

            results.update({
                'pearson_correlation': pearson_corr,
                'pearson_p_value': p_pearson,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': p_spearman
            })

        elif (x_type in ['categorical', 'boolean', 'text']) and y_type == 'numeric':
            group_stats = self.data.groupby(x_column)[y_column].agg(['mean', 'std', 'count', 'min', 'max'])
            results['group_statistics'] = group_stats.to_dict()

            try:
                groups = [group for name, group in self.data.groupby(x_column)[y_column] if len(group) > 1]
                if len(groups) >= 2:
                    f_stat, p_value = stats.f_oneway(*groups)
                    results.update({
                        'anova_f_statistic': f_stat,
                        'anova_p_value': p_value
                    })
            except Exception as e:
                results['anova_error'] = str(e)

        return results

    def univariate_analysis_plot(self, column: str) -> Dict[str, Any]:
        """
        Perform comprehensive univariate analysis based on column type.
        
        Args:
            column (str): Column name to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing statistical measures
        """
        col_type = self._get_column_type(column)
        stats_dict = {}
        
        if col_type in ['numeric', 'boolean']:
            stats_dict.update({
                'mean': self.data[column].mean(),
                'median': self.data[column].median(),
                'std': self.data[column].std(),
                'variance': self.data[column].var(),
                'skewness': self.data[column].skew(),
                'kurtosis': self.data[column].kurtosis(),
                'q1': self.data[column].quantile(0.25),
                'q3': self.data[column].quantile(0.75),
                'iqr': self.data[column].quantile(0.75) - self.data[column].quantile(0.25)
            })
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Distribution plot
            sns.histplot(data=self.data, x=column, kde=True, ax=ax1)
            ax1.set_title(f'Distribution of {column}')
            
            # Box plot
            sns.boxplot(y=self.data[column], ax=ax2)
            ax2.set_title(f'Box Plot of {column}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'univariate_{column}.png'))
            plt.close()
            
        elif col_type in ['categorical', 'text']:
            # Value counts and proportions
            value_counts = self.data[column].value_counts()
            proportions = self.data[column].value_counts(normalize=True)
            
            stats_dict.update({
                'unique_values': self.data[column].nunique(),
                'mode': self.data[column].mode()[0],
                'value_counts': value_counts.to_dict(),
                'proportions': proportions.to_dict()
            })
            
            # Bar plot for categorical variables
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.data, y=column, order=value_counts.index)
            plt.title(f'Distribution of {column}')
            plt.savefig(os.path.join(self.plots_dir, f'categorical_{column}.png'))
            plt.close()
        
        return stats_dict
    
    def bivariate_analysis_plot(self, x_column: str, y_column: str) -> Dict[str, Any]:
        """
        Perform bivariate analysis between two variables based on their types.
        
        Args:
            x_column (str): First variable
            y_column (str): Second variable
            
        Returns:
            Dict[str, Any]: Dictionary containing analysis results
        """
        x_type = self._get_column_type(x_column)
        y_type = self._get_column_type(y_column)
        
        results = {}
        
        if x_type == 'numeric' and y_type == 'numeric':
            # Correlation analysis for numeric variables
            pearson_corr, p_value_pearson = pearsonr(self.data[x_column], self.data[y_column])
            spearman_corr, p_value_spearman = spearmanr(self.data[x_column], self.data[y_column])
            
            results.update({
                'pearson_correlation': pearson_corr,
                'pearson_p_value': p_value_pearson,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': p_value_spearman
            })
            
            # Scatter plot with regression line
            plt.figure(figsize=(10, 6))
            sns.regplot(data=self.data, x=x_column, y=y_column)
            plt.title(f'Scatter Plot: {x_column} vs {y_column}')
            plt.savefig(os.path.join(self.plots_dir, f'scatter_{x_column}_vs_{y_column}.png'))
            plt.close()
            
        elif (x_type in ['categorical', 'boolean', 'text']) and y_type == 'numeric':
            # Group statistics
            group_stats = self.data.groupby(x_column)[y_column].agg(['mean', 'std', 'count'])
            results['group_statistics'] = group_stats.to_dict()
            
            # Box plot for categorical vs numeric
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=self.data, x=x_column, y=y_column)
            plt.xticks(rotation=45)
            plt.title(f'Box Plot: {x_column} vs {y_column}')
            plt.savefig(os.path.join(self.plots_dir, f'boxplot_{x_column}_vs_{y_column}.png'))
            plt.close()
            
            # ANOVA test
            groups = [group for name, group in self.data.groupby(x_column)[y_column]]
            f_stat, p_value = stats.f_oneway(*groups)
            results.update({
                'anova_f_statistic': f_stat,
                'anova_p_value': p_value
            })
            
        return results
    
    def statistical_tests(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
        """
        Perform statistical tests between two groups.
        
        Args:
            group1 (np.ndarray): First group of observations
            group2 (np.ndarray): Second group of observations
            
        Returns:
            Dict[str, float]: Dictionary containing test results
        """
        # T-test
        t_stat, t_p_value = ttest_ind(group1, group2)
        
        # Wilcoxon test
        try:
            w_stat, w_p_value = wilcoxon(group1, group2)
        except ValueError:
            w_stat, w_p_value = np.nan, np.nan
            
        # Additional tests
        # Mann-Whitney U test
        u_stat, u_p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.ks_2samp(group1, group2)
            
        return {
            't_statistic': t_stat,
            't_p_value': t_p_value,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_p_value': w_p_value,
            'mannwhitney_statistic': u_stat,
            'mannwhitney_p_value': u_p_value,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p_value
        }
    
    def decile_analysis(self, target: str, prediction: str) -> Tuple[pd.DataFrame, float]:
        """
        Perform decile analysis and calculate lift.
        
        Args:
            target (str): Actual target variable column name
            prediction (str): Predicted values column name
            
        Returns:
            Tuple[pd.DataFrame, float]: Decile summary and Gini coefficient
        """
        df = self.data.copy()
        df['Decile'] = pd.qcut(df[prediction], q=10, labels=False)
        
        decile_summary = df.groupby('Decile').agg({
            target: ['count', 'mean', 'std', 'min', 'max'],
            prediction: ['mean', 'std', 'min', 'max']
        }).round(3)
        
        # Calculate lift
        baseline = df[target].mean()
        decile_summary['Lift'] = decile_summary[(target, 'mean')] / baseline
        
        # Calculate cumulative lift
        decile_summary['Cumulative_Lift'] = decile_summary['Lift'].cumsum() / (np.arange(10) + 1)
        
        # Plot lift and cumulative lift chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Lift chart
        decile_summary['Lift'].plot(marker='o', ax=ax1)
        ax1.set_title('Decile Lift Chart')
        ax1.set_xlabel('Decile')
        ax1.set_ylabel('Lift')
        ax1.grid(True)
        
        # Cumulative lift chart
        decile_summary['Cumulative_Lift'].plot(marker='o', ax=ax2)
        ax2.set_title('Cumulative Lift Chart')
        ax2.set_xlabel('Decile')
        ax2.set_ylabel('Cumulative Lift')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate Gini coefficient
        sorted_predictions = df.sort_values(prediction, ascending=False)
        lorenz_curve = np.cumsum(sorted_predictions[target]) / sum(sorted_predictions[target])
        gini = 1 - 2 * sum(lorenz_curve) / len(lorenz_curve)
        
        return decile_summary, gini
    
    def correlation_matrix(self, method: str = 'pearson', include_types: List[str] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix using specified method.
        
        Args:
            method (str): Correlation method ('pearson' or 'spearman')
            include_types (List[str]): Types of columns to include ('numeric', 'boolean')
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        if include_types is None:
            include_types = ['numeric', 'boolean']
        
        # Select columns of specified types
        cols_to_include = [col for col in self.data.columns 
                          if self._get_column_type(col) in include_types]
        
        # Calculate correlation matrix
        corr_matrix = self.data[cols_to_include].corr(method=method)
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 8))
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    annot=True, 
                    cmap='coolwarm', 
                    center=0,
                    fmt='.2f',
                    square=True)
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'correlation_matrix_{method}.png'))
        plt.close()
        
        return corr_matrix
    
    def describe_data(self, columns: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate descriptive statistics for specified columns based on their data types.
        
        Args:
            columns (List[str], optional): List of column names to describe. 
                                        If None, describes all columns.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with data type as key and describe results as value
        """
        if columns is None:
            columns = self.data.columns
            
        # Group columns by type
        numeric_cols = []
        categorical_cols = []
        boolean_cols = []
        text_cols = []
        
        for col in columns:
            col_type = self._get_column_type(col)
            if col_type == 'numeric':
                numeric_cols.append(col)
            elif col_type == 'categorical':
                categorical_cols.append(col)
            elif col_type == 'boolean':
                boolean_cols.append(col)
            else:
                text_cols.append(col)
        
        results = {}
        
        # Numeric description
        if numeric_cols:
            results['numeric'] = self.data[numeric_cols].describe()
            
        # Categorical and text description
        if categorical_cols or text_cols:
            cat_text_cols = categorical_cols + text_cols
            results['categorical'] = self.data[cat_text_cols].describe(include=['object', 'category'])
            
        # Boolean description
        if boolean_cols:
            bool_desc = self.data[boolean_cols].describe()
            bool_desc.loc['true_count'] = self.data[boolean_cols].sum()
            bool_desc.loc['false_count'] = len(self.data) - self.data[boolean_cols].sum()
            results['boolean'] = bool_desc
            
        return results
    
    def compare_distributions(self, column: str, group_column: str = None, group1: np.ndarray = None, group2: np.ndarray = None) -> Dict[str, Any]:
        """
        Compare distributions using Mann-Whitney U test and Kolmogorov-Smirnov test.
        Can either compare two groups within a column or two separate arrays.
        
        Args:
            column (str): Column name to analyze (if using group_column)
            group_column (str, optional): Column name containing group labels
            group1 (np.ndarray, optional): First group of values
            group2 (np.ndarray, optional): Second group of values
            
        Returns:
            Dict[str, Any]: Dictionary containing test results and visualizations
        """
        results = {}
        
        # If group_column is provided, split data into groups
        if group_column is not None:
            unique_groups = self.data[group_column].unique()
            if len(unique_groups) < 2:
                raise ValueError("Need at least 2 groups to compare distributions")
            
            group1 = self.data[self.data[group_column] == unique_groups[0]][column]
            group2 = self.data[self.data[group_column] == unique_groups[1]][column]
            group_labels = [str(unique_groups[0]), str(unique_groups[1])]
        else:
            if group1 is None or group2 is None:
                raise ValueError("Must provide either group_column or both group1 and group2")
            group_labels = ['Group 1', 'Group 2']
        
        # Perform Mann-Whitney U test
        mw_stat, mw_pval = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        results['mannwhitney'] = {
            'statistic': mw_stat,
            'p_value': mw_pval
        }
        
        # Perform Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(group1, group2)
        results['ks_test'] = {
            'statistic': ks_stat,
            'p_value': ks_pval
        }
        
        # Visualize distributions
        plt.figure(figsize=(15, 5))
        
        # KDE plot
        plt.subplot(1, 2, 1)
        sns.kdeplot(data=group1, label=group_labels[0])
        sns.kdeplot(data=group2, label=group_labels[1])
        plt.title('Distribution Comparison (KDE)')
        plt.legend()
        
        # Box plot
        plt.subplot(1, 2, 2)
        combined_data = pd.DataFrame({
            'values': np.concatenate([group1, group2]),
            'groups': np.repeat(group_labels, [len(group1), len(group2)])
        })
        sns.boxplot(data=combined_data, x='groups', y='values')
        plt.title('Distribution Comparison (Box Plot)')
        
        plt.tight_layout()
        plt.show()
        
        # Add descriptive statistics
        results['descriptive_stats'] = {
            group_labels[0]: {
                'mean': np.mean(group1),
                'median': np.median(group1),
                'std': np.std(group1),
                'iqr': np.percentile(group1, 75) - np.percentile(group1, 25)
            },
            group_labels[1]: {
                'mean': np.mean(group2),
                'median': np.median(group2),
                'std': np.std(group2),
                'iqr': np.percentile(group2, 75) - np.percentile(group2, 25)
            }
        }
        
        return results

    def basic_summary(self):
        """Print basic summary statistics of the dataset"""
        print("\nDataset Info:")
        print(self.data.info())
        
        print("\nNumerical Features Summary:")
        print(self.data.describe())
        
        print("\nMissing Values:")
        missing = self.data.isnull().sum()
        if missing.any():
            print(missing[missing > 0])
        else:
            print("No missing values found")
            
        print("\nFeature Correlations with Target:")
        target = self.data.columns[-1]  # Assuming target is the last column
        correlations = self.data.corr()[target].sort_values(ascending=False)
        print(correlations)

    def plot_distributions(self):
        """Plot distributions of numerical features and target variable"""
        try:
            # Create plots directory if it doesn't exist
            os.makedirs(self.plots_dir, exist_ok=True)
            
            # Get numeric columns
            numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
            n_cols = len(numeric_cols)
            n_rows = (n_cols + 2) // 3  # 3 plots per row
            
            # Distribution plots
            plt.figure(figsize=(15, 5 * n_rows))
            for idx, col in enumerate(numeric_cols, 1):
                plt.subplot(n_rows, 3, idx)
                sns.histplot(data=self.data, x=col, kde=True)
                plt.title(f'Distribution of {col}')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'feature_distributions.png'))
            plt.close()
            
            # Correlation heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(self.data[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlations')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'correlation_heatmap.png'))
            plt.close()
            
            # Boxplots for outlier detection
            plt.figure(figsize=(15, 5))
            self.data[numeric_cols].boxplot()
            plt.title('Feature Distributions (Box Plots)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'feature_boxplots.png'))
            plt.close()
            
            # Scatter plots against target
            target = self.data.columns[-1]  # Assuming target is the last column
            n_features = len(numeric_cols) - 1  # Exclude target
            n_rows = (n_features + 2) // 3
            
            plt.figure(figsize=(15, 5 * n_rows))
            plot_idx = 1
            for col in numeric_cols:
                if (col != target):
                    plt.subplot(n_rows, 3, plot_idx)
                    plt.scatter(self.data[col], self.data[target], alpha=0.5)
                    plt.xlabel(col)
                    plt.ylabel(target)
                    plt.title(f'{col} vs {target}')
                    plot_idx += 1
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'feature_target_relationships.png'))
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not generate one or more plots - {str(e)}")
            import logging
            logging.error(f"Error in plot_distributions: {str(e)}")