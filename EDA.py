import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
import seaborn as sns

import matplotlib.pyplot as plt

# Decile Lift Chart
def decile_lift_chart(data, target, score):
    data['decile'] = pd.qcut(data[score], 10, labels=False)
    lift_table = data.groupby('decile').agg(
        total=('decile', 'size'),
        responders=(target, 'sum')
    )
    lift_table['response_rate'] = lift_table['responders'] / lift_table['total']
    lift_table['cumulative_responders'] = lift_table['responders'].cumsum()
    lift_table['cumulative_response_rate'] = lift_table['cumulative_responders'] / lift_table['responders'].sum()
    return lift_table

# T-Test
def perform_t_test(group1, group2):
    t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
    return t_stat, p_value

# ANOVA Test
def perform_anova(*groups):
    f_stat, p_value = f_oneway(*groups)
    return f_stat, p_value

# Chi-Square Test
def perform_chi_square(data, col1, col2):
    contingency_table = pd.crosstab(data[col1], data[col2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p, dof, expected

# Univariate Analysis
def univariate_analysis(data, column):
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std_dev': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'skewness': data[column].skew(),
        'kurtosis': data[column].kurt()
    }
    return stats

# Bivariate Analysis
def bivariate_analysis(data, col1, col2):
    correlation = data[col1].corr(data[col2])
    sns.scatterplot(x=data[col1], y=data[col2])
    plt.title(f'Scatter Plot: {col1} vs {col2}')
    plt.show()
    return correlation

# Multivariate Analysis
def multivariate_analysis(data, target):
    sns.pairplot(data, hue=target)
    plt.show()

# Lorenz Curve
def lorenz_curve(data, column):
    sorted_data = np.sort(data[column])
    cumulative = np.cumsum(sorted_data) / sorted_data.sum()
    cumulative = np.insert(cumulative, 0, 0)
    plt.plot(np.linspace(0, 1, len(cumulative)), cumulative, label='Lorenz Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Equality Line')
    plt.title('Lorenz Curve')
    plt.xlabel('Cumulative Share of Population')
    plt.ylabel('Cumulative Share of Value')
    plt.legend()
    plt.show()