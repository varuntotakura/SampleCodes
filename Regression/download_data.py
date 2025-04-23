from sklearn.datasets import fetch_california_housing
import pandas as pd

# Download California housing dataset
housing = fetch_california_housing()

# Create a pandas DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target

# Save to CSV
df.to_csv('housing.csv', index=False)
print("Housing dataset has been downloaded and saved to housing.csv")
print(f"Dataset shape: {df.shape}")
print("\nFeatures:")
for name in housing.feature_names:
    print(f"- {name}")
print(f"\nTarget: House price (in $100,000s)")