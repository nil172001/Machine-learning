import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path, sep=",", header=0)
    return df

# Compute descriptive statistics
def descriptive_statistics(df):
    print("Basic Statistics:")
    print(df.describe())    # this is used to show the mean, std, min, max, etc. of the data.

# Visualize data distribution
def plot_histograms(df):
    df.hist(figsize=(12, 8), bins=30)
    plt.suptitle("Feature Distributions")
    plt.show()

# Visualize correlations
def plot_correlation_matrix(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

# Detect outliers using boxplots
def plot_boxplots(df):
    plt.figure(figsize=(12, 6))
    df.plot(kind='box', subplots=True, layout=(2, 5), figsize=(15, 8))
    plt.suptitle("Boxplots for Outlier Detection")
    plt.show()

# Perform PCA
def perform_pca(df, n_components=2):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_scaled)
    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 6))
    plt.bar(range(1, n_components+1), explained_variance, alpha=0.7, align='center')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance')
    plt.title('PCA Explained Variance')
    plt.show()

    return principal_components

# Fit a Generalized Additive Model (GAM)
def fit_gam(df):
    # Assuming the target is 'ozone' and the rest are predictors
    X = df.drop('ozone', axis=1).values  # Features (predictors)
    y = df['ozone'].values  # Target variable
    
    # Standardize features (important for GAMs)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit a GAM model (spline-based for each predictor)
    gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8)).fit(X_scaled, y)

    # Plot the effects of each feature
    plt.figure(figsize=(12, 8))
    titles = ['Temperature', 'Inversion Base Height', 'Wind Speed', 'Humidity', 'Vandenberg Height', 'Daggot Pressure Gradient', 'Inversion Base Temperature', 'Visibility', 'Day of Year']
    
    for i, ax in enumerate(gam.partial_dependence()):
        plt.subplot(3, 3, i+1)
        ax.plot()
        plt.title(titles[i])
        plt.tight_layout()

    plt.show()

    # Print model summary
    print(gam.summary())

# Main function
def main():
    file_path = input("Type the name of the file: ")
    df = load_data(file_path)
    descriptive_statistics(df)
    plot_histograms(df)
    plot_correlation_matrix(df)
    plot_boxplots(df)
    principal_components = perform_pca(df)
    fit_gam(df)

main()
