# PLP-WEEK-6
Plp Assignments on Python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
def load_data():
    try:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

df = load_data()
if df is not None:
    # Task 1: Compute Basic Statistics
    print("Basic Statistics:")
    print(df.describe())
    
    # Task 2: Grouping Analysis
    print("\nMean values grouped by species:")
    print(df.groupby('species').mean())
    
    # Task 3: Data Visualization
    plt.figure(figsize=(12, 8))
    
    # Line Chart: Trend over Index (Simulating time-series for petal length)
    plt.subplot(2, 2, 1)
    plt.plot(df.index, df['petal length (cm)'], label='Petal Length', color='blue')
    plt.title('Petal Length Over Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    
    # Bar Chart: Average Petal Length per Species
    plt.subplot(2, 2, 2)
    sns.barplot(x='species', y='petal length (cm)', data=df, palette='viridis')
    plt.title('Average Petal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Petal Length (cm)')
    
    # Histogram: Distribution of Sepal Length
    plt.subplot(2, 2, 3)
    plt.hist(df['sepal length (cm)'], bins=15, color='green', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Sepal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Frequency')
    
    # Scatter Plot: Sepal Length vs Petal Length
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
else:
    print("Data loading failed.")
