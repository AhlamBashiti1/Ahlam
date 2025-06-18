import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.datasets import load_iris
# This code is for loading datasets and plotting histograms and pairplots.
def load_dataset(name):
    if name == 'iris':
        iris_data = load_iris()
        df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
        df['target'] = iris_data.target
        #description=df.describe()
       
        return df #,description
    elif name == 'circles':
        df=pd.read_csv("Data/circles.csv")
        # Check if 'target' column exists, if not create it
           
        #description=df.describe()
        
        return df #,description
    
    elif name == 'moons':
        return pd.read_csv("Data/moons.csv")
    elif name == '3gaussians-std0.6':
        return pd.read_csv("Data/3gaussians-std0.6.csv")
    elif name == '3gaussians-std0.9':
        return pd.read_csv("Data/3gaussians-std0.9.csv")
    else:
        print(f"Dataset '{name}' not recognized.")
        return None

def plot_histograms(df, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    df.select_dtypes(include='number').hist(ax=ax, bins=20)
    plt.suptitle(f"{title} - Histograms", fontsize=16)
    plt.tight_layout()
   
    # Don't call plt.show() here

def plot_pairplot(df, title):
    # sns.pairplot returns a FacetGrid object, which creates its own figure
    pairplot_fig = sns.pairplot(df, hue='target' if 'target' in df.columns else None)
    pairplot_fig.fig.suptitle(f"{title} - Pairplot", fontsize=16, y=1.02)
    # Don't call plt.show() here
# Plot distributions
def plot_data_distribution(df, title):
    plt.figure(figsize=(6, 5))
    
    if 'target' in df.columns:
        sns.scatterplot(x=df.columns[0], y=df.columns[1], hue='target', data=df, palette='deep')
    else:
        sns.scatterplot(x=df.columns[0], y=df.columns[1], data=df)
    
    plt.title(f"{title} - Data Distribution")
    plt.tight_layout()
    return plt
import matplotlib.pyplot as plt
import seaborn as sns
def plot_data_distributionClor(df, title):
    plt.figure(figsize=(6, 5))
    
    if 'target' in df.columns:
        sns.scatterplot(
            x=df.columns[0],
            y=df.columns[1],
            hue=df['target'].astype(str),  # convert to string to ensure categorical mapping
            data=df,
            palette='tab10',
            s=70  # optional: make points a bit larger
        )
    else:
        sns.scatterplot(x=df.columns[0], y=df.columns[1], data=df)
    
    plt.title(f"{title} - Data Distribution")
    plt.tight_layout()
    return plt



# To check the PCA for Iris dataset
def plotcorrelation_matrix(df, title):
    features = df.drop(columns=['target'], errors='ignore')

        # Compute correlation matrix
    corr = features.corr()

        # Display correlation matrix as heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()



if __name__ == "__main__":
    available_datasets = ['iris', 'circles', 'moons', '3gaussians-std0.6', '3gaussians-std0.9']

    if len(sys.argv) < 2:
        print("Please choose one of the fllowing datasets:")
        for name in available_datasets:
            print("-", name)
        dataset_name = input("Type dataset name here: ").strip()
    else:
        dataset_name = sys.argv[1]

    df = load_dataset(dataset_name)
    if df is None:
        print(f"Dataset '{dataset_name}' not found. Please choose from: {', '.join(available_datasets)}")
    else:
        description = df.describe() 

        print(f"Dataset '{dataset_name}' loaded successfully.")
        print("Description of the dataset:")
        print(description)
        #print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
    if df is not None:

        plot_histograms(df, dataset_name)
        plot_pairplot(df, dataset_name)
        fig=plot_data_distributionClor(df, dataset_name)
        plt.show()  
        fig.show
    