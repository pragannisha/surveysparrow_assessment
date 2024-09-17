import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def load_datasets(data_folder, sample_size=10000):
    """
    Load all CSV datasets from the given folder and return a sample.
    """
    datasets = []
    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(data_folder, file)
            df = pd.read_csv(file_path)
            df = convert_dtypes(df)
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=1)
            datasets.append(df)
    return datasets

def convert_dtypes(df):
    """
    Convert dataframe columns to more memory-efficient types.
    """
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

def reduce_cardinality(df, column, threshold=10):
    """
    Group less frequent categories into 'Other' to reduce cardinality.
    """
    value_counts = df[column].value_counts()
    rare_categories = value_counts[value_counts < threshold].index
    df[column] = df[column].replace(rare_categories, 'Other')
    return df

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, scaling numerical features,
    and encoding categorical features.
    """
    # Separate features and target
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    # Reduce cardinality of categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X = reduce_cardinality(X, col)
    
    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    
    # Define pipelines
    num_pipeline = Pipeline([ 
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([ 
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse=True, handle_unknown='ignore'))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        sparse_threshold=0.0
    )
    
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y

def preprocess_data_chunked(df, chunk_size=10000):
    """
    Process large datasets in chunks to avoid memory issues.
    """
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    # Reduce cardinality of categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X = reduce_cardinality(X, col)
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    
    num_pipeline = Pipeline([ 
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([ 
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse=True, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        sparse_threshold=0.0
    )
    
    # Process data in chunks
    processed_chunks = []
    for start in range(0, len(X), chunk_size):
        end = min(start + chunk_size, len(X))
        X_chunk = X.iloc[start:end]
        y_chunk = y.iloc[start:end]
        X_processed_chunk = preprocessor.fit_transform(X_chunk)
        processed_chunks.append((X_processed_chunk, y_chunk))
    
    # Combine all chunks
    X_processed = np.vstack([chunk[0] for chunk in processed_chunks])
    y_processed = pd.concat([chunk[1] for chunk in processed_chunks])
    
    return X_processed, y_processed

def preprocess_datasets(data_folder):
    """
    Load and preprocess all datasets from the given folder.
    """
    datasets = load_datasets(data_folder)
    processed_datasets = [preprocess_data_chunked(df) for df in datasets]
    return processed_datasets


def eda_plots(X, y):
    """
    Generate EDA plots and return them as byte streams.
    """
    # Create DataFrame for easier plotting
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.DataFrame(y, columns=['churn'])
    
    # Create a byte stream for each plot
    plots = {}
    
    # Target variable distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='churn', data=y_df)
    plt.title('Churn Variable Distribution')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['churn_distribution'] = buf
    
    # Feature distributions
    plt.figure(figsize=(12, 8))
    sns.histplot(X_df, kde=True, bins=30, palette="viridis")
    plt.title('Feature Distributions')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['feature_distributions'] = buf
    
    # Feature correlation matrix
    plt.figure(figsize=(12, 8))
    correlation_matrix = X_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['correlation_matrix'] = buf
    
    # Pairplot for feature relationships
    if X_df.shape[1] <= 10:
        plt.figure(figsize=(12, 8))
        sns.pairplot(X_df)
        plt.title('Pairplot of Features')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['pairplot'] = buf
    
    # Boxplots for feature distributions
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(X_df.columns):
        plt.subplot(len(X_df.columns)//3 + 1, 3, i + 1)
        sns.boxplot(x=y_df['churn'], y=X_df[feature])
        plt.title(f'Boxplot of {feature}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['boxplots'] = buf
    
    # Feature vs. Churn scatter plots
    if X_df.shape[1] <= 10:
        plt.figure(figsize=(12, 8))
        for feature in X_df.columns:
            plt.subplot(len(X_df.columns)//3 + 1, 3, list(X_df.columns).index(feature) + 1)
            sns.scatterplot(x=X_df[feature], y=y_df['churn'])
            plt.title(f'{feature} vs Churn')
            plt.xlabel(feature)
            plt.ylabel('Churn')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['scatter_plots'] = buf
    
    return plots