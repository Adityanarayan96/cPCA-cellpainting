# Contrastive PCA (cPCA) - Version 1.0.0

This repository contains a custom implementation of **Contrastive Principal Component Analysis (cPCA)**, an unsupervised learning technique that enhances traditional PCA by contrasting the variance in a target dataset against a control dataset. The implementation provides a flexible framework for users to input their data, configure parameters, and run the analysis.

## Features

1. **Basic cPCA implementation**: 
   - The class provides methods to compute contrastive PCA using user-specified hyperparameters (`alpha` and `num_components`).
   
2. **Customizable parameters**: 
   - `alpha`: A contrastive constant that adjusts the relative weight of the control data covariance in the analysis.
   - `num_components`: The number of principal components to retain.

3. **Control selection**: 
   - Automatically extracts control and target data points based on metadata columns provided by the user.

4. **Output**: 
   - Projects data onto principal components and saves the result in a `.parquet` format.

5. **Logging**: 
   - Method calls and potential errors are logged for easy debugging and transparency.

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_directory>
```
2.	Install dependencies using the requirements.txt file:
```bash
pip install -r requirements.txt
```
3.	Ensure your data is preprocessed (centered and standardized) and stored as a .parquet file before using the class.

**Usage**

1. Import and Setup

Make sure to import necessary modules and configure logging for detailed tracking:
```bash
from contrastive_pca import ContrastivePCA
import logging
from logging_config import setup_logging

setup_logging()  # Initialize logging
```

2. Initialize Contrastive PCA
```bash
cpca = ContrastivePCA(data_path="path/to/data.parquet", grid=False, alpha=1000, num_components=550)
```
3. Running cPCA

To perform contrastive PCA and save the projected data:
```bash
cpca.contrast(column="MetaLabel", controls=["Control1", "Control2"], output_path="path/to/output.parquet")
```

Future Enhancements

1.	Automatic hyperparameter tuning: Integrate grid_search to find the optimal alpha and num_components based on silhouette scores.
2.	Softer component selection: Move from a hard cut-off for the number of components to a softer, weight-based selection.
3.	Visualization: Add plotting capabilities for easier interpretation of the projected data.
