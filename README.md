Here’s a streamlined version of your README.md that you can copy and paste, including both the cPCA description and instructions for generating the requirements.txt file, without added descriptions of functions:

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

	2.	Install dependencies using the requirements.txt file:

pip install -r requirements.txt

	3.	Ensure your data is preprocessed (centered and standardized) and stored as a .parquet file before using the class.

Generating requirements.txt

In case you need to regenerate or update the requirements.txt file, follow these steps:

1. Create and activate a virtual environment (optional but recommended):

	•	On macOS/Linux:

python3 -m venv env
source env/bin/activate


	•	On Windows:

python -m venv env
env\Scripts\activate



2. Install necessary dependencies

Install all the necessary packages for your project. For example:

pip install numpy pandas scikit-learn

3. Generate the requirements.txt file

After installing the required packages, generate a requirements.txt file by running:

pip freeze > requirements.txt

This will create a requirements.txt file that lists all installed packages with their version numbers.

Usage

1. Import and Setup

Make sure to import necessary modules and configure logging for detailed tracking:

from contrastive_pca import ContrastivePCA
import logging
from logging_config import setup_logging

setup_logging()  # Initialize logging

2. Initialize Contrastive PCA

cpca = ContrastivePCA(data_path="path/to/data.parquet", grid=False, alpha=1000, num_components=550)

3. Running cPCA

To perform contrastive PCA and save the projected data:

cpca.contrast(column="MetaLabel", controls=["Control1", "Control2"], output_path="path/to/output.parquet")

Future Enhancements

	1.	Automatic hyperparameter tuning: Integrate grid_search to find the optimal alpha and num_components based on silhouette scores.
	2.	Softer component selection: Move from a hard cut-off for the number of components to a softer, weight-based selection.
	3.	Visualization: Add plotting capabilities for easier interpretation of the projected data.
