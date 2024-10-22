from typing import Tuple
import numpy as np
from metrics.silhouette import silhouette_label, silhouette_batch
from logging_config import log_method_call
from cpca.cpca import ContrastivePCA


@log_method_call
def grid_search(data_path: str, 
                alpha_range: Tuple[float, float, int], 
                num_components_range: Tuple[int, int, int],
                column: str="Metadata_JCP2022",
                contrast: str="DMSO",
                label:str = "Metadata_JCP2022",
                batch:str = "Metadata_Source") -> Tuple[float, int]:
    """
    Function to implement a grid search on a random subset of samples to tune hyperparameters

    Parameters:
    ===========
    data_path: Path to dataframe 
    alpha_range: Three dimensional tuple with start end and number of points for alpha, scaled in log dimension
    num_components_range: Three dimensional tuple with start end and number of points for alpha
    column: The column to extract controls based on labels
    contrast: The label value from column for controls
    label: Labels for score computation
    batch: Batch for score computation
    
    Returns:
    ========
    Tuple[float, int]:
    Best values of alpha and num_components to use for contrastivePCA

    Future:
    =======
    Add functionlity to only check random subset of samples for faster search
    Currently only supports silhouette score, update for other metrics
    Option to single grid search, when weights replace num_components

    """
    # Return evenly spaced points, logspace for alpha and linear for num_components
    alpha_search = np.logspace(alpha_range[0],
                               alpha_range[1],
                               alpha_range[2])
    num_components_search = np.arange(num_components_range[0], 
                                      num_components_range[1], 
                                      num_components_range[2])
    
    # Initialize dictionary of scores
    scores ={}

    # Search over range of grid and populate scores dictionary
    for num_components in num_components_search:
        for alpha in alpha_search:
            cpca = ContrastivePCA(data_path, num_components, alpha)
            df = cpca.contrast_no_save(column, contrast)
            scores[(num_components, alpha)] = (silhouette_label(df, label), silhouette_batch(df, label, batch))
    
    return scores
            