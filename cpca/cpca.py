from typing import List, Optional, Tuple
import sys
import functools
sys.path.append("..")
import logging
from logging_config import setup_logging, log_method_call
import pandas as pd
import numpy as np
logger = logging.getLogger(__name__)

class ContrastivePCA:
    """
    Version 1.0.0
    
    """

    def __init__(self, data_path: str, alpha: float, num_components: int):
        """
        Constructs necessary attributes (hyperparameters) for ContrastivePCA

        Parameters:
        ===========
        alpha : contrastive constant, can be set or found through grid search
        num_components : number of components to retain, will update to weights in the future, can be set or found through grid search
        data_path: path to dataframe.parquet, initialised as df
        """

        self.alpha = alpha
        if num_components <= 0:
            logger.error("Non-positive num_components passed")
            raise ValueError(f"{num_components} must be positive")
        self.num_components = num_components
        self.df = pd.read_parquet(data_path)
        self.feat_len = self.df[self.df.columns[~self.df.columns.str.startswith("Meta")]].shape[1]
    
    @log_method_call
    def contrast(self, column: str, controls: List[str], output_path: str) -> None:
        """
        Computes contrastive PCA and stores projection of entire df (contrasts and targets) in output path.
        This function calls many external functions listed below
        # Fill with function calls

        Parameters:
        ===========
        output_path: path to store output parquet file "needs to be .parquet"
        """

        # Extract controls from df
        control_indices, target_indices = self.extract_controls(column, controls)

        # Compute covariances from indices
        target_cov, control_cov = self.compute_covariances(target_indices, control_indices)

        # Contrast between target_cov and control_cov, retain only num_components
        projection_basis = self.compute_contrastive_PCA(target_cov, control_cov)

        # Project df onto basis
        df_projected = self.project_data(projection_basis)

        # Save to output path
        pd.DataFrame.to_parquet(output_path)
    
    @log_method_call
    def extract_controls(self, column: str, controls: List[str]) -> Tuple[pd.Index, pd.Index]:
        """
        Extracts indices of controls, the non-control indices are targets

        Parameters:
        ==========
        column: Metadata column in data to obtain indices corresponding to controls
        controls: List of metadata to be pooled to controls

        Returns:
        ========
        (pd.Index, pd.Index)
            (Control indices, Target indices)

        """

        controls_index_list = [self.df[self.df[column] == control].index for control in controls]
        controls_indices = functools.reduce(lambda x, y: x.union(y), controls_index_list) # lambda is a throwaway function
        target_indices = self.df.drop(controls_indices).index

        return Tuple[controls_indices, target_indices]

    @log_method_call
    def compute_covariances(self, target_indices: pd.Index, control_indices: pd.Index) ->  Tuple[np.ndarray, np.ndarray]:
        logger.error("compute_covariances not implemented")
        raise NotImplementedError

    @log_method_call
    def compute_contrastivePCA(self, target_covariance: np.ndarray, control_covariance: np.ndarray) -> np.ndarray:
        logger.error("compute_contrastivePCA not implemented")
        raise NotImplementedError

    @log_method_call
    def project_data(self, basis: np.ndarray) -> pd.DataFrame:
        logger.error("project_data not implemented")
        raise NotImplementedError