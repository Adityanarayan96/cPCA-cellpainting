from typing import List, Optional, Tuple
import sys
import functools
sys.path.append("..")
import logging
from importlib import reload
reload(logging)
from logging_config import setup_logging, log_method_call
import pandas as pd
import numpy as np
from numpy import linalg as LA
from search.gridsearch import grid_search
logger = logging.getLogger(__name__)

class ContrastivePCA:
    """
    Version 1.0.0
    
    """

    def __init__(self, data_path: str, grid: bool=False, alpha: float=1000, num_components: int=550):
        """
        Constructs necessary attributes (hyperparameters) for ContrastivePCA

        Parameters:
        ===========
        grid: True if alpha, num_components set using grid search of silhoutte scores, update to true after grid search is implemented
        alpha : contrastive constant, can be set or found through grid search
        num_components : number of components to retain, will update to weights in the future, can be set or found through grid search
        data_path: path to dataframe.parquet, initialised as df, must be preprocessed (centered and standardized)
        """

        self.alpha = alpha

        if num_components <= 0:
            logger.error("Non-positive num_components passed")
            raise ValueError(f"{num_components} must be positive")
        self.num_components = num_components

        self.df = pd.read_parquet(data_path)
        
        if self.df.columns.str.startswith("Meta").sum() == 0:
            logger.error("No columns start with Meta")
            raise ValueError("Dataframe must contain metadata columns")
        self.feat_len = self.df[self.df.columns[~self.df.columns.str.startswith("Meta")]].shape[1]

        if grid:
            self.alpha, self.num_components = grid_search()
    
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
        target_indices, control_indices = self.extract_controls(column, controls)

        # Compute covariances from indices
        target_cov, control_cov = self.compute_covariances(target_indices, control_indices)

        # Contrast between target_cov and control_cov, retain only num_components
        projection_basis = self.compute_contrastivePCA(target_cov, control_cov)

        # Project df onto basis
        df_projected = self.project_data(projection_basis)

        # Save to output path
        df_projected.to_parquet(output_path)
    
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

        return (target_indices, controls_indices)

    @log_method_call
    def compute_covariances(self, target_indices: pd.Index, control_indices: pd.Index) ->  Tuple[np.ndarray, np.ndarray]:
        """
        Method that computes and returns the covariance matrices for target and controls

        Parameters:
        ===========
        target_indices: Indices in dataframe that correspond to target, must have dimension > 1
        control_indices: Indices in dataframe that correspond to control, must have dimension > 1

        Returns:
        ========
        (np.ndarray, np.ndarray)
        The covariances of target and control
        """

        if min(len(target_indices), len(control_indices)) <= 1:
            logger.error("Indices not greater than 2")
            ValueError("There must be more than 2 indices")

       
        df_feat = self.df[self.df.columns[~self.df.columns.str.startswith("Meta")]]

        target_cov = (df_feat.loc[target_indices].values.T).dot(df_feat.loc[target_indices].values)/ (len(target_indices) - 1)
        control_cov = (df_feat.loc[control_indices].values.T).dot(df_feat.loc[control_indices].values)/ (len(control_indices) - 1)

        return(target_cov, control_cov)

    @log_method_call
    def compute_contrastivePCA(self, target_covariance: np.ndarray, control_covariance: np.ndarray) -> np.ndarray:
        """
        Method that contrasts target covariance matrix with controls convariance matrix

        Parameters:
        ===========
        target_covariance, control_covariance: numpy arrays of feat_len x feat_len

        Returns:
        ========
        The eigenvectors truncated to num_components
        """
        if np.shape(target_covariance)!=np.shape(control_covariance):
            logger.error("The covariance matrices are no compatible")
            ValueError("The covariance matrices are not compatible")

        w, v = LA.eig(target_covariance - self.alpha*control_covariance)
        eigen_indices = np.argpartition(w, -self.num_components)[-self.num_components:]
        eigen_indices = eigen_indices[np.argsort(-w[eigen_indices])]
        top_eigenvectors = v[:,eigen_indices]

        return top_eigenvectors

    @log_method_call
    def project_data(self, basis: np.ndarray) -> pd.DataFrame:
        """
        Projects data on to eigenvectors and renames features

        Parameters:
        ===========
        basis: The subspace to project data onto

        Returns:
        ========
        Pd.Dataframe
        Dataframe with new features projected in the direction of basis
        """

        projected_data = np.array(self.df[self.df.columns[~self.df.columns.str.startswith("Meta")]].dot(basis))
        features_PCA = [f"PCA_{i}" for i in range(projected_data.shape[1])]

        metadata = self.df[self.df.columns[self.df.columns.str.startswith("Meta")]]
        projected_df = pd.DataFrame(projected_data, columns=features_PCA)

        projected_df = pd.concat([projected_df, metadata], axis=1)

        return projected_df