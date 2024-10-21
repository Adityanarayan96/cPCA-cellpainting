from typing import List, Optional, Tuple
import sys
import functools
sys.path.append("..")
import logging
from logging_config import setup_logging
import pandas as pd
logger = logging.getLogger(__name__)

class ContrastivePCA:
    """
    Version 1.0.0
    
    """

    def __init__(self, alpha: float, num_components: int):
        """
        Constructs necessary attributes (hyperparameters) for ContrastivePCA

        Parameters:
        ===========
        alpha : contrastive constant, can be set or found through grid search
        num_components : number of components to retain, will update to weights in the future, can be set or found through grid search
        """

        self.alpha = alpha
        if num_components <= 0:
            logger.error("Non-positive num_components passed")
            raise ValueError(f"{num_components} must be positive")
        self.num_components = num_components
    
    def extract_controls(self, data_path: str, column: str, controls: List[str]) -> Tuple[pd.Index, pd.Index]:
        """
        Extracts indices of controls, the non-control indices are targets

        Parameters:
        ==========
        data_path: Path to parquet file
        column: Metadata column in data to obtain indices corresponding to controls
        controls: List of metadata to be pooled to controls

        Returns:
        ========
        (pd.Index, pd.Index)
            (Control indices, Target indices)

        """

        df = pd.read_parquet(data_path)
        controls_index_list = [df[df[column] == control].index for control in controls]
        controls_indices = functools.reduce(lambda x, y: x.union(y), controls_index_list) # lambda is a throwaway function
        target_indices = df.drop(controls_indices).index

        return Tuple[controls_indices, target_indices]


    

