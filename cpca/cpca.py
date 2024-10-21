from typing import List, Optional
import sys
sys.path.append("..")
import logging
from logging_config import setup_logging
logger = logging.getLogger(__name__)

class ContrastivePCA:
    """
    Version 1.0.0
    
    """

    def __init__(self, alpha: float, num_components: int):
        """
        Constructs necessary attributes (hyperparameters) for ContrastivePCA

        Parameters:
        alpha : contrastive constant, can be set or found through grid search
        num_components : number of components to retain, will update to weights in the future, can be set or found through grid search
        """

        self.alpha = alpha
        if num_components <= 0:
            logger.error("Non-positive num_components passed")
            raise ValueError(f"{num_components} must be positive")
        self.num_components = num_components
    
    
