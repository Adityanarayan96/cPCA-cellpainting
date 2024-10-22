import pandas as pd
from sklearn.metrics import silhouette_score
from logging_config import log_method_call

@log_method_call
def silhouette(df: pd.Dataframe, column: str, metric: str="cosine") -> float:
    """
    Calculates the silhouette score according to column indicated

    Parameters:
    ===========
    dataframe: The dataframe to calculate scores on, must be filtered if DMSO should not included
    column: The column with cluster assignments
    metric: Metric used to calculate Silhouette Score, default is cosine

    Returns:
    ========
    Silhouette score (Renormalised between 0 and 1)

    """

    score = silhouette_score(df[df.columns[~df.columns.str.startswith("Meta")]], df[column], metric=metric)
    score = (score + 1)/2
    return score

