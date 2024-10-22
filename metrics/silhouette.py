import pandas as pd
from sklearn.metrics import silhouette_score
from logging_config import log_method_call

def silhouette_label(data_path: str, column: str, metric: str="cosine", filter_DMSO: bool=True) -> float:
    """
    Calculates the normalised Silhouette label score using the silhouette function

    Parameters:
    ===========
    data_path: data_path: Path to dataframe.parquet
    column: Label column for cluster assingment, typically compound or genes
    metric: Metric used to calculate Silhouette score
    filter_DMSO: Filter DMSO controls, default is True

    Returns:
    ========
    float
        Normalised Silhoutte label score

    """
    df = pd.read_parquet(data_path)
    if filter_DMSO:
        df = df[df.columns[column]!="DMSO"]
    
    return silhouette(df, column, metric=metric)



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
    float
        Silhouette score (Renormalised between 0 and 1)

    """

    score = silhouette_score(df[df.columns[~df.columns.str.startswith("Meta")]], df[column], metric=metric)
    score = (score + 1)/2
    return score

