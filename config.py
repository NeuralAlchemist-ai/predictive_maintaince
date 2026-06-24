from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Type

@dataclass
class PipelineConfig:
    data_path: str = "./data"
    dataset_name: str = "FD002"
    random_state: int = 42
    num_clusters: int = 6
    url: str = 'http://localhost:8080'

    cluster_model_class: Type = KMeans
    scaler_class: Type = StandardScaler
