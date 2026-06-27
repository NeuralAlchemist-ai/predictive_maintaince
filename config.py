from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Type, List

@dataclass
class PipelineConfig:
    data_path: str = "./datasets"
    dataset_name: str = "FD002"
    random_state: int = 42
    num_clusters: int = 6
    url: str = 'http://localhost:8080'

    cluster_model_class: Type = KMeans
    scaler_class: Type = StandardScaler
    sensor_cols: List[str] = field(default_factory=lambda: [f'sensor{i}' for i in range(1, 22)])
