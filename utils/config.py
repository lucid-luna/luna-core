# ====================================================================
#  File: utils/config.py
# ====================================================================
"""
Configuration loader for L.U.N.A. Core.

`config/` 폴더의 YAML 파일을 로드하여 L.U.N.A. Core의 설정을 관리합니다.
"""

import yaml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    name: str
    num_labels: int
    
@dataclass
class DataConfig:
    raw_dir: str
    processed_dir: str
    train_split: str
    validation_split: str
    test_split: str
    
@dataclass
class TrainConfig:
    output_dir: str
    epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    eval_strategy: str
    save_strategy: str
    best_metric: str

@dataclass
class InferenceConfig:
    threshold: Optional[float] = None
    label_list: List[str] = None
    
@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    inference: InferenceConfig
    max_length: int
    
def load_config(name: str) -> Config:
    """
    YAML 파일에서 설정을 로드하여 Config 객체를 반환합니다.
    
    Args:
        config_path (str): YAML 파일의 경로 (예: "config/models.yaml")
        
    Returns:
        Config: 설정 정보를 담고 있는 Config 객체
    """
    path = f"config/{name}.yaml"
    with open(path, 'r', encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    return Config(
        model=ModelConfig(**config_data['model']),
        data=DataConfig(**config_data['data']),
        train=TrainConfig(
            output_dir=config_data['train']['output_dir'],
            epochs=config_data['train']['epochs'],
            train_batch_size=config_data['train']['train_batch_size'],
            eval_batch_size=config_data['train']['eval_batch_size'],
            learning_rate=float(config_data['train']['learning_rate']),
            eval_strategy=config_data['train']['eval_strategy'],
            save_strategy=config_data['train']['save_strategy'],
            best_metric=config_data['train']['best_metric']
        ),
        inference=InferenceConfig(**config_data['inference']),
        max_length=config_data['max_length']
    )

def load_config_dict(name: str) -> dict:
    path = f"config/{name}.yaml"
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_multitask_config_dict(name: str) -> dict:
    path = f"config/{name}.yaml"
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
