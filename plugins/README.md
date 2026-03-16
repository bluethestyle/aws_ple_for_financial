# Plugins

커스텀 컴포넌트를 여기에 추가합니다. 각 플러그인은 레지스트리에 자동 등록됩니다.

## 커스텀 태스크 추가

```python
# plugins/tasks/my_task.py
from core.task.base import AbstractTask, TaskConfig
from core.task.registry import TaskRegistry
import torch, torch.nn.functional as F

@TaskRegistry.register("my_custom_task")
class MyCustomTask(AbstractTask):
    def compute_loss(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())

    def predict(self, logits):
        return torch.sigmoid(logits)
```

## 커스텀 피처 트랜스포머 추가

```python
# plugins/features/my_transformer.py
from core.feature.base import AbstractFeatureTransformer
from core.feature.registry import FeatureRegistry
import numpy as np

@FeatureRegistry.register("log_scale")
class LogScaler(AbstractFeatureTransformer):
    def __init__(self, cols: list[str]):
        self.cols = cols

    def fit(self, df):
        return self

    def transform(self, df):
        df = df.copy()
        df[self.cols] = np.log1p(df[self.cols].clip(lower=0))
        return df
```

## 커스텀 데이터 소스 추가

```python
# plugins/data_sources/my_source.py
# S3 외에 다른 소스(RDS, DynamoDB, Kafka 등)에서 데이터를 가져올 때 사용
```
