from enum import Enum


class TaskType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    RANKING = "ranking"


class LossType(str, Enum):
    BCE = "bce"
    FOCAL = "focal"
    CROSS_ENTROPY = "ce"
    MSE = "mse"
    HUBER = "huber"
    LISTNET = "listnet"
