from pydantic import BaseSettings
from enum import Enum

class ModelArch(str, Enum):
    MOBILENET_V3 = "mobilenetv3"
    RESNET18 = "resnet18"

class AppSettings(BaseSettings):
    model_arch: ModelArch = ModelArch.RESNET18
    request_max_items: int = 10


app_settings = AppSettings()