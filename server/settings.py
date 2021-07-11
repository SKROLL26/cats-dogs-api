from pydantic import BaseSettings

class AppSettings(BaseSettings):
    state_dict_path: str = "resnet18_cat_dog.pt"
    request_max_items: int = 10


app_settings = AppSettings()