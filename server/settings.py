from pydantic import BaseSettings

class AppSettings(BaseSettings):
    state_dict_path: str


app_settings = AppSettings()