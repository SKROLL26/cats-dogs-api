from pydantic import BaseModel
from pydantic import Field
from server.settings import app_settings

class Image(BaseModel):
    id_: str = Field(..., alias="ID", description="ID of the image")
    img_code: bytes = Field(..., description="Base64 encoded image")

class Prediction(BaseModel):
    id_: str = Field(..., alias="ID", description="ID of the image")
    cat_proba: float = Field(..., description="Probability of belonging to cat class")
    dog_proba: float = Field(..., description="Probability of belonging to dog class")

class RequestBody(BaseModel):
    photos: list[Image] = Field(..., max_items=app_settings.request_max_items)

class ResponseBody(BaseModel):
    results: list[Prediction]