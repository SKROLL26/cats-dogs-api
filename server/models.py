from pydantic import BaseModel
from pydantic import Field
class ModelInput(BaseModel):
    id_:str = Field(alias="ID", description="ID of the image")
    img_code: str = Field(description="Base64 encoded image")

class ModelOutput(BaseModel):
    id_:str = Field(alias="ID", description="ID of the image")
    cat_prob:float = Field(description="Probability of belonging to cat class")
    dog_prob:float = Field(description="Probability of belonging to dog class")

class RequestBody(BaseModel):
    photos: list[ModelInput]

class ResponseBody(BaseModel):
    results: list[ModelOutput]