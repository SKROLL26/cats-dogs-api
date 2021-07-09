from fastapi import APIRouter
from server.models import RequestBody, ResponseBody

router = APIRouter(prefix="/api/v1/catsdogs")

@router.post("/predict", response_model=ResponseBody)
def predict(request_body: RequestBody):
    return None