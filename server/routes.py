from base64 import b64decode
from io import BytesIO
import binascii

import torch
from fastapi import APIRouter
from fastapi.exceptions import HTTPException
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from PIL import Image

from server.models import RequestBody, ResponseBody
from server.settings import app_settings

router = APIRouter(prefix="/api/v1/catsdogs")
model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(app_settings.state_dict_path))
model.eval()
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@router.post("/predict", response_model=ResponseBody)
def predict(request_body: RequestBody):
    img_codes = [x.img_code for x in request_body.photos]
    img_ids = [x.id_ for x in request_body.photos]
    try:
        imgs = [Image.open(BytesIO(b64decode(img_code, validate=True))) for img_code in img_codes]
    except binascii.Error:
        raise HTTPException(
            status_code=422,
            detail="Base64 string is not valid"
        )
    
    imgs_transformed = [transform(img) for img in imgs]
    inputs = torch.stack(imgs_transformed)
    preds = model(inputs)
    preds_probas = torch.softmax(preds, 1).tolist()

    return {"results": [{"ID": img_id, "cat_prob": cat_prob, "dog_prob": dog_prob} for img_id, (cat_prob, dog_prob) in zip(img_ids, preds_probas)]}
