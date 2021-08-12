import torch
from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.exceptions import HTTPException
from torchvision.models import resnet18, mobilenet
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from PIL import Image

from server.models import ResponseBody
from server.settings import app_settings, ModelArch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

router = APIRouter(prefix="/api/v1/catsdogs")
if app_settings.model_arch is ModelArch.RESNET18:
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("models/resnet18.pt"))
elif app_settings.model_arch is ModelArch.MOBILENET_V3:
    model = mobilenet.mobilenet_v3_small()
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load("models/mobilenetv3.pt"))
model = model.to(DEVICE)
model.eval()

async def get_preds(files: list[UploadFile] = File(...)) -> tuple[torch.Tensor, list[str]]:
    files = list(filter(lambda file: file.content_type in ("image/jpeg", "image/png"), files))
    if len(files) > app_settings.request_max_items:
        raise HTTPException(status_code=422, detail=f"Max {app_settings.request_max_items} images allowed")
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    imgs = [Image.open(file.file) for file in files]
    imgs_transformed = [transform(img) for img in imgs]
    inputs = torch.stack(imgs_transformed).to(DEVICE)
    preds = model(inputs)
    if DEVICE.type.startswith("cuda"):
        torch.cuda.empty_cache()
    return preds.cpu(), [file.filename for file in files]

@router.post("/predictproba", response_model=ResponseBody)
def predict_proba_files(preds: tuple[torch.Tensor, list[str]] = Depends(get_preds)):
    preds, file_names = preds
    preds_probas = torch.softmax(preds, 1).tolist()
    return {
        "results": [{"ID": file_name, "cat_proba": cat_proba, "dog_proba": dog_proba} for file_name, (cat_proba, dog_proba) in zip(file_names, preds_probas)]
    }

@router.post("/predict")
def predict_files(preds: tuple[torch.Tensor, list[str]] = Depends(get_preds)):
    preds, file_names = preds
    preds_classes = map(lambda label: "cat" if label == 0 else "dog", torch.argmax(torch.softmax(preds, 1), 1).tolist())
    return {
        "results": [{"ID": file_name, "prediction": prediction} for file_name, prediction in zip(file_names, preds_classes)]
    }

