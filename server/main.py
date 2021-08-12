from fastapi import FastAPI
from server.routes import router
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="Cats and Dogs classification API",
    redoc_url=None,
    docs_url="/"
)

app.include_router(router)