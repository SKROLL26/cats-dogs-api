from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.routes import router
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="Cats and Dogs classification API",
    redoc_url=None,
    docs_url="/"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
app.include_router(router)