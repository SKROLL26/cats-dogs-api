from fastapi import FastAPI
from server.routes import router

app = FastAPI(
    title="Cats and Dogs classification API",
    redoc_url=None
)

app.include_router(router)