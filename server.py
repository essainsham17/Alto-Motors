from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import triage

api = FastAPI()


class Inquiry(BaseModel):
    message: str


@api.post("/api/triage")
def run_triage(payload: Inquiry):
    return triage(payload.message)   # returns the full state dict as JSON


@api.get("/")
def index():
    return FileResponse("static/index.html")


api.mount("/static", StaticFiles(directory="static"), name="static")