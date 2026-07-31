from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import Inquiry_Desk

api = FastAPI()


class Inquiry(BaseModel):
    message: str


@api.post("/api/Inquiry_Desk")
def run_Inquiry_Desk(payload: Inquiry):
    return Inquiry_Desk(payload.message)   # returns the full state dict as JSON


@api.get("/")
def index():
    return FileResponse("static/index.html")


api.mount("/static", StaticFiles(directory="static"), name="static")