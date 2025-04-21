from fastapi import APIRouter

router = APIRouter()

@router.get("/api/fastapi/hello")
def say_hello():
    return {"message": "Hello from FastAPI!"}
