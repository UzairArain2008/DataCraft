from fastapi import APIRouter

router = APIRouter()

@router.get("/health")  # this is the exact path
def health():
    return {"status": "ok"}
