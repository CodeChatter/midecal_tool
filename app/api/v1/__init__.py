from fastapi import APIRouter
from .endpoints.mask import router as mask_router
from .endpoints.detect import router as detect_router
from .endpoints.ocr import router as ocr_router

v1_router = APIRouter(prefix="/api")
v1_router.include_router(mask_router)
v1_router.include_router(detect_router)
v1_router.include_router(ocr_router)
