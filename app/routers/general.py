from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, JSONResponse, Response
from typing import List
from app.services import data_service
from app.core.config import settings
import os


router = APIRouter(tags=["General"])

@router.get("/")
async def root():
    file_path = os.path.join(settings.STATUS_CODES_DIR, "404.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return Response("Welcome to OptiTrade API", status_code=status.HTTP_200_OK)


@router.get("/news", response_model=List[dict])
async def get_news_endpoint(page: int = 1, page_size: int = 10):
    return data_service.get_news_data(page, page_size)


@router.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all(path_name: str, request: Request):
    api_prefixes = ["/auth", "/users", "/stocks", "/watchlist", "/portfolio", "/orders", "/transactions", "/news", "/openapi.json", "/docs", "/redoc"]
    if any(request.url.path.startswith(prefix) for prefix in api_prefixes) and request.url.path != f"/{path_name}":
         pass

    status_map = {
        404: "404.html",
    }
    
    error_file = status_map.get(404)
    file_path = os.path.join(settings.STATUS_CODES_DIR, error_file)

    if os.path.exists(file_path):
        return FileResponse(file_path, status_code=404)
    return JSONResponse(content={"detail": "Not Found"}, status_code=status.HTTP_404_NOT_FOUND)