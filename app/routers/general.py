import os
from typing import List

import starlette.responses
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from app.core.config import settings
from app.services import data_service

router = APIRouter(tags=["General"])


@router.get("/")
async def root() -> starlette.responses.FileResponse:
    file_path = os.path.join(settings.STATUS_CODES_DIR, "404.html")
    return FileResponse(file_path)


@router.get("/news", response_model=List[dict])
async def get_news_endpoint(page: int = 1, page_size: int = 10):
    return data_service.get_news_data(page, page_size)


@router.api_route(
    "/{path_name:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def catch_all(path_name: str, request: Request):
    api_prefixes = [
        "/users",
        "/stocks",
        "/watchlist",
        "/portfolio",
        "/orders",
        "/transactions",
        "/news",
    ]
    if (
        any(request.url.path.startswith(prefix) for prefix in api_prefixes)
        and request.url.path != f"/{path_name}"
    ):
        pass

    status_map = {
        404: "404.html",
    }

    error_file = status_map.get(404)
    file_path = os.path.join(settings.STATUS_CODES_DIR, error_file)

    return FileResponse(file_path, status_code=404)
