# routers/logs.py
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from routers.auth import require_api_key

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("/recent")
async def get_recent_logs(count: int = 50, level: Optional[str] = None,
                          category: Optional[str] = None):
    """Get recent log entries with optional filters."""
    from logging_system import app_logger
    logs = app_logger.get_recent_logs(count=count, level=level, category=category)
    return {
        "count": len(logs),
        "logs": logs,
    }


@router.get("/summary")
async def get_log_summary():
    """Get logging system summary."""
    from logging_system import app_logger
    return app_logger.get_summary()


@router.get("/counts")
async def get_log_counts():
    """Get event counts."""
    from logging_system import app_logger
    return app_logger.get_counts()


@router.post("/export", dependencies=[Depends(require_api_key)])
async def export_logs():
    """Export all logs to JSON file."""
    from logging_system import app_logger
    filepath = app_logger.export_logs()
    return {
        "message": "Logs exported successfully",
        "filepath": filepath,
    }


@router.get("/download")
async def download_logs():
    """Download the main log file."""
    import os
    log_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "logs", "rescueway.log"
    )
    if os.path.exists(log_file):
        return FileResponse(
            log_file,
            media_type="text/plain",
            filename="rescueway.log"
        )
    raise HTTPException(status_code=404, detail="Log file not found")


@router.get("/emergencies")
async def get_emergency_logs(count: int = 20):
    """Get only emergency-related logs."""
    from logging_system import app_logger
    return {
        "logs": app_logger.get_recent_logs(count=count, category="emergency"),
    }


@router.get("/errors")
async def get_error_logs(count: int = 20):
    """Get only error logs."""
    from logging_system import app_logger
    return {
        "logs": app_logger.get_recent_logs(count=count, level="ERROR"),
    }