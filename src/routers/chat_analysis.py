from fastapi import APIRouter, Query, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import aiofiles
import os
from pathlib import Path

from ..models.chat import ChatAnalysisResponse, ErrorResponse
from ..utils.chat_analyzer import ChatAnalyzer
from ..utils.auth import get_current_user

router = APIRouter(prefix="/api/chat", tags=["chat"])

# 임시 파일 저장 경로
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

async def cleanup_temp_file(file_path: Path):
    """임시 파일 정리"""
    try:
        if file_path.exists():
            os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up temp file: {e}")

@router.post("/analyze", 
    response_model=ChatAnalysisResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def analyze_chat(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="채팅 내보내기 텍스트 파일", media_type="text/plain"),
    chat_id: str = Query(None, description="채팅방 ID"),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """채팅 파일을 분석하여 결과를 반환합니다."""
    
    # 임시 파일 저장
    temp_file_path = UPLOAD_DIR / f"temp_{file.filename}"
    try:
        async with aiofiles.open(temp_file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # 채팅 분석 수행
        analyzer = ChatAnalyzer()
        result = await analyzer.analyze_single_chat(
            str(temp_file_path),
            user_id=current_user.get('id') if current_user else None,
            is_authenticated=bool(current_user),
            chat_id=chat_id
        )
        
        # 임시 파일 정리 태스크 등록
        background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        return result

    except Exception as e:
        # 에러 발생 시 임시 파일 정리
        if temp_file_path.exists():
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        ) 