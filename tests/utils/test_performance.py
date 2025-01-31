import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd

@pytest.mark.asyncio
async def test_large_dataset_performance(analyzer, base_sample_df):
    """대용량 데이터 처리 성능 테스트"""
    # 대량의 샘플 데이터 생성
    large_df = pd.concat([base_sample_df] * 1000, ignore_index=True)
    
    start_time = datetime.now()
    result = await analyzer.analyze_interactions(large_df, 'test_chat')
    end_time = datetime.now()
    
    processing_time = (end_time - start_time).total_seconds()
    assert processing_time < 30  # 30초 이내 처리 검증

@pytest.mark.asyncio
async def test_concurrent_analysis(analyzer, base_sample_df):
    """동시 분석 처리 테스트"""
    tasks = []
    for _ in range(5):
        task = asyncio.create_task(
            analyzer.analyze_interactions(base_sample_df, f'test_chat_{_}')
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    assert len(results) == 5
    assert all(isinstance(r, dict) for r in results) 