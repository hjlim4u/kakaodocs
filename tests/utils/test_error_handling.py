import pytest
from src.utils.exceptions import (
    InvalidInputError,
    AnalysisError,
    ResourceNotFoundError
)

@pytest.mark.asyncio
async def test_invalid_data_format(analyzer):
    """잘못된 데이터 형식 처리 테스트"""
    invalid_data = {'invalid': 'format'}
    with pytest.raises(InvalidInputError):
        await analyzer.analyze_interactions(invalid_data, 'test_chat')

@pytest.mark.asyncio
async def test_resource_not_found(analyzer, base_sample_df):
    """리소스 없음 에러 처리 테스트"""
    with pytest.raises(ResourceNotFoundError):
        await analyzer.analyze_interactions(
            base_sample_df,
            'nonexistent_chat',
            {'missing_resource': True}
        )

@pytest.mark.asyncio
async def test_analysis_error_handling(analyzer, base_sample_df):
    """분석 중 발생하는 에러 처리 테스트"""
    corrupted_df = base_sample_df.copy()
    corrupted_df.loc[0, 'datetime'] = None
    
    with pytest.raises(AnalysisError):
        await analyzer.analyze_interactions(corrupted_df, 'test_chat') 