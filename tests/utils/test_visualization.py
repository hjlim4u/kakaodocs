import pytest
import pandas as pd
from datetime import datetime
from src.utils.visualization import ChatVisualizer
import time
import os

@pytest.fixture
def sample_df():
    data = {
        'datetime': [
            datetime(2024, 1, 1, 12, 0),
            datetime(2024, 1, 1, 13, 0),
            datetime(2024, 1, 1, 14, 0)
        ],
        'sender': ['User1', 'User2', 'User1'],
        'message': ['Hello', 'Hi', 'How are you?']
    }
    return pd.DataFrame(data)

@pytest.fixture
def visualizer():
    return ChatVisualizer()

def test_create_timeline(visualizer, sample_df):
    fig = visualizer.create_timeline(sample_df)
    assert fig is not None
    assert 'data' in fig
    assert len(fig.data) > 0

def test_create_interaction_heatmap(visualizer, sample_df):
    fig = visualizer.create_interaction_heatmap(sample_df)
    assert fig is not None
    assert 'data' in fig
    assert len(fig.data) > 0

def test_timeline_data_validation(visualizer, sample_df):
    """타임라인 데이터 유효성 검증"""
    fig = visualizer.create_timeline(sample_df)
    
    # x축이 시간 단위인지 확인
    assert all(isinstance(x, (int, float)) for x in fig.data[0].x)
    # y축이 양수인지 확인
    assert all(y >= 0 for y in fig.data[0].y)
    # 레이아웃 설정 확인
    assert fig.layout.title.text == 'Message Distribution by Hour'
    assert fig.layout.xaxis.title.text == 'Hour of Day'

def test_heatmap_data_validation(visualizer, sample_df):
    """히트맵 데이터 유효성 검증"""
    fig = visualizer.create_interaction_heatmap(sample_df)
    
    # 데이터 행렬이 대칭인지 확인
    data_matrix = fig.data[0].z
    assert all(data_matrix[i][j] == data_matrix[j][i] 
              for i in range(len(data_matrix)) 
              for j in range(len(data_matrix)))
    # 대각선이 0인지 확인
    assert all(data_matrix[i][i] == 0 
              for i in range(len(data_matrix)))

def test_visualization_performance(visualizer, base_sample_df):
    """시각화 성능 테스트"""
    large_df = pd.concat([base_sample_df] * 1000, ignore_index=True)
    
    start_time = time.time()
    fig = visualizer.create_timeline(large_df)
    end_time = time.time()
    
    assert (end_time - start_time) < 5  # 5초 이내 처리
    assert fig is not None

def test_visualization_export(visualizer, base_sample_df):
    """시각화 결과 내보내기 테스트"""
    fig = visualizer.create_timeline(base_sample_df)
    
    # HTML 내보내기 테스트
    html_path = 'test_timeline.html'
    visualizer.export_figure(fig, html_path)
    assert os.path.exists(html_path)
    os.remove(html_path)  # 테스트 후 정리 