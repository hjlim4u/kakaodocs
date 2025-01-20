import pytest
import pandas as pd
from datetime import datetime
from src.utils.visualization import ChatVisualizer

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