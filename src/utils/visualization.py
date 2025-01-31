import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
from datetime import datetime, timedelta
import asyncio
from typing import Dict

class ChatVisualizer:
    async def create_timeline(self, df: pd.DataFrame) -> Dict:
        """시간별 메시지 분포 시각화"""
        return await asyncio.to_thread(self._create_timeline_sync, df)
    
    def _create_timeline_sync(self, df: pd.DataFrame) -> Dict:
        # 시간대별 메시지 분포
        df['hour'] = df['datetime'].dt.hour
        hourly_counts = df.groupby('hour').size()
        
        # fig = go.Figure(data=[
        #     go.Bar(x=hourly_counts.index, 
        #           y=hourly_counts.values,
        #           name='Messages per Hour')
        # ])
        
        # fig.update_layout(
        #     title='Message Distribution by Hour',
        #     xaxis_title='Hour of Day',
        #     yaxis_title='Number of Messages'
        # )
        
        return {}
    
    async def create_interaction_heatmap(self, df: pd.DataFrame) -> Dict:
        """상호작용 히트맵 생성"""
        return await asyncio.to_thread(self._create_interaction_heatmap_sync, df)
    
    def _create_interaction_heatmap_sync(self, df: pd.DataFrame) -> Dict:
        # 사용자간 상호작용 히트맵
        users = df['sender'].unique()
        interaction_matrix = pd.DataFrame(0, 
                                        index=users, 
                                        columns=users)
        
        prev_sender = None
        for sender in df['sender']:
            if prev_sender and prev_sender != sender:
                interaction_matrix.loc[prev_sender, sender] += 1
                interaction_matrix.loc[sender, prev_sender] += 1
            prev_sender = sender
            
        return {}
    