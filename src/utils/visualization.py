import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

class ChatVisualizer:
    def create_timeline(self, df) -> go.Figure:
        # 시간대별 메시지 분포
        df['hour'] = df['datetime'].dt.hour
        hourly_counts = df.groupby('hour').size()
        
        fig = go.Figure(data=[
            go.Bar(x=hourly_counts.index, 
                  y=hourly_counts.values,
                  name='Messages per Hour')
        ])
        
        fig.update_layout(
            title='Message Distribution by Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Number of Messages'
        )
        
        return fig
    
    def create_interaction_heatmap(self, df) -> go.Figure:
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
            
        return px.imshow(interaction_matrix,
                        title='User Interaction Heatmap') 