import numpy as np
import pandas as pd
from typing import Union, List

def calculate_outlier_threshold(
    values: Union[np.ndarray, List[float]], 
    iqr_multiplier: float = 1.5,
    min_threshold: float = None,
    remove_lower: bool = True
) -> tuple:
    """IQR 방식으로 이상치 임계값 계산"""
    # NumPy 배열을 pandas Series로 변환
    values_series = pd.Series(values)
    
    q1 = values_series.quantile(0.25)
    q3 = values_series.quantile(0.75)
    iqr = q3 - q1
    
    # remove_lower가 False인 경우 하한값은 데이터의 최소값으로 설정
    lower_bound = q1 - (iqr * iqr_multiplier) if remove_lower else values_series.min()
    upper_bound = q3 + (iqr * iqr_multiplier)
    
    # 최소 임계값 적용
    if min_threshold is not None and remove_lower:
        lower_bound = max(lower_bound, min_threshold)
    
    return lower_bound, upper_bound

def adjust_outliers(
    values: Union[np.ndarray, List[float]], 
    iqr_multiplier: float = 1.5,
    remove_lower: bool = True
) -> np.ndarray:
    """이상치를 임계값으로 조정"""
    lower_bound, upper_bound = calculate_outlier_threshold(
        values, 
        iqr_multiplier, 
        remove_lower=remove_lower
    )
    adjusted_values = np.clip(values, lower_bound, upper_bound)
    return adjusted_values