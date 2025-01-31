class InvalidInputError(Exception):
    """잘못된 입력 데이터 예외"""
    pass

class AnalysisError(Exception):
    """분석 중 발생한 예외"""
    pass

class ResourceNotFoundError(Exception):
    """리소스를 찾을 수 없는 예외"""
    pass 