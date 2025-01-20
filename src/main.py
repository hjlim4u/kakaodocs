import asyncio
import json
from typing import Dict, List
from utils.chat_analyzer import ChatAnalyzer
from pprint import pprint

async def analyze_chats(chat_files: List[str]) -> Dict[str, Dict]:
    analyzer = ChatAnalyzer()
    return await analyzer.analyze_chats(chat_files)

def print_analysis_results(analysis: dict):
    """분석 결과를 구조화하여 출력"""
    print("\n=== 기본 통계 ===")
    print(f"총 메시지 수: {analysis['basic_stats']['message_count']}")
    print(f"참여자 수: {analysis['basic_stats']['participant_count']}")
    print(f"평균 메시지 길이: {analysis['basic_stats']['avg_message_length']:.1f}자")
    print(f"메시지 길이 표준편차: {analysis['basic_stats']['message_length_std']:.1f}")
    print("\n참여자별 메시지 수:")
    for user, count in analysis['basic_stats']['messages_per_participant'].items():
        print(f"- {user}: {count}개")

    # print("\n=== 대화 패턴 ===")
    # if 'daily_patterns' in analysis['patterns']:
    #     daily = analysis['patterns']['daily_patterns']
    #     print(f"가장 활발한 시간대: {daily.get('peak_hour', 'N/A')}시")
    #     print(f"가장 활발한 요일: {daily.get('peak_weekday', 'N/A')}")

    print("\n=== 대화 구간 분석 ===")
    for segment in analysis['segment_analyses']:
        print(f"\n시간 구간: {segment['period']['start']} ~ {segment['period']['end']}")
        if 'response_patterns' in segment['analysis']:
            resp = segment['analysis']['response_patterns']
            print(f"중앙값 응답 시간: {resp['median_response_time']:.1f}초")
            print("\n응답 시간 패턴:")
            for pair, stats in resp['response_time_by_pair'].items():
                print(f"- {pair}: {stats['median']:.1f}초 (응답 {stats['count']}회)")

    if 'sentiment_analysis' in analysis:
        print("\n=== 감정 분석 ===")
        if 'user_sentiments' in analysis['sentiment_analysis']:
            print("\n사용자별 감정 분석:")
            for user, sentiments in analysis['sentiment_analysis']['user_sentiments'].items():
                print(f"\n{user}:")
                print(f"- 긍정: {sentiments['positive']:.2%}")
                print(f"- 부정: {sentiments['negative']:.2%}")
                print(f"- 중립: {sentiments['neutral']:.2%}")
                print(f"- 감정 변화도: {sentiments['sentiment_std']:.3f}")
                print(f"- 메시지 수: {sentiments['message_count']}개")

        if 'sentiment_flow' in analysis['sentiment_analysis']:
            print("\n대화 감정 흐름 (최근 10개 구간):")
            recent_flow = analysis['sentiment_analysis']['sentiment_flow'][-10:]
            for flow in recent_flow:
                print(f"\n시간대: {flow['start_time']} ~ {flow['end_time']}")
                print(f"- 긍정: {flow['avg_sentiment']['positive']:.2%}")
                print(f"- 부정: {flow['avg_sentiment']['negative']:.2%}")
                print(f"- 중립: {flow['avg_sentiment']['neutral']:.2%}")

    print("\n=== 대화 역학 분석 ===")
    dynamics = analysis.get('conversation_dynamics', {})
    
    # 대화 구간별 활동 패턴
    for segment in analysis['segment_analyses']:
        if 'conversation_dynamics' in segment['analysis']:
            seg_dynamics = segment['analysis']['conversation_dynamics']
            print(f"\n[구간: {segment['period']['start']} ~ {segment['period']['end']}]")
            
            # 활동 패턴
            if 'activity_pattern' in seg_dynamics:
                print("\n시간대별 활동:")
                for minute, count in seg_dynamics['activity_pattern'].items():
                    print(f"- {minute}분: {count}회")
            
            # 참여 불균형도
            if 'participation_inequality' in seg_dynamics:
                print(f"\n참여 불균형도: {seg_dynamics['participation_inequality']:.3f}")
            
            # 중심성 분석
            if 'flow_centrality' in seg_dynamics:
                print("\n중심성 분석:")
                for metric, values in seg_dynamics['flow_centrality'].items():
                    if values:  # 값이 있는 경우에만 출력
                        print(f"\n{metric} 중심성:")
                        for user, score in values.items():
                            print(f"- {user}: {score:.3f}")

    file_name = analysis['file_name'].split('.')[0]
    with open(f"{file_name}_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n전체 분석 결과가 {file_name}_analysis.json 파일로 저장되었습니다.")

async def main():
    chat_files = [
        'KakaoTalkChats_android.txt',
        'Talk_2025.1.15 19_40-1_ios.txt'
    ]
    
    results = await analyze_chats(chat_files)
    
    for file_path, analysis in results.items():
        print(f"\n{'='*20} {file_path} 분석 결과 {'='*20}")
        print_analysis_results(analysis)
        print("\n" + "="*70)

if __name__ == "__main__":
    asyncio.run(main()) 