# 입력 데이터
# 수정된 입력 데이터
DEMAND = {
    'A': 10,  # 작업량 균형 조정
    'B': 10
}

PROCESSING_TIME = {
    'OP_1': {'M_1': 5, 'M_2': 5, 'M_3': 8, 'M_4': 8},
    'OP_2': {'M_1': 7, 'M_2': 7, 'M_3': [3, 9], 'M_4': [3, 9]},  
    'OP_3': {'M_1': [4, 8], 'M_2': [4, 8], 'M_3': 6, 'M_4': 6}
}
# op = operation -> 작업 종류를 의미 
ALLOWED_MACHINES = {
    'OP_1': {'M_1': 'A', 'M_2': 'A', 'M_3': 'B', 'M_4': 'B'},
    'OP_2': {'M_1': 'A', 'M_2': 'A', 'M_3': ['A', 'B'], 'M_4': ['A', 'B']},
    'OP_3': {'M_1': ['A', 'B'], 'M_2': ['A', 'B'], 'M_3': 'B', 'M_4': 'B'}
}

SETUP_TIME = 1

# 학습 파라미터
N_EPISODES = 500
GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_LIMIT = 8000
LEARNING_RATE = 0.0005
# 스케줄링 규칙
RULE_LIST = ['SPT', 'LPT' , 'MOR', 'LOR', 'SPTSSU'] # 디스패칭 룰 추가 