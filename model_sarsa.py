import random
import math
import matplotlib.pyplot as plt

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
N_EPISODES = 50000
ALPHA = 0.01
GAMMA = 0.8
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.1

USE_COUNT_STATE = True
STATE_INTERVAL = 3
USE_JOB_TYPE_ACTION = True
# 스케줄링 규칙
RULE_LIST = ['SPT', 'LPT' , 'MOR', 'LOR', 'MWR', 'SPTSSU'] # 디스패칭 룰 추가 

class JobScheduler:
    def __init__(self, action_type='job_type', rule_list=None, state_type='count',state_interval=1):
        self.action_type = action_type
        self.rule_list = rule_list if rule_list else RULE_LIST
        self.state_type = state_type
        self.state_interval = state_interval
        self.processing_time = PROCESSING_TIME
        self.reset()
    # __init__() : action_type , rule_list , state_type 설정 후 reset() 함수 실행 
    def reset(self):
        # 기존 reset 코드 유지
        self.demand = DEMAND.copy() # DEMAND로부터 복사
        self.total_jobs = sum(DEMAND.values()) # 총 processing time 계산 
        self.remaining_jobs = DEMAND.copy() # DEMAND로부터 복사 
        
        self.machine_status = {
            'OP_1': {f'M_{i}': {'last_job': None, 'completion_time': 0} for i in range(1, 5)},
            'OP_2': {f'M_{i}': {'last_job': None, 'completion_time': 0} for i in range(1, 5)},
            'OP_3': {f'M_{i}': {'last_job': None, 'completion_time': 0} for i in range(1, 5)}
        } # 각각의 오퍼레이션과 머신에서 마지막 작업이 무엇이었는지와 완료시간을 저장하는 변수
        # machine 의 상태 표시 
        # 간트 차트를 위한 상세 스케줄 정보 저장
        self.machine_schedules = {
            'OP_1': {f'M_{i}': [] for i in range(1, 5)},
            'OP_2': {f'M_{i}': [] for i in range(1, 5)},
            'OP_3': {f'M_{i}': [] for i in range(1, 5)}
        }
        # machine 의 작업 표시 
        self.completed_op1 = []
        self.completed_op2 = []
        self.job_sequence = []
        self.job_sequence_info = []
        self.makespan = 0
        
        return self.get_state()
    # reset() : 클래스 전체를 초기화 
    def is_machine_available(self, operation, machine, job_type):
        allowed = ALLOWED_MACHINES[operation][machine]
        if isinstance(allowed, list): # allowed 가 단일값인지 list 인지 확인 (A 이거나 [A,B])
            return job_type in allowed # job_type가 allowed list에 있는지 확인하여 true or false 를 반환 
        return job_type == allowed # job_type가 allowed 에 있는지 확인하여 true or false 를 반환 
    # is_machine_available() : job_type이 allowed에 있는 값인지를 확인
    def calculate_start_time(self, operation, machine, job_type, job_id):
        # 기계의 이전 작업 완료시간
        machine_completion_time = self.machine_status[operation][machine]['completion_time']
        # operatioin = op_1 or op_2 , machnie = M_1, M_2 ... , 그 안의 딕셔너리는 = last_job , completion_time 
        if operation == 'OP_3' : # OP_3의 경우 OP_2D의 완료시간을 고려 
            op2_competion_time = 0
            for job in self.completed_op2:
                if job['job_type'] == job_type and job['job_id'] == job_id:
                    op2_competion_time = job['completion_time']
                    break
            return max(op2_competion_time, machine_completion_time)
        # OP_2의 경우 OP_1 완료시간 고려
        elif operation == 'OP_2':
            # 현재 job_type과 job_id에 해당하는 OP_1 완료 시간 찾기
            op1_completion_time = 0
            for job in self.completed_op1:
                if job['job_type'] == job_type and job['job_id'] == job_id: # type 과 type 그리고 id와 id가 맞았을 때
                    op1_completion_time = job['completion_time']
                    break # for 루프를 종료 

            # OP_1 완료시간과 기계의 이전 작업 완료시간 중 더 큰 값을 시작시간으로 반환
            return max(op1_completion_time, machine_completion_time)

        # OP_1의 경우 기계의 이전 작업 완료시간을 시작시간으로 반환
        return machine_completion_time
    # calculate_start_time() : 다음 작업(op2)이 시작될 시간을 계산 / op1의 경우 선행 작업이 없기 때문에 따로 시작 시간을 계산 x / op1의 완료 시간들을 이용하여 op2 의 시작시간을 계산 
    def step(self, action):
        if self.action_type == 'job_type': 
            scheduled = self.schedule_job_type(action)
        else:
            scheduled = self.schedule_by_rule(action)
        # action_type 이 job_type 인지 rule_type인지에 따라 scheduling 방식이 달라지고 scheduled 값이 달라짐
        if not scheduled:
            return self.get_state(), -1000, True
            
        # 모든 작업이 완료되었는지 확인
        done = sum(self.remaining_jobs.values()) == 0
        # 모든 작업이 완료되면 done = true 
        # Reward는 makespan의 음수값
        reward = -self.makespan if done else -0.1 # 모든 작업이 완료되는 데 걸리는 시간을 계산하여 reward 를 계산 
        
        return self.get_state(), reward, done 
    # step() : 한 step이 끝났을 경우의 state, reward, done을 반환 (스케줄링이 완료된 경우)
    def calculate_processing_time(self, operation, machine, job_type):
        processing_time = PROCESSING_TIME[operation][machine]
        
        # OP_2의 M_3, M_4에 대해서는 작업 타입별 생산시간 반환
        if operation == 'OP_2' and machine in ['M_3', 'M_4']:
            return processing_time[0] if job_type == 'A' else processing_time[1]
        elif operation == 'OP_3' and machine in ['M_1','M_2']:
            return processing_time[0] if job_type == 'A' else processing_time[1] # AM수정
        
        # 나머지 경우는 기계별 고정 생산시간 반환
        return processing_time
    # calculate_processing_time() : operation, machine, job_type에 따라 그에 따른 processing_time을 계산 
    def schedule_job_type(self, action):
        job_types = list(DEMAND.keys()) # list[A,B]
        job_type = job_types[action % len(job_types)] # action 에 따라 A , B 중 하나 설정 
        
        if self.remaining_jobs[job_type] <= 0:
            return False
        # 특정 타입에 남아 있는 작업이 없음면 (0이면) False return 
        job_id = DEMAND[job_type] - self.remaining_jobs[job_type] + 1 
        # job_id : 현재 작업의 index 
        # job_sequence_info에 작업 정보 추가
        self.job_sequence_info.append({
            'job_type': job_type,
            'job_id': job_id
        }) # 현재 작업중인 job_type 과 job_id 의 정보를 추가해서 저장 

        # OP_1에서 가능한 가장 빠른 시작 시간을 가진 기계 선택
        best_start_time_op1 = float('inf') # 초기 아무 기계가 할당 되지 않았을 때는 무한대로 설정 (infinite)
        best_machine_op1 = None
        job_id = DEMAND[job_type] - self.remaining_jobs[job_type] + 1
        best_setup_time_op1 = 0

        for machine in ALLOWED_MACHINES['OP_1'].keys():
            if self.is_machine_available('OP_1', machine, job_type): # job_type이 allowed 에 있는지 없는지 확인 
                # 시작 가능 시간 계산
                start_time = self.calculate_start_time('OP_1', machine, job_type, job_id) # 시작시간 계산 

                # Setup time 계산
                setup_time = SETUP_TIME if (self.machine_status['OP_1'][machine]['last_job'] is not None and
                                        self.machine_status['OP_1'][machine]['last_job'] != job_type) else 0
                # 마지막 작업이 없지 않고 마지막 작업이 현재 진행하려는 작업과 다르다면 setuptime = 1 , 그게 아니면 setuptime = 0 
                processing_time = self.calculate_processing_time('OP_1', machine, job_type) # processing_time 계산 
                # Setup time을 포함한 완료시간 계산
                completion_time = start_time + setup_time + processing_time

                if start_time < best_start_time_op1:
                    best_start_time_op1 = start_time
                    best_machine_op1 = machine
                    best_setup_time_op1 = setup_time
                    best_completion_time_op1 = completion_time
                    best_processing_time_op1 = processing_time
                # 계산된 start_time 이 best_start_time_op1 보다 작으면 갱신됨 아니면 유지
        if best_machine_op1 is None:
            return False

        # OP_1 작업 기록
        self.completed_op1.append({
            'job_type': job_type,
            'job_id': job_id,
            'completion_time': best_completion_time_op1
        })
        # completed_op1 에 작업 할 job_type , job_id , completion_time 추가 
        self.machine_schedules['OP_1'][best_machine_op1].append({
            'job_type': job_type,
            'job_id': job_id,
            'start_time': best_start_time_op1,
            'completion_time': best_completion_time_op1,
            'setup_time': best_setup_time_op1,
            'processing_time': best_processing_time_op1
        })
        # machine_Schedules 에 start_time에 따라 머신에 할당 된 작업들을 순서대로 기록 
        self.machine_status['OP_1'][best_machine_op1]['last_job'] = job_type
        self.machine_status['OP_1'][best_machine_op1]['completion_time'] = best_completion_time_op1

        # OP_2 스케줄링
        best_start_time_op2 = float('inf')
        best_machine_op2 = None
        best_setup_time_op2 = 0

        for machine in ALLOWED_MACHINES['OP_2'].keys():
            if self.is_machine_available('OP_2', machine, job_type):
                # 시작 가능 시간 계산
                start_time = self.calculate_start_time('OP_2', machine, job_type, job_id)

                # Setup time 계산
                setup_time = SETUP_TIME if (self.machine_status['OP_2'][machine]['last_job'] is not None and
                                        self.machine_status['OP_2'][machine]['last_job'] != job_type) else 0

                processing_time = self.calculate_processing_time('OP_2', machine, job_type)
                # Setup time을 포함한 완료시간 계산
                completion_time = start_time + setup_time + processing_time

                if start_time < best_start_time_op2:
                    best_start_time_op2 = start_time
                    best_machine_op2 = machine
                    best_setup_time_op2 = setup_time
                    best_completion_time_op2 = completion_time
                    best_processing_time_op2 = processing_time

        if best_machine_op2 is None:
            return False
        
        # OP_2 작업 기록
        self.completed_op2.append({
            'job_type': job_type,
            'job_id': job_id,
            'completion_time': best_completion_time_op2
        })

        # OP_2 작업 기록
        self.machine_schedules['OP_2'][best_machine_op2].append({
            'job_type': job_type,
            'job_id': job_id,
            'start_time': best_start_time_op2,
            'completion_time': best_completion_time_op2,
            'setup_time': best_setup_time_op2,
            'processing_time': best_processing_time_op2
        })

        self.machine_status['OP_2'][best_machine_op2]['last_job'] = job_type
        self.machine_status['OP_2'][best_machine_op2]['completion_time'] = best_completion_time_op2
        # OP_3 스케줄링
        best_start_time_op3 = float('inf')
        best_machine_op3 = None
        best_setup_time_op3 = 0

        for machine in ALLOWED_MACHINES['OP_3'].keys():
            if self.is_machine_available('OP_3', machine, job_type):
                # 시작 가능 시간 계산
                start_time = self.calculate_start_time('OP_3', machine, job_type, job_id)

                # Setup time 계산
                setup_time = SETUP_TIME if (self.machine_status['OP_3'][machine]['last_job'] is not None and
                                        self.machine_status['OP_3'][machine]['last_job'] != job_type) else 0

                processing_time = self.calculate_processing_time('OP_3', machine, job_type)
                # Setup time을 포함한 완료시간 계산
                completion_time = start_time + setup_time + processing_time

                if start_time < best_start_time_op3:
                    best_start_time_op3 = start_time
                    best_machine_op3 = machine
                    best_setup_time_op3 = setup_time
                    best_completion_time_op3 = completion_time
                    best_processing_time_op3 = processing_time

        if best_machine_op3 is None:
            return False

        # OP_3 작업 기록
        self.machine_schedules['OP_3'][best_machine_op3].append({
            'job_type': job_type,
            'job_id': job_id,
            'start_time': best_start_time_op3,
            'completion_time': best_completion_time_op3,
            'setup_time': best_setup_time_op3,
            'processing_time': best_processing_time_op3
        })

        self.machine_status['OP_3'][best_machine_op3]['last_job'] = job_type
        self.machine_status['OP_3'][best_machine_op3]['completion_time'] = best_completion_time_op3
        
        self.makespan = max(self.makespan, best_completion_time_op3) # makespan 계산 
        self.job_sequence.append(job_type) # 작업 순서 
        self.remaining_jobs[job_type] -= 1 # 남은 작업 계산 

        return True
    # schedule_job_type : job_type에 따른 스케줄링
    def schedule_by_rule(self, action):
        rule = self.rule_list[action % len(self.rule_list)] # rule을 설정 
        
        available_jobs = []
        for job_type in DEMAND.keys():
            if self.remaining_jobs[job_type] > 0:
                job_id = DEMAND[job_type] - self.remaining_jobs[job_type] + 1
                # 각 작업의 총 처리시간 계산 (가능한 최소 시간 기준)
                min_op1_machine, min_op1_time = min(((m, self.calculate_processing_time('OP_1', m, job_type)) 
                                                     for m in ALLOWED_MACHINES['OP_1'].keys() 
                                                     if self.is_machine_available('OP_1', m, job_type)),key=lambda x: x[1])
                setup_time_op1 = SETUP_TIME if (self.machine_status['OP_1'][min_op1_machine]['last_job'] is not None and
                                            self.machine_status['OP_1'][min_op1_machine]['last_job'] != job_type) else 0
                min_op1_time_plus = setup_time_op1 + min_op1_time
                min_op2_machine, min_op2_time = min(((m, self.calculate_processing_time('OP_2', m, job_type)) 
                                                     for m in ALLOWED_MACHINES['OP_2'].keys() 
                                                     if self.is_machine_available('OP_2', m, job_type)),key=lambda x: x[1]) 
                setup_time_op2 = SETUP_TIME if (self.machine_status['OP_2'][min_op2_machine]['last_job'] is not None and
                                            self.machine_status['OP_2'][min_op2_machine]['last_job'] != job_type) else 0
                min_op2_time_plus = setup_time_op2 + min_op2_time
                min_op3_machine, min_op3_time = min(((m, self.calculate_processing_time('OP_3', m, job_type)) 
                                                     for m in ALLOWED_MACHINES['OP_3'].keys() 
                                                     if self.is_machine_available('OP_3', m, job_type)),key=lambda x: x[1])
                setup_time_op3 = SETUP_TIME if (self.machine_status['OP_3'][min_op3_machine]['last_job'] is not None and
                                            self.machine_status['OP_3'][min_op3_machine]['last_job'] != job_type) else 0
                min_op3_time_plus = setup_time_op3 + min_op3_time
                # op , machine , job_type에 따라 가장 적은 processing time을 계산 함
                total_time = min_op1_time + min_op2_time + min_op3_time # ex)  job_id = 3 번의 , job_type = a의 op1과 op2의 총 처리시간 = total_time 
                total_time_plus = min_op1_time_plus + min_op2_time_plus + min_op3_time_plus
                available_jobs.append((job_type, job_id, total_time,total_time_plus))
                
        if not available_jobs:
            return False
            
        # 규칙에 따라 작업 선택 (디스패칭 룰 추가 sst, ssu, mor, mwr, sptssu 중 3개 추가 ) 
        if rule == 'SPT':
            selected_job = min(available_jobs, key=lambda x: x[2]) # availabled_job에서 total_time이 가장 작은 job을 할당 
        elif rule == 'LPT':
            selected_job = max(available_jobs, key=lambda x: x[2])
        elif rule == 'MOR':
            temp = 0 
            remaining_job_type = ""
            for job_type in self.remaining_jobs.keys() :
                if self.remaining_jobs[job_type] > temp :
                    temp = self.remaining_jobs[job_type]
                    remaining_job_type = job_type
                else : 
                    temp = temp
                    remaining_job_type = remaining_job_type
            selected_job = remaining_job_type
        elif rule == 'LOR':        
            temp = float('inf')
            remaining_job_type = ""
            for job_type in self.remaining_jobs.keys() :
                if self.remaining_jobs[job_type] < temp :
                    temp = self.remaining_jobs[job_type]
                    remaining_job_type = job_type
                else : 
                    temp = temp
                    remaining_job_type = remaining_job_type
            selected_job = remaining_job_type
        else : # SPTSSU
            selected_job = min(available_jobs, key=lambda x: x[3])
        job_type = selected_job[0]
        return self.schedule_job_type(list(DEMAND.keys()).index(job_type))
    # schedule_by_rule : rule에 따른 작업 선택 
    def get_state(self):
        if self.state_type == 'count':
            state = []

            # 1. 각 작업 타입별 남은 수량
            for job_type in DEMAND.keys():
                state.append(self.remaining_jobs[job_type])

            # 2. OP_1의 각 기계별 현재 작업 정보
            for machine in ALLOWED_MACHINES['OP_1'].keys():
                # 현재 작업 타입을 one-hot 인코딩 (None=0, A=1, B=2)
                current_job = self.machine_status['OP_1'][machine]['last_job']
                if current_job is None:
                    state.extend([0, 0])  # [0, 0] = no job
                elif current_job == 'A':
                    state.extend([1, 0])  # [1, 0] = job A
                else:  # job B
                    state.extend([0, 1])  # [0, 1] = job B

                # 해당 기계의 현재 완료시간을 전체 makespan으로 정규화
                completion_time = self.machine_status['OP_1'][machine]['completion_time']
                state.append(completion_time / (completion_time + 1))  # 0으로 나누는 것 방지

            # 3. OP_2의 각 기계별 현재 작업 정보 (OP_1과 동일한 방식)
            for machine in ALLOWED_MACHINES['OP_2'].keys():
                current_job = self.machine_status['OP_2'][machine]['last_job']
                if current_job is None:
                    state.extend([0, 0])
                elif current_job == 'A':
                    state.extend([1, 0])
                else:  # job B
                    state.extend([0, 1])

                completion_time = self.machine_status['OP_2'][machine]['completion_time']
                state.append(completion_time / (completion_time + 1))

            # 4. OP_3의 각 기계별 현재 작업 정보 (OP_1과 동일한 방식)
            for machine in ALLOWED_MACHINES['OP_3'].keys():
                # 현재 작업 타입을 one-hot 인코딩 (None=0, A=1, B=2)
                current_job = self.machine_status['OP_3'][machine]['last_job']
                if current_job is None:
                    state.extend([0, 0])  # [0, 0] = no job
                elif current_job == 'A':
                    state.extend([1, 0])  # [1, 0] = job A
                else:  # job B
                    state.extend([0, 1])  # [0, 1] = job B

                # 해당 기계의 현재 완료시간을 전체 makespan으로 정규화
                completion_time = self.machine_status['OP_3'][machine]['completion_time']
                state.append(completion_time / (completion_time + 1))  # 0으로 나누는 것 방지

            # 5. OP_1 완료 후 OP_2 대기 중인 작업 수
            op1_completed = len([job for job in self.completed_op1  # OP1에서 완료된 A 타입의 개수를 구함
                               if job['job_type'] == 'A']) 
            op2_assigned_A = len([job for machine in self.machine_schedules['OP_2'].values() # OP2에서 할당된 A 타입의 개수를 구함 
                                for job in machine if job['job_type'] == 'A'])
            waiting_A1 = op1_completed - op2_assigned_A

            op1_completed = len([job for job in self.completed_op1 
                               if job['job_type'] == 'B'])
            op2_assigned_B = len([job for machine in self.machine_schedules['OP_2'].values()
                                for job in machine if job['job_type'] == 'B'])
            waiting_B1 = op1_completed - op2_assigned_B

            state.extend([waiting_A1, waiting_B1])

            op2_completed_A = len([job for job in self.completed_op2 if job['job_type'] == 'A'])
            op3_assigned_A = len([job for machine in self.machine_schedules['OP_3'].values()
                              for job in machine if job['job_type'] == 'A'])
            waiting_A2 = op2_completed_A - op3_assigned_A

            op2_completed_B = len([job for job in self.completed_op2 if job['job_type'] == 'B'])
            op3_assigned_B = len([job for machine in self.machine_schedules['OP_3'].values()
                              for job in machine if job['job_type'] == 'B'])
            waiting_B2 = op2_completed_B - op3_assigned_B

            state.extend([waiting_A2, waiting_B2])

            return state
        else:  # 'progress'
            # 전체 진행률
            completed = self.total_jobs - sum(self.remaining_jobs.values())
            return [completed / self.total_jobs]
    # cout : 작업 상태, 기계 상태, 남은 작업 수량 등의 정보를 표현 / prorgress : 전체 작업 진행률을 표현 
    def get_state_size(self):
        if self.state_type == 'count':
            return 3 + (4 * 3) + (4 * 3) + (4 * 3) + 3
        else:  # 'progress'
            return 1

    def get_valid_actions(self):
        if self.action_type == 'job_type':
            return [i for i, job_type in enumerate(DEMAND.keys()) if self.remaining_jobs[job_type] > 0] # 남아있는 작업의 인덱스를 반환 A OR B
        else: # action_type = rule 이라면 
            return list(range(len(self.rule_list))) # [spt, lpt]
        
class SARSAAgent: # sarsa 알고리즘 
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

    def get_action(self, state):
        state = tuple(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        if state not in self.q_table:
            self.q_table[state] = [0] * self.action_size

        return self.q_table[state].index(max(self.q_table[state]))
    # 랜덤하게 아무 state를 return 하거나 q_table 중 가장 높은 q 값을 가진 state를 return
    def update_q_table(self, state, action, reward, next_state, next_action, done):
        state = tuple(state)
        next_state = tuple(next_state)
        if state not in self.q_table:
            self.q_table[state] = [0] * self.action_size

        if next_state not in self.q_table:
            self.q_table[next_state] = [0] * self.action_size

        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action] if not done else 0
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[state][action] = new_q

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

#EPSILON = 0.8
#EPSILON_DECAY = 0.9998
#EPSILON_MIN = 0.05
def train(n_episodes=N_EPISODES, state_type='count', state_interval=STATE_INTERVAL, action_type='job_type', rule_list=None):
    env = JobScheduler(action_type=action_type, rule_list=rule_list, state_type=state_type, state_interval=state_interval) 
    action_size = len(env.rule_list) if action_type == 'rule' else len(env.processing_time) # action_type = rule 이면 action_size = rule 개수, job_typed이면 processing_time 개수
    state_size = env.get_state_size()
    agent = SARSAAgent(state_size=state_size, action_size=action_size)

    for episode in range(n_episodes): # 에피소드 수 만큼 에피소드 루프 탐험
        state = env.reset() # reset된 current_state를 return 받음 
        action = agent.get_action(state) # sarsa 알고리즘으로 선택된 행동을 반환 받음
        done = False
        total_reward = 0 

        while not done:
            next_state, reward, done = env.step(action) # action에 따라 한 사이클 돌아서 state, reard, done 을 할당받음 
            next_action = agent.get_action(next_state) # next_state 에 따라 sarsa 알고리즘으로 다음 action 을 할당받음 
            agent.update_q_table(state, action, reward, next_state, next_action, done) # q_table 업데이트 
            state = next_state # state 업데이트 
            action = next_action # action 업데이트 
            total_reward += reward # reward 업데이트 
            makespan = -total_reward
        # 모든 사이클을 돌며 total reward 계산 (한 에피소드에 모든 job_list가 배치된 후의 total reward가 계산됨)
        if episode % (N_EPISODES//10) == 0: # 일정 에피소드마다 아래의 문장 출력 
            print(f"Episode: {episode}, Job Sequence: {env.job_sequence}, makespan: {makespan}, Epsilon: {agent.epsilon:.2f}")

    return env, agent
# 하나의 에피소드는 job_list를 모두 배치시키며 total reward를 계산하고, 이를  에피소드 파라미터 만큼 반복 -> sarsa 알고리즘으로 계속해서 순서를 업데이트 하면서 최적에 가까운 env, agent를 반환
def test(env, agent):
    state = env.reset()
    done = False
    action_sequence = []
    job_sequence = []

    while not done:
        action = agent.get_action(state)
        action_sequence.append(action)
        next_state, reward, done = env.step(action)
        job_sequence.append(env.job_sequence[-1] if env.job_sequence else None)
        state = next_state
    # 한 사이클의 action_sequence와 job_sequence를 저장 
    print("Final Job Sequence:", job_sequence)
    if env.action_type == 'rule': 
        rule_sequence = [env.rule_list[action % len(env.rule_list)] for action in action_sequence]
        print("Rule Sequence:", rule_sequence)
    else:
        action_sequence = [job for job in job_sequence if job]
        print("Action Sequence:", action_sequence)
    print("makespan:", env.makespan)
    # action_type 에 따라 job_sequence를 출력하고, 최종 total tardiness를 출력 
    gantt = GanttChart()
    gantt.visualize_schedule(env)
    
class GanttChart:
    def __init__(self):
        self.colors = {
            'A': '#FF9999',
            'B': '#66B2FF',
            'setup': '#000000'
        }
        
    def visualize_schedule(self, env):
        job_counter = 1
        job_numbers = {}

        for job_info in env.job_sequence_info:
            key = (job_info['job_type'], job_info['job_id'])
            if key not in job_numbers:
                job_numbers[key] = job_counter
                job_counter += 1

        op1_schedule = []
        op2_schedule = []
        op3_schedule = []

        for machine in ALLOWED_MACHINES['OP_1'].keys():
            machine_schedule = env.machine_schedules['OP_1'][machine]
            for job in machine_schedule:
                job_num = job_numbers[(job['job_type'], job['job_id'])]
                op1_schedule.append({
                    'Machine': machine,
                    'Job': job['job_type'],
                    'JobNum': f"Job {job_num}",
                    'Start': job['start_time'],
                    'End': job['completion_time'],
                    'Setup': job['setup_time'] > 0
                })

        for machine in ALLOWED_MACHINES['OP_2'].keys():
            machine_schedule = env.machine_schedules['OP_2'][machine]
            for job in machine_schedule:
                job_num = job_numbers[(job['job_type'], job['job_id'])]
                op2_schedule.append({
                    'Machine': machine,
                    'Job': job['job_type'],
                    'JobNum': f"Job {job_num}",
                    'Start': job['start_time'],
                    'End': job['completion_time'],
                    'Setup': job['setup_time'] > 0
                })

        for machine in ALLOWED_MACHINES['OP_3'].keys():
            machine_schedule = env.machine_schedules['OP_3'][machine]
            for job in machine_schedule:
                job_num = job_numbers[(job['job_type'], job['job_id'])]
                op3_schedule.append({
                    'Machine': machine,
                    'Job': job['job_type'],
                    'JobNum': f"Job {job_num}",
                    'Start': job['start_time'],
                    'End': job['completion_time'],
                    'Setup': job['setup_time'] > 0
                })

        max_time = env.makespan

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))  # 전체 크기 조정

        self._draw_operation_chart(axes[0], op1_schedule, "LITHOGRAPHY", max_time)
        self._draw_operation_chart(axes[1], op2_schedule, "ETCH", max_time)
        self._draw_operation_chart(axes[2], op3_schedule, "DEPOSITION", max_time)

        state_type = 'Count State' if env.state_type == 'count' else 'Progress State'
        action_type = 'Job Type Action' if env.action_type == 'job_type' else 'Rule-based Action'
        plt.suptitle(f'Gantt Chart ({state_type} / {action_type})  (Makespan: {max_time})', y=0.97, fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85, hspace=0.5)
        plt.show()

    def _draw_operation_chart(self, ax, schedule, title, max_time):
        machines = sorted(list(set(job['Machine'] for job in schedule)))

        ax.set_ylim(-0.5, len(machines) - 0.5)
        ax.set_yticks(range(len(machines)))
        ax.set_yticklabels(machines)

        ax.set_xlim(-1, max_time + 3)
        time_interval = max(1, max_time // 10)
        ax.set_xticks(range(0, int(max_time) + time_interval, time_interval))

        ax.set_title(title, pad=5)  # 타이틀과 차트 간격 조정
        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')

        ax.grid(True, axis='x', alpha=0.3)

        for job in schedule:
            y = machines.index(job['Machine'])

            if job['Setup']:
                setup_width = SETUP_TIME
                setup_rect = ax.barh(y, setup_width, left=job['Start'], 
                                   color=self.colors['setup'],
                                   alpha=0.9,
                                   edgecolor='none')

            width = job['End'] - job['Start']
            if job['Setup']:
                width -= SETUP_TIME
                start = job['Start'] + SETUP_TIME
            else:
                start = job['Start']

            job_rect = ax.barh(y, width, left=start,
                              color=self.colors[job['Job']], 
                              edgecolor='black',
                              linewidth=1)

            ax.text(start + width/2, y, job['JobNum'], 
                   ha='center', va='center',
                   fontsize=10)

        ax.axvline(x=max_time, color='red', linestyle='--', alpha=0.5)

        if title == "OP_1":
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['A'], 
                             label='Type A', edgecolor='black'),
                plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['B'],
                             label='Type B', edgecolor='black'),
                plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['setup'],
                             label='Setup Time', edgecolor='none')
            ]
            ax.legend(handles=legend_elements, 
                     loc='upper right', 
                     bbox_to_anchor=(1.13, 1))
            ax.text(max_time + 0.2, len(machines) - 1, 
                  f'Makespan: {max_time}', 
                  color='red', va='top', ha='left')

if __name__ == "__main__": # 코드가 직접 실행되었을 때만 아래 코드를 실행 / 이 코드에서는 별 다른 기능 없이 아래 구문을 실행  
    # 'count' 상태 정의로 Job Type 액션으로 학습
    print("\nTraining with job type actions and 'count' state definition:")
    env, agent = train(state_type='count', action_type='job_type' if USE_JOB_TYPE_ACTION else 'rule')
    test(env, agent)

    # 'count' 상태 정의로 Rule 기반 액션으로 학습
    print("\nTraining with rule-based actions and 'count' state definition:")
    env, agent = train(state_type='count', action_type='rule', rule_list=RULE_LIST)
    test(env, agent)

    # 'progress' 상태 정의로 Job Type 액션 학습 (3개씩 묶음)
    print("\nTraining with job type actions and 'progress' state definition (interval 3):")
    env, agent = train(state_type='progress', state_interval=STATE_INTERVAL, action_type='job_type' if USE_JOB_TYPE_ACTION else 'rule')
    test(env, agent)

    # 'progress' 상태 정의로 Rule 기반 액션 학습 (3개씩 묶음)
    print("\nTraining with rule-based actions and 'progress' state definition (interval 3):")
    env, agent = train(state_type='progress', state_interval=STATE_INTERVAL, action_type='rule', rule_list=RULE_LIST)
    test(env, agent)
