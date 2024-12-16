from config import *

class FlowShopScheduler:
    def __init__(self, action_type='job_type', rule_list=None, state_type='count'):
        self.action_type = action_type
        self.rule_list = rule_list if rule_list else RULE_LIST
        self.state_type = state_type
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