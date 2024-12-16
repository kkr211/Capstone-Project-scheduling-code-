import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict  # defaultdict 모듈 import

# 입력 데이터
DEMAND = {
    'A': 10,
    'B': 10
}

PROCESSING_TIME = {
    'LITHO': {'M_1': 5, 'M_2': 5, 'M_3': 8, 'M_4': 8},
    'ETCH': {'M_1': 7, 'M_2': 7, 'M_3': [3, 9], 'M_4': [3, 9]},  
    'DEPO': {'M_1': [4, 8], 'M_2': [4, 8], 'M_3': 6, 'M_4': 6}
}
# op = operation -> 작업 종류를 의미 
ALLOWED_MACHINES = {
    'LITHO': {'M_1': 'A', 'M_2': 'A', 'M_3': 'B', 'M_4': 'B'},
    'ETCH': {'M_1': 'A', 'M_2': 'A', 'M_3': ['A', 'B'], 'M_4': ['A', 'B']},
    'DEPO': {'M_1': ['A', 'B'], 'M_2': ['A', 'B'], 'M_3': 'B', 'M_4': 'B'}
}

SETUP_TIME = 1

# 각 작업의 초기 수요
jobs = []
for job_type, count in DEMAND.items():
    for i in range(count):
        jobs.append((job_type, ['LITHO', 'ETCH', 'DEPO']))

# 각 작업의 최적 기계와 시간을 찾는 함수
def get_processing_time(op, machine, job):
    time = PROCESSING_TIME[op][machine]
    if isinstance(time, list):
        if job == 'A':
            return time[0]
        else:
            return time[1]
    return time

# 작업을 각 기계에 배치 (LPT 적용)
operation_schedule = defaultdict(lambda: defaultdict(list))
machine_completion_time = defaultdict(lambda: defaultdict(int))
last_job_type_on_machine = defaultdict(lambda: defaultdict(str))
global_machine_completion_time = defaultdict(int)

for job, operations in jobs:
    for op in operations:
        # 각 기계에서 작업 시간을 기반으로 LPT 규칙에 따라 작업 정렬
        candidate_machines = ALLOWED_MACHINES[op]
        sorted_machines = sorted(
            candidate_machines.keys(),
            key=lambda m: get_processing_time(op, m, job),
            reverse=True  # 처리 시간이 긴 순서로 정렬
        )

        best_machine = None
        best_completion_time = float('inf')
        best_setup_time = 0
        best_start_time = 0

        for machine in sorted_machines:
            allowed_jobs = candidate_machines[machine]
            if isinstance(allowed_jobs, list) and job not in allowed_jobs:
                continue
            elif isinstance(allowed_jobs, str) and job != allowed_jobs:
                continue

            setup_time = 0
            if last_job_type_on_machine[op][machine] != job and last_job_type_on_machine[op][machine] != '':
                setup_time = SETUP_TIME

            start_time = max(machine_completion_time[op][machine], global_machine_completion_time[machine])
            processing_time = get_processing_time(op, machine, job)
            completion_time = start_time + setup_time + processing_time

            if completion_time < best_completion_time:
                best_completion_time = completion_time
                best_machine = machine
                best_setup_time = setup_time
                best_start_time = start_time

        # Update schedules
        operation_schedule[op][best_machine].append((best_start_time, best_start_time + best_setup_time + get_processing_time(op, best_machine, job), job))
        machine_completion_time[op][best_machine] = best_completion_time
        global_machine_completion_time[best_machine] = best_completion_time
        last_job_type_on_machine[op][best_machine] = job

# 결과 출력
for op in ['LITHO', 'ETCH', 'DEPO']:
    print(f"\nOperation {op}:")
    for machine, schedule in operation_schedule[op].items():
        print(f"  Machine {machine}:")
        for start_time, end_time, job in schedule:
            print(f"    Job {job}: Start at {start_time}, End at {end_time}")

# makespan 계산
makespan = max(max(times.values()) for times in machine_completion_time.values())
print(f"\nMakespan: {makespan}")

# 간트차트 생성
fig, ax = plt.subplots(figsize=(12, 8))

colors = {
    'A': 'tab:blue',
    'B': 'tab:orange'
}

yticks = []
yticklabels = []
y_base = 0
bar_height = 0.8

for op in ['LITHO', 'ETCH', 'DEPO']:
    machines = operation_schedule[op]
    for machine in ['M_1', 'M_2', 'M_3', 'M_4']:
        yticks.append(y_base + bar_height / 2)
        yticklabels.append(f"{op} {machine}")
        if machine in machines:
            tasks = machines[machine]
            for start_time, end_time, job in tasks:
                ax.broken_barh([(start_time, end_time - start_time)], (y_base, bar_height), facecolors=colors[job])
                ax.text(start_time + (end_time - start_time) / 2, y_base + bar_height / 2, job, color='white', ha='center', va='center', fontsize=8)
        y_base += bar_height + 0.2

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_xlabel("Time")
ax.set_ylabel("Operations and Machines")
ax.grid(True)

# 범례 추가
patches = [mpatches.Patch(color=color, label=f"Job {job}") for job, color in colors.items()]
ax.legend(handles=patches)

plt.title("Gantt Chart for Job Scheduling with LPT Rule")
plt.show()