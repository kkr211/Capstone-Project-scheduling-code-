import matplotlib.pyplot as plt
from config import *

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
        # 타이틀수정
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
