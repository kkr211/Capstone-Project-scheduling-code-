import torch
import torch.nn.functional as F
import torch.optim as optim
from config import *
from scheduler import FlowShopScheduler
from models import Qnet, ReplayBuffer
from ganttchart import GanttChart

def train_dqn(n_episodes=N_EPISODES, state_type='count', action_type='job_type', rule_list=None):
    env = FlowShopScheduler(action_type=action_type, rule_list=rule_list, state_type=state_type)
    action_size = len(env.rule_list) if action_type == 'rule' else len(DEMAND) # action_size : rule_type 일 때는 rule_list 만큼 size를 갖고, job_type일 시에는 job의 수만큼 size를 가짐
    state_size = env.get_state_size()

    q = Qnet(state_size, action_size)
    q_target = Qnet(state_size, action_size)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    optimizer = optim.Adam(q.parameters(), lr=LEARNING_RATE)
    # optim.Adam : 확률적 경사 하강법을 기반으로 Adam 옵티마이저를 구현 / 효율적인 q-value 함수 학습에 사용 
    for episode in range(n_episodes):
        epsilon = max(0.01, 0.08 - 0.01*(episode/200))
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            valid_actions = env.get_valid_actions()
            action = q.sample_action(torch.FloatTensor(state), epsilon, valid_actions) # state=obs, 즉 state 에 따라 신경망을 거쳐 출력값이 정해지고, 
            #epsilon에 따라 random으로 다음 행동 정해지거나, 가장 큰 출력값을 산출하는 행동이 다음 행동으로 선택됨.
            next_state, reward, done = env.step(action) # 선택된 행동에 따라, 다음 상태, 보상, done이 결정
            done_mask = 0.0 if done else 1.0 # done(모든 작업이 할당됨)이면 done_mask=0 아니면 done_mask=1
            memory.put((state, action, reward/100.0, next_state, done_mask)) # memory=buffer 에 해당 값들이 저장됨
            state = next_state # state 업데이트 
            total_reward += reward # reward 업데이트 

            if memory.size() > 2000:
                s, a, r, s_prime, done_mask = memory.sample(BATCH_SIZE) # sample 함수를 통해 memory의 값 중 일부를 뽑아서 batch를 만듬
                q_out = q(s) # 현재상태에 대한 q 값을 구함 
                q_a = q_out.gather(1, a) # 행동 a 에 대한 q 값을 구함 

                valid_actions = [env.get_valid_actions() for _ in range(BATCH_SIZE)] # batch 안에 있는 값들의 valid action을 리스트로 저장 
                max_q_prime = torch.zeros(BATCH_SIZE, 1) # zeors() :  모든 값이 0으로 채워진 tensor를 크기만큼 생성 
                q_prime = q_target(s_prime)
                for i, actions in enumerate(valid_actions):
                    if actions:
                        max_q_prime[i] = q_prime[i, actions].max().unsqueeze(0)

                target = r + GAMMA * max_q_prime * done_mask
                loss = F.smooth_l1_loss(q_a, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 100 == 0 and episode != 0:
            q_target.load_state_dict(q.state_dict())
            print(f"Episode: {episode}, Job Sequence: {env.job_sequence}, Makespan: {env.makespan}, Epsilon: {epsilon:.2f}")

    return env, q

def test_dqn(env, q):
    state = env.reset()
    done = False
    job_sequence = []

    while not done:
        valid_actions = env.get_valid_actions()
        action = q.sample_action(torch.FloatTensor(state), 0, valid_actions)
        next_state, reward, done = env.step(action)
        job_sequence.append(env.job_sequence[-1] if env.job_sequence else None)
        state = next_state

    print("Final Job Sequence:", job_sequence)
    print("Final Makespan:", env.makespan)
    gantt = GanttChart()
    gantt.visualize_schedule(env)