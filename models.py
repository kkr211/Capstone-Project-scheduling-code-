import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from config import *
import random

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_LIMIT)
    # deque : 자료형 중 하나로 양방향 접근이 가능하다. 입력된 순서가 유지되며, maxlen을 통해 입력 가능한 크기를 설정 가능하다 오래된 자료들은 자동적으로 삭제된다. 
    # collections : 데크, 네임드튜플, 카운터 등 추가적인 자료형을 제공 
    # + import ~ as * (해당 모듈을 *라는 이름으로 가져옴) vs from ~ import * (해당 모듈 안의 *를 가져옴) 
    def put(self, transition):
        self.buffer.append(transition) 
    # put() : transition을 buffer에 추가 
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n) # buffer에 있는 값 중 n개 만큼 랜덤으로 추출한 후 mini_batch에 저장 
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask]) # mini_batch에 있는 값을 각각의 list(s,a,r,s_prime,done_mask)로 나눠서 저장 
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    # tensor ? 0에서부터 n차원까지의 데이터 구조 / 데이터가 tensor 구조로 저장되면 pytorch를 이용하여 자동미분을 통해 기울기를 구할 수 있다. 
    # torch.tensor() : 자료형을 tensor로 변환 
    # sample() : mini_bath에 있는 값을 s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst 로 나눈후 tensor 자료형으로 저장 
    def size(self):
        return len(self.buffer)
    # size() : buffer의 크기를 반환 (transition의 개수)
# ReplayBuffer() : transition 값을 buffer에 저장하고 buffer에 저장된 값 중 일부를 추출하여 tensor 형태로 반환
class Qnet(nn.Module):
    def __init__(self, state_size, action_size):
        super(Qnet, self).__init__() # + 여기서의 __init__ 함수는 nn.Module 에서 이미 정의된 함수이다.
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    # 신경망 구조 : state_size(입력층) -> fc1 -> fc2 -> fc3 -> action_size(출력층)
    # class 자식클래스(부모클래스) : -> 클래스를 상속하기 위한 방법 
    # nn.Module : pytorch에서 신경망 모델을 정의하는 모듈 
    # super() : 자식클래스에서 부모클래스의 메서드(함수)를 호출하는 함수 
    # Qnet() : 신경망 모델의 크기를 정의
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    # x 값이 신경망에 따라 값이 바뀌어서 반한됨 
    # x -> 선형 변환 -> ReLU (비선형 활성화 함수) ... -> X (정책 결정, Q 값 예측 등에 사용)
    # forward() : 신경망 모델의 동작을 정의 (x값이 신경망 모델을 거쳐 값이 바뀜)
    def sample_action(self, obs, epsilon, valid_actions):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.choice(valid_actions) # 0~1까지 랜덤한 값이 epsilon 보다 작을 경우 유효 액션 목록 중 하나를 랜덤으로 선택 
        else:
            return valid_actions[out[valid_actions].argmax().item()] # valid_actions 리스트가 인덱스가 되어 out중 타당한 액션을 하는 값들이 선별됨. 그 후, out 값 중 가장 큰 값을 가진 out의 인덱스가 선택됨
        # items() : tensor 자료형을 기본 자료형으로 변경 
        # argmax() : 값들 중 가장 큰 값의 인덱스를 반환 
    # sample_action() : 신경망에 따라 Q값을 반환받은 후 다음 동작을 결정 (epsilon-greedy)
# Qnet() : Q값을 산출하는 신경망 구조를 정의하고, 신경망에 따라 다음 동작을 결정하는 함수를 정의 