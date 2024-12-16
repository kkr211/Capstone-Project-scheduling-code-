from train import train_dqn, test_dqn
from config import *

if __name__ == "__main__":
    # 'count' 상태 정의로 Job Type 액션으로 학습
    print("Training with job type actions and 'count' state definition:")
    env, q = train_dqn(state_type='count', action_type='job_type')
    test_dqn(env, q)

    # 'count' 상태 정의로 Rule 기반 액션으로 학습
    print("\nTraining with rule-based actions and 'count' state definition:")
    env, q = train_dqn(state_type='count', action_type='rule', rule_list=RULE_LIST)
    test_dqn(env, q)

    # 'progress' 상태 정의로 Job Type 액션 학습
    print("\nTraining with job type actions and 'progress' state definition:")
    env, q = train_dqn(state_type='progress', action_type='job_type')
    test_dqn(env, q)

    # 'progress' 상태 정의로 Rule 기반 액션 학습
    print("\nTraining with rule-based actions and 'progress' state definition:")
    env, q = train_dqn(state_type='progress', action_type='rule', rule_list=RULE_LIST)
    test_dqn(env, q)