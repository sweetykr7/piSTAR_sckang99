# 필수 라이브러리와 모듈을 임포트합니다.
import pandas as pd  # 데이터 처리를 위한 라이브러리
import numpy as np  # 수치 계산을 위한 라이브러리
import torch  # PyTorch, 딥러닝 프레임워크
import torch.nn as nn  # PyTorch의 신경망 모듈
import torch.nn.functional as F  # PyTorch의 함수형 인터페이스
import torch.optim as optim  # 최적화 알고리즘 모듈
from torch.distributions import Categorical  # 확률 분포 관련 유틸리티
from tqdm import tqdm  # 진행률 표시 바
from collections import deque  # deque 컬렉션, 효율적인 데이터 삽입 및 삭제를 제공
import random  # 난수 생성
import enum  # 열거형 상수 지원

# 데이터 파일을 읽고, 결측치를 제거한 후 'Date' 열을 인덱스로 설정합니다.
df = pd.read_csv('SS00001.csv')  # 데이터 파일 읽기
df = df.dropna()  # 결측치 제거
df = df.set_index("Date")  # 'Date' 열을 인덱스로 설정


# 시그모이드 함수를 정의합니다.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 시그모이드 계산식


# 데이터 전처리: 'Close' 열에서 이전 값과의 차이를 구하고 시그모이드 함수를 적용합니다.
df['dClose'] = sigmoid(df['Close'] - df['Close'].shift(1))
df['MACD'] = sigmoid(df['MACD'])  # 'MACD' 열에 시그모이드 함수 적용
df['CCI'] = sigmoid(df['CCI'])  # 'CCI' 열에 시그모이드 함수 적용
df['RSI'] = np.log(df['RSI'])  # 'RSI' 열에 로그 함수 적용
df['ADX'] = np.log(df['ADX'])  # 'ADX' 열에 로그 함수 적용

# 각 기술적 지표를 선택하여 새로운 DataFrame 생성
indicator = df.loc[:, ['RSI', 'MACD', 'CCI', 'ADX']]
close = df['Close']  # 종가 데이터
dclose = df['dClose']  # 일일 종가 변화율
data = pd.concat([close, dclose, indicator], axis=1)  # 새로운 DataFrame 생성

# 훈련 데이터와 테스트 데이터로 분할
train_data = data.loc[(data.index > '1998-01-01') & (data.index <= '2019-11-31'), :]
test_data = data.loc[(data.index >= '2019-12-01') & (data.index <= '2021-05-31'), :]

# 하이퍼파라미터 설정
learning_rate = 1.0e-6  # 학습률
gamma = 0.98  # 감가율
buffer_limit = 1000  # 버퍼 크기
batch_size = 32  # 배치 크기
max_trade_share = 10  # 최대 거래 주식 수
tau = 0.001  # 소프트 업데이트 파라미터
action_space = 2 * max_trade_share + 1  # 행동 공간 크기 계산


# 타겟 신경망을 소스 신경망으로 부드럽게 업데이트하는 함수
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# 타겟 신경망을 소스 신경망의 파라미터로 완전히 대체하는 함수
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# 거래 시뮬레이션을 위한 클래스 정의
class Trade():
    def __init__(self, data, starting_balance=1000000, episodic_length=20, mode='train'):
        super(Trade, self).__init__()
        self.data = data  # 사용할 데이터
        self.price_column = 'Close'  # 가격 정보가 담긴 컬럼
        self.dClose_column = 'dClose'  # 일일 종가 변화율 컬럼
        self.indicator_columns = ['RSI', 'MACD', 'CCI', 'ADX']  # 기술적 지표 컬럼
        self.episodic_length = episodic_length  # 에피소드 길이
        self.starting_balance = starting_balance  # 시작 자본
        self.commission_rate = 0.0  # 수수료율
        self.cash = self.starting_balance  # 현재 현금
        self.shares = 0  # 보유 주식 수
        self.total_episodes = len(data)  # 전체 에피소드 수
        self.cur_step = 0  # 현재 스텝
        self.mode = mode  # 모드 ('train' 또는 'test')

    # 시뮬레이션을 초기화하고 첫 관찰을 반환합니다.
    def reset(self):
        self.cash = self.starting_balance  # 현금을 초기 자본으로 설정
        self.shares = 0  # 주식 수를 0으로 초기화
        if self.mode == 'train':
            self.cur_step = self.next_episode  # 훈련 모드에서는 무작위 에피소드로 시작
        else:
            self.cur_step = 0  # 테스트 모드에서는 처음부터 시작
        return self.next_observation()  # 첫 관찰 반환

    # 행동에 따라 매수 또는 매도를 수행하고, 다음 상태와 보상을 반환합니다.
    def step(self, action):
        balance = self.cur_balance  # 현재 잔액 저장
        self.cur_step += 1  # 스텝 수 증가
        if self.cur_step < self.total_steps - 1:
            self.take_action(action)  # 주어진 행동 수행
            state = self.next_observation()  # 다음 상태 관찰
            reward = self.cur_balance - balance  # 보상 계산: 잔액 변화량

        done = self.cur_step == self.total_steps - 2  # 종료 조건 검사
        return state, reward, done

    # 지정된 행동을 수행합니다 (매수 또는 매도).
    def take_action(self, action):
        action -= max_trade_share  # 행동에서 최대 거래 주식 수를 뺌
        if action > 0:  # 매수 행동인 경우
            share = action  # 매수할 주식 수
            price = self.cur_close_price * (1 + self.commission_rate)  # 구매 가격 계산
            if self.cash < price * share:  # 현금이 부족하면 구매 가능한 최대 주식 수 계산
                share = int(self.cash / price)
            self.cash -= price * share  # 현금 감소
            self.shares += share  # 주식 수 증가

        elif action < 0:  # 매도 행동인 경우
            share = -1 * action  # 매도할 주식 수
            price = self.cur_close_price * (1 - self.commission_rate)  # 판매 가격 계산
            if self.shares < share:  # 보유 주식보다 많이 매도하려면 보유 주식 전체를 매도
                share = self.shares
            self.cash += price * share  # 현금 증가
            self.shares -= share  # 주식 수 감소

    # 다음 관찰 상태를 생성합니다.
    def next_observation(self):
        obs = []
        obs = np.append(obs, [self.cur_dclose_price])  # 종가 변화율을 관찰 배열에 추가
        obs = np.append(obs, [np.log(self.cur_balance), np.log(self.shares + 1.0e-6),
                              np.log(self.cur_close_price)])  # 현재 잔액, 보유 주식 수, 현재 종가의 로그 값을 추가
        return np.append(obs, [self.cur_indicators])  # 기술적 지표 값 추가

    # 무작위 에피소드 시작 지점을 결정합니다.
    @property
    def next_episode(self):
        return random.randrange(0, self.total_episodes - batch_size)

    # 현재 스텝에서의 기술적 지표 값을 반환합니다.
    @property
    def cur_indicators(self):
        indicators = self.data[self.indicator_columns]
        return indicators.values[self.cur_step]

    # 현재 스텝에서의 일일 종가 변화율을 반환합니다.
    @property
    def cur_dclose_price(self):
        dclose = self.data[self.dClose_column].values
        d = self.cur_step - self.episodic_length
        if d >= 0:
            return dclose[d:self.cur_step]
        else:
            return np.append(np.array(-d * [dclose[0]]), dclose[0:self.cur_step])

    # 총 스텝 수를 반환합니다.
    @property
    def total_steps(self):
        return len(self.data)

    # 현재 스텝에서의 종가를 반환합니다.
    @property
    def cur_close_price(self):
        return self.data[self.price_column].iloc[self.cur_step]

    # 현재 잔액을 반환합니다.
    @property
    def cur_balance(self):
        return self.cash + (self.shares * self.cur_close_price)  # 현금과 현재 보유 주식의 가치 합


# 경험 재생 버퍼: 학습을 위해 경험(상태, 행동, 보상, 다음 상태, 종료 플래그)을 저장
class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen=buffer_limit)  # 설정된 버퍼 한도로 deque 생성

    # 경험을 버퍼에 저장합니다.
    def put(self, transition):
        self.buffer.append(transition)

    # 버퍼에서 무작위로 경험 배치를 샘플링합니다.
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(
            a_lst), \
            torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_lst, dtype=torch.float)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    # 버퍼의 크기를 반환합니다.
    def size(self):
        return len(self.buffer)


# DQN 신경망 모델 클래스
class Dqn(nn.Module):
    def __init__(self, state_size, action_size):
        super(Dqn, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)  # 첫 번째 선형 계층
        self.ln1 = nn.LayerNorm(512)  # 첫 번째 레이어 노말라이제이션
        self.fc2 = nn.Linear(512, 256)  # 두 번째 선형 계층
        self.ln2 = nn.LayerNorm(256)  # 두 번째 레이어 노말라이제이션
        self.fc3 = nn.Linear(256, action_size)  # 출력 계층

    # 순전파 함수
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))  # 첫 번째 계층을 거쳐 활성화 함수 적용
        x = F.relu(self.ln2(self.fc2(x)))  # 두 번째 계층을 거쳐 활성화 함수 적용
        value = self.fc3(x)  # 출력값 계산
        return value

    # 행동 선택 함수: 주어진 상태에서 행동을 샘플링
    def sample_action(self, state, epsilon):
        out = self.forward(state)  # 상태에 대한 예측
        coin = random.random()  # 무작위 수 생성
        if coin < epsilon:  # 탐험(랜덤 행동)
            return np.random.randint(0, action_space)
        else:  # 이용(모델 예측에 따른 행동)
            return out.argmax().item()


# 신경망을 훈련하는 함수
def train_net(q, memory, optimizer):
    total_loss = []
    q.train()

    # 버퍼에서 배치를 샘플링해 여러 번 학습
    for _ in range(4):
        s, a, r, s_prime, done = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a)

        # 타겟 네트워크에서 다음 상태에 대한 값 계산
        with torch.no_grad():
            qtarget_out = q(s_prime)
        bestaction = qtarget_out.argmax(1).unsqueeze(1)

        max_q_prime = qtarget_out.gather(1, bestaction)
        target = r + gamma * max_q_prime * done  # 타겟 계산
        loss = F.mse_loss(q_a, target)  # 손실 계산
        total_loss.append(loss.detach().numpy().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(total_loss)


# 전체 훈련 과정을 관리하는 함수
def train(window_size=20, starting_balance=100000, resume_epoch=0, max_epoch=1000):
    mode = 'train'
    env = Trade(train_data, starting_balance, window_size, mode)
    state_size = window_size + 7
    action_size = action_space
    q = Dqn(state_size, action_size)
    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    save_interval = 100
    epochs = max_epoch
    loss_history = []
    pv_history = []  # 포트폴리오 가치 추적
    start_epoch = resume_epoch

    torch.manual_seed(0)
    np.random.seed(0)

    if start_epoch > 0:
        q.load_state_dict(torch.load("Dqnmodel_ep" + str(start_epoch)))

    pbar = tqdm(range(start_epoch, epochs))

    for n_epi in pbar:
        if start_epoch == 0:
            epsilon = max(0.01, 1.0 - 1.0 * n_epi / epochs)
        else:
            epsilon = 0.01

        s = env.reset()
        done = False
        action_history = []

        for _ in range(2000):
            state = torch.from_numpy(s).float()
            a = q.sample_action(state, epsilon)
            s_prime, r, done = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime
            action_history.append(a)
            if done:
                break

        if memory.size() > batch_size:
            loss = train_net(q, memory, optimizer)

        loss_history.append(loss)

        np_actions = np.array(action_history)
        index_0 = len(np.where(np_actions == 10)[0])
        index_1 = len(np.where(np_actions > 10)[0])
        index_2 = len(np.where(np_actions < 10)[0])
        pv_history.append(env.cur_balance)
        pbar.set_description(str(index_0) + "/" + str(index_1) + "/" + str(index_2) + "/" + "%.4f" % env.cur_balance)

        if n_epi % save_interval == 0:
            torch.save(q.state_dict(), "Dqnmodel_ep" + str(n_epi))

    torch.save(q.state_dict(), "Dqnmodel_epfinal")

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(loss_history)
    axs[0].set_ylabel('policy_loss', fontsize=12)
    axs[0].set_xlabel("update", fontsize=12)

    axs[1].plot(pv_history)
    axs[1].set_ylabel('profit', fontsize=12)
    axs[1].set_xlabel("date", fontsize=12)

    plt.savefig('DQN_S00001_test.png')
    plt.show()
    plt.pause(3)
    plt.close()


# 테스트 함수 정의: 학습된 모델로 테스트 데이터에서의 성능을 평가합니다.
def test(window_size=20, starting_balance=100000, model_epi='final'):
    mode = 'test'
    env = Trade(test_data, starting_balance, window_size, mode)
    state_size = window_size + 7
    action_size = action_space
    q = Dqn(state_size, action_size)
    q.load_state_dict(torch.load("Dqnmodel_ep" + str(model_epi)))
    q.eval()

    action_history = []
    pv_history = []  # 포트폴리오 가치 기록

    s = env.reset()
    done = False

    while not done:
        state = torch.from_numpy(s).float()
        q_out = q(state)
        a = q_out.argmax().numpy().item()  #
        s_prime, r, done = env.step(a)
        s = s_prime
        action_history.append(a)
        pv = np.exp(s_prime[window_size])
        pv_history.append(pv)

    np_actions = np.array(action_history)
    test_close = test_data["Close"].values

    index_0 = np.where(np_actions == 10)[0]
    index_1 = np.where(np_actions > 10)[0]
    index_2 = np.where(np_actions < 10)[0]

    print(str(len(index_0)) + "/" + str(len(index_1)) + "/" + str(len(index_2)) + "/" + "%.4f" % env.cur_balance)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, sharex=True)

    if len(index_0) > 0:
        axs[0].scatter(index_0, test_close[index_0], c='red', label='hold', marker='^')
    if len(index_1) > 0:
        axs[0].scatter(index_1, test_close[index_1], c='green', label='buy', marker='>')
    if len(index_2) > 0:
        axs[0].scatter(index_2, test_close[index_2], c='blue', label='sell', marker='v')

    axs[0].plot(test_close)
    axs[0].legend()
    axs[0].set_ylabel('Close', fontsize=22)

    axs[1].plot(pv_history, c='red', label='pv')
    axs[1].plot(test_data['Close'].values * starting_balance / test_data['Close'].iloc[0], c='black', label='close')
    axs[1].legend()
    axs[1].set_ylabel('Portfolio', fontsize=22)
    axs[1].set_xlabel("date", fontsize=22)
    plt.savefig('DQN_S00001_test.png')
    plt.show()
    plt.pause(3)
    plt.close()


# 메인 함수: 트레이닝과 테스트 함수 호출
if __name__ == '__main__':
    starting_balance = 100000
    train(window_size=7, starting_balance=starting_balance, resume_epoch=0, max_epoch=1000)
    test(window_size=7, starting_balance=starting_balance, model_epi='100')
