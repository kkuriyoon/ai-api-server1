import numpy as np

class AndModel:
    def __init__(self):
        # 파라메터
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

    def train(self):
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 0, 0, 1])        
        
        
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # 총 입력 계산
                total_input = np.dot(inputs[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = outputs[i] - prediction

                '''
                print(f'inputs[i] : {inputs[i]}')
                print(f'weights : {self.weights}')
                print(f'bias before update: {self.bias}')
                print(f'prediction: {prediction}')
                print(f'error: {error}')
                '''


                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error
                #print('====')        

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)    
    

class OrModel:
    def __init__(self):
        # OR 2개 입력
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

    def train(self):
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
        outputs = np.array([0, 1, 1, 1])

        for _ in range(epochs):
            for i in range(len(inputs)):
                #총 계산 입력
                total_input = np.dot(inputs[i], self.weights) + self.bias
                #예측 출력 계산
                prediction = self.step_function(total_input)
                #오차 계산
                error = outputs[i] - prediction

                #가중치와 편향 업데이트
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)
    





class NotModel:
    def __init__(self):
        # NOT은 입력 1개
        self.weight = np.random.rand(1)
        self.bias = np.random.rand(1)

    def train(self):
        learning_rate = 0.3
        epochs = 20
        inputs = np.array([[0],[1]])
        outputs = np.array([1, 0])

        for _ in range(epochs):
            for i in range(len(inputs)):
                total_input = np.dot(inputs[i], self.weight) + self.bias
                prediction = self.step_function(total_input)
                error = outputs[i] - prediction

                self.weight += learning_rate * error * inputs[i]
                self.bias += learning_rate * error

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        total_input = np.dot(x, self.weight) + self.bias
        return self.step_function(total_input)





import torch
import torch.nn as nn
import torch.optim as optim


class XORModel:
    def __init__(self):
        # 2 -> 2 -> 1 구조의 간단한 MLP
        self.net = nn.Sequential(
            nn.Linear(2, 2), #입력크기2, 출력 크기 2(은닉층)
            nn.ReLU(), #은닉층 활성화 함수
            nn.Linear(2, 1), #은닉층 출력(2) -> 최종 출력(1)
            nn.Sigmoid() #시그모이드 - 범위 0~1
        )
        self.criterion = nn.BCELoss()  # 이진분류
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.5) #확률적 경사하강법(SGD)로 모델의 파라미터 업데이트, lr은 학습률

    def train(self):
        inputs = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float)
        outputs = torch.tensor([[0],[1],[1],[0]], dtype=torch.float)

        epochs = 1000
        for _ in range(epochs):
            pred = self.net(inputs)
            loss = self.criterion(pred, outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        # x: 예 [0,1], [1,1] 형태
        inp = torch.tensor([x], dtype=torch.float)  # shape [1,2]
        pred = self.net(inp)                       # shape [1,1]
        # 0.5 기준으로 0/1 판별
        return int(pred.item() > 0.5)