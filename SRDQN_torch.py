import os
import time
from time import gmtime, strftime

import numpy as np
import random
from collections import deque
import torch.nn as nn
import torch
import torch.optim as optim


class dqnylz(nn.Module):
    def __init__(self, agentNum,config):
        super(dqnylz, self).__init__()
        random.seed(1)
        self.agentNum=agentNum
        self.config=config
        modelNumber = 'model' + str(agentNum + 1)
        # self.addressName = 'model'+str(agentNum+1)+'/savetrained' + str(self.config.address) + '/network-'
        self.address = os.path.join(self.config.model_dir,
                                    modelNumber)  # 'model'+str(agentNum+1)+'/savetrained'+ str(self.config.address)
        self.addressName = self.address + '/network-'
        if self.config.maxEpisodesTrain != 0:
            self.epsilon = config.epsilonBeg
        else:
            self.epsilon = 0
        #self.epsilonRed = self.epsilonBuild()
        self.inputSize = self.config.stateDim * self.config.multPerdInpt
        self.timeStep = 0
        self.learning_rate = 0  # this is used when we have decaying
        self.iflrReseted = False  # this is used to manage the scale of lr

        # init replay memory
        self.replayMemory = deque()
        self.replaySize = 0

        # create input placeholders
        #self.createInputs()

        self.current_net = self.createQnet()
        self.target_net = self.current_net
        self.target_net.load_state_dict(self.current_net.state_dict())
        self.optimizer = optim.Adam(self.current_net.parameters(), self.learning_rate)
        # self.traget_net=Qnet(outputs)
        # self.BATCH_SIZE = batch_size
        # self.Gamma = 0
        # self.memory = ReplayMemory(1000)
        # self.optimizer = optim.Adam(self.current_net.parameters(), lr=learning_rate, weight_decay=0.001)

    def setInitState(self, observation):
        self.currentState = np.stack([observation for _ in range(self.config.multPerdInpt)],
                                     axis=0)  # multPerdInpt observations stacked. each row is an observation

    def createQnet(self):
        modules=[]
        modules.append(nn.Linear(self.config.nodes[0],self.config.nodes[1]))
        # print(self.config.nodes)
        for j in range(1,self.config.NoHiLayer+1):
            modules.append(nn.Linear(self.config.nodes[j],self.config.nodes[j+1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(self.config.nodes[j+1],self.config.baseActionSize))
        #print(self.config.baseActionSize)
        Qnet=nn.Sequential(*modules)
        return Qnet


    def trainQNetwork(self):
        minibatch = random.sample(self.replayMemory, self.config.batchSize)
        state_batch = [data[0] for data in minibatch]  # dim: each item is multPerInput*stateDim
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]
        # print(len(state_batch))
        currentState = torch.tensor(state_batch).to(torch.float32).reshape(len(state_batch),50)
        Q_batch=self.current_net(currentState).to(torch.float32)
        # print(Q_batch)
        # print(torch.max(Q_batch,axis=1).values)
        # print((1-np.array(minibatch)[:,4])) 

        Q_target_batch=torch.tensor(reward_batch).to(torch.float32) + self.config.gamma*torch.max(Q_batch,axis=1).values.to(torch.float32)
        Q_batch_1=torch.max(Q_batch,axis=1).values.to(torch.float32)


        criterion = nn.MSELoss()
        loss=criterion(Q_target_batch,Q_batch_1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if (self.timeStep + 1) % self.config.saveInterval == 0:
            torch.save(self.current_net,self.addressName)
            print("network weights are saved")

        if self.timeStep % self.config.dnnUpCnt == 0:
            self.target_net=self.current_net
            #self.copyTargetQNetwork()

    def getDNNAction(self, playType):
        action = np.zeros(self.config.actionListLen)
        action_index = 0

        currentState=torch.tensor(self.currentState).to(torch.float32).reshape(50)
        if playType == "train":
            if (random.random() <= self.epsilon) or (self.replaySize < self.config.minReplayMem):
                action_index = random.randrange(self.config.actionListLen)
                action[action_index] = 1
            else:
                QValue=self.current_net(currentState)
                action_index = torch.argmax(QValue)
                action[action_index] = 1
        elif playType == "test":
            QValue = self.current_net(currentState)

            action_index = torch.argmax(QValue)
            action[action_index] = 1

    def train(self,nextObservation,action,reward,terminal,playType):
        newState = np.append(self.currentState[1:, :], [nextObservation], axis=0)

        if playType == "train":
            if self.config.MultiAgent:
                if self.config.MultiAgentRun[self.agentNum]:
                    self.replayMemory.append([self.currentState, action, reward, newState, terminal])
                    self.replaySize = len(self.replayMemory)
            else:
                self.replayMemory.append([self.currentState, action, reward, newState, terminal])
                self.replaySize = len(self.replayMemory)

            if self.replaySize > self.config.maxReplayMem and self.config.MultiAgentRun[self.agentNum]:
                self.replayMemory.popleft()
                self.trainQNetwork()
                state = "train"
                self.timeStep += 1

            elif self.replaySize >= self.config.minReplayMem and self.config.MultiAgentRun[self.agentNum]:
                # Train the network
                state = "train"
                self.trainQNetwork()
                self.timeStep += 1
            else:
                state = "observe"

            if terminal and state == "train":
                self.epsilonReduce()

        # print(info)
        # print("AGENT", self.agentNum,"/TRAINING_ITER", self.timeStep, "/ STATE", state, \)
        # "/ EPSILON", self.epsilon

        self.currentState = newState

    def epsilonReduce(self):
        def epsilonReduce(self):
            # Reduces the values of epsilon at each iteration of episode
            if self.epsilon > self.config.epsilonEnd:
                self.epsilon -= self.epsilonRed

    def epsilonBuild(self):  # this function specifies how much we should deduct from /epsilon at each game
        betta = 0.8
        if self.config.maxEpisodesTrain != 0:
            epsilon_red = (self.config.epsilonBeg - self.config.epsilonEnd) / (self.config.maxEpisodesTrain * betta)
        else:
            epsilon_red = 0
        return epsilon_red

