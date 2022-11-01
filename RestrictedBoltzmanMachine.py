"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved
    Filename: RestrictedBoltzmanMachine.py
    Description: Implementation for RBM from strach.
    Created by Melrose-Lbt 2022-11-1
"""
import torch
import numpy as np
import matplotlib.pyplot as plt


class RBM:
    def __init__(self, v_num, h_num, epoch, batch_size, k, mode='bernoulli', lr=0.091, device=False) -> None:
        
        self.v_num = v_num  # visible unit number
        self.h_num = h_num  # hidden unit number
        self.mode = mode    # bernoulli or gaussian
        self.lr = lr        # learning rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.k = k

        if torch.cuda.is_available() and device==True:
            dev = "cuda:0"
        else:
            dev = "cpu"
        
        self.device = torch.device(dev)

        # initialize the parameters
        std_val = 4 * np.sqrt(6. / (self.v_num + self.h_num))
        self.W = torch.normal(mean=0, std=std_val, size=(self.h_num, self.v_num))
        self.v_bias = torch.zeros(size=(1, self.v_num), dtype=torch.float32)
        self.h_bias = torch.zeros(size=(1, self.h_num), dtype=torch.float32)

    def sample_hidden(self, v_data):

        t_v_data = torch.transpose(v_data, 1, 2)
        # print("w shape is :{}  t_v_data shape is :{} ".format(self.W.shape, t_v_data.shape))
        a = torch.matmul(self.W, t_v_data)
        b = a + self.h_bias.t()
        hidden_prob = torch.sigmoid(torch.matmul(self.W, t_v_data) + self.h_bias.t())
        
        if self.mode == 'bernoulli':
            return torch.transpose(hidden_prob, 1, 2), torch.bernoulli(hidden_prob).transpose(1, 2)
        if self.mode == 'guassian':
            return torch.transpose(hidden_prob, 1, 2), torch.add(hidden_prob, torch.normal(mean=0, std=1, size=hidden_prob.shape)).transpose(1, 2) #torch.normal(mean=0, std=1, size=hidden_prob.shape)

    def sample_visible(self, h_data):
        
        visible_prob = torch.sigmoid(torch.matmul(h_data, self.W) + self.v_bias)
        
        if self.mode == 'bernoulli':
            return visible_prob, torch.bernoulli(visible_prob)
        if self.mode == 'guassian':
            return visible_prob, torch.add(visible_prob, torch.normal(mean=0, std=1, size=visible_prob.shape)) #torch.normal(mean=0, std=1, size=visible_prob.shape)
    
    def reconstruct_visible(self, h_data):
        visible_prob = torch.sigmoid(torch.matmul(h_data, self.W) + self.v_bias)
        return visible_prob

    def contrastive_divergence(self, v0, vk, ph0, phk):

        dW = (torch.matmul(torch.transpose(ph0, 1, 2), v0) - torch.matmul(torch.transpose(phk, 1, 2), vk)) / v0.shape[1]
        dW = torch.mean(dW, dim=0)
        dv_b = torch.mean(v0 - vk)
        dh_b = torch.mean(ph0 - phk)

        # update the parameters
        self.W += self.lr * dW
        self.v_bias += self.lr * dv_b
        self.h_bias += self.lr * dh_b

    def train(self, dataloader, testloader):

        for epo in range(self.epoch):
            print("EPOCH {} begin:".format(epo))
            train_loss = 0
            counter = 0
            for img, label in dataloader:
                
                vk = img.view((self.batch_size, 1, -1))
                v0 = img.view((self.batch_size, 1, -1))

                # ph0, _ = self.sample_hidden(v0)

                # for k in range(self.k):
                #     _, hk = self.sample_hidden(vk)
                #     _, vk = self.sample_visible(hk)
                
                # phk, _ = self.sample_hidden(vk)
                # self.contrastive_divergence(v0, vk, ph0, phk)

                ph0, h0 = self.sample_hidden(v0)

                for k in range(self.k):
                    _, hk = self.sample_hidden(vk)
                    _, vk = self.sample_visible(hk)
                
                vk = self.reconstruct_visible(hk)
                phk, hk = self.sample_hidden(vk)
                self.contrastive_divergence(v0, vk, h0, hk)

                train_loss = torch.mean(torch.abs(v0 - vk))
                print("Train loss is {}".format(train_loss))
                counter += 1

                if epo == 2 and counter % 32 == 0:
                    plt.figure()
                    plt.subplot(1, 2, 1)
                    plt.imshow(v0[1].squeeze().view(28, 28))
                    plt.title('Ground Truth')
                    plt.subplot(1, 2, 2)
                    plt.imshow(vk[1].squeeze().view(28, 28))
                    plt.title('Reconstruct')
                    plt.show()
            self.test(testloader)
    
    def test(self, dataloader):
        for img, label in dataloader:
            v0 = img.view((self.batch_size, 1, -1))
            ph0, h0 = self.sample_hidden(v0) 
            vk = self.reconstruct_visible(h0)
            train_loss = torch.mean(torch.abs(v0 - vk))
            print("Test loss is {}".format(train_loss))
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(v0[1].squeeze().view(28, 28))
            plt.title('Test Ground Truth')
            plt.subplot(1, 2, 2)
            plt.imshow(vk[1].squeeze().view(28, 28))
            plt.title('Test Reconstruct')
            plt.show()