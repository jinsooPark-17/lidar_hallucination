#!/usr/bin/env python3

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(1,3,4,stride=2, padding=1)
        self.conv2 = nn.Conv2d(3,6,4,stride=2, padding=1)
        self.conv3 = nn.Conv2d(6,10,4,stride=2, padding=1)
        self.fc1 = nn.Linear(16*16*10, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

        self.max_action = max_action
    
    def forward(self, state):
        # dim(state) = (batch, channel, size_x, size_y) = (batch, 1, 128, 128) 
        x = F.relu( self.conv1(state) )
        x = F.relu( self.conv2(x) )
        x = F.relu( self.conv3(x) )
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        return self.max_action * torch.tanh( self.fc3(x) )

class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.conv1 = nn.Conv2d(1,3,4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(3,6,4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(6,10,4, stride=2, padding=1)
        self.fc1 = nn.Linear(16*16*10 + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256,1)

        # Q2 architecture
        self.conv4 = nn.Conv2d(1,3,4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(3,6,4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(6,10,4, stride=2, padding=1)
        self.fc4 = nn.Linear(16*16*10 + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256,1)

    def forward(self, state, action):
        q2 = F.relu( self.conv4(state) )
        q2 = F.relu( self.conv5(q2) )
        q2 = F.relu( self.conv6(q2) )
        q2 = torch.flatten(q2, 1)
        # Concatenate extracted state with action
        q2 = torch.cat([q2, action], 1)
        q2 = F.relu( self.fc4(q2) )
        q2 = F.relu( self.fc5(q2) )
        q2 = self.fc6(q2)

        q1 = F.relu( self.conv1(state) )
        q1 = F.relu( self.conv2(q1) )
        q1 = F.relu( self.conv3(q1) )
        q1 = torch.flatten(q1, 1)
        # Concatenate extracted state with action
        q1 = torch.cat([q1, action], 1)
        q1 = F.relu( self.fc1(q1) )
        q1 = F.relu( self.fc2(q1) )
        q1 = self.fc3(q1)

        return q1, q2

    def Q1(self, state, action):
        q1 = F.relu( self.conv1(state) )
        q1 = F.relu( self.conv2(q1) )
        q1 = F.relu( self.conv3(q1) )
        q1 = torch.flatten(q1, 1)
        # Concatenate extracted state with action
        q1 = torch.cat([q1, action], 1)
        q1 = F.relu( self.fc1(q1) )
        q1 = F.relu( self.fc2(q1) )
        q1 = self.fc3(q1)
        return q1

class TD3(object):
    def __init__(self,
                action_dim=6, 
                max_action=10.0, 
                discount=0.99, 
                tau=0.005, 
                policy_noise=0.04, 
                noise_clip=0.1, 
                policy_update_freq=2
                ):
        self.actor = Actor(action_dim, max_action).to(device) # cuda operation
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(action_dim).to(device) # cuda operation
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_update_freq = policy_update_freq

        self.n_iter = 0

    def select_action(self, state):
        state = torch.FloatTensor( state ).to(device) # cuda operation
        return self.actor(state).cpu().data.numpy().flatten()

    def save(self, filename):
        torch.save( self.actor.state_dict(), filename+"_actor")
        torch.save( self.actor_optim.state_dict(), filename+"_actor_optimizer")
        torch.save( self.critic.state_dict(), filename+"_critic")
        torch.save( self.critic_optim.state_dict(), filename+"_critic_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename+"_actor", map_location=device))
        self.actor_optim.load_state_dict(torch.load(filename+"_actor_optimizer", map_location=device))
        self.actor_target = copy.deepcopy(self.actor)

        self.critic.load_state_dict(torch.load(filename+"_critic", map_location=device))
        self.critic_optim.load_state_dict(torch.load(filename+"_critic_optimizer", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)

    def train(self, replay_buffer, batch_size=256):
        self.n_iter += 1

        # sample from replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Add clipped noise to the selected action
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clip(-self.max_action, self.max_action)

            # compute target critic values
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
        
        # Get current Q estimates
        curr_Q1, curr_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(curr_Q1, target_Q) + F.mse_loss(curr_Q2, target_Q)

        # Optimize critics
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Delayed policy updates
        if self.n_iter % self.policy_update_freq == 0:
            # compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

if __name__=='__main__':
    from PIL import Image
    im = np.array(Image.open('sample.png').convert('L'))
    im = torch.tensor( im/255., dtype=torch.float32).view(1,128,128)

    batch = torch.stack([im, im,im,im,im], dim=0) # stack batch

    ac = Actor(6, 4)
    cr = Critic( 6 )

    actions = ac(batch)
    print(actions)
    q1, q2 = cr(batch, actions)
