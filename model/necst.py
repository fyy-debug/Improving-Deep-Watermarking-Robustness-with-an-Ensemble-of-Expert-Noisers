import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class NECST(nn.Module):
    """
    NECST model for channel coding
    """

    def __init__(self, FLAGS, device: torch.device):
        super(NECST, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(FLAGS.message_length, 512),
            nn.LeakyReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(512, FLAGS.redundant_length),
        )

        self.decoder = nn.Sequential(
            nn.Linear(FLAGS.redundant_length, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, FLAGS.message_length),
        )
        device = torch.device('cuda:0' if  torch.cuda.is_available() else 'cpu')
        self.device = device
        self.prob1 = 0.0
        self.prob2 = 0.3
        self.message_length = FLAGS.message_length
        self.redundant_length = FLAGS.redundant_length
        self.bs = 100
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)
        self.mse_loss = nn.MSELoss()
        self.max_iter = FLAGS.necst_iter

    def encode(self, message):
        redundant_message = self.encoder(message)
        redundant_message = torch.clamp(redundant_message, min=0.0, max=1.0)
        redundant_message = (redundant_message >= 0.5).float()
        return redundant_message

    def decode(self, red_dec_message):
        red_dec_message = torch.clamp(red_dec_message,min=0.0,max=1.0)
        red_dec_message = (red_dec_message >= 0.5).float()
        decoded_message = self.decoder(red_dec_message)
        return decoded_message

    def pretrain(self):
          for step in range(self.max_iter):
              self.optimizer.zero_grad()
              bsc_prob = random.uniform(self.prob1, self.prob2)
              messages = torch.Tensor(np.random.choice([0, 1], (self.bs, self.message_length))).to(self.device)
              redundant_messages = self.encoder(messages)
              # noise_message = (redundant_messages.round().int() ^ torch.Tensor(np.random.random(redundant_messages.shape) <= bsc_prob).int().to(self.device)).float()
              flip_flag = torch.Tensor(self.bs, self.redundant_length).uniform_(self.prob1, self.prob2).to(self.device)
              flip_flag = torch.bernoulli(flip_flag)
              redundant_messages = torch.clamp(redundant_messages,min=0.0,max=1.0)
              redundant_messages[flip_flag == 1.] = 1 - redundant_messages[flip_flag == 1.]
              #redundant_messages = (redundant_messages >= 0.5).float()
              #print(redundant_message)
              decoded_messages = self.decoder(redundant_messages)
              # calculating loss
              # loss = self.mse_loss(decoded_messages, messages)

              loss = F.binary_cross_entropy_with_logits(decoded_messages, messages)
              loss.backward()
              self.optimizer.step()

              if step % 1000 == 0:
                  decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
                  bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                          self.bs * messages.shape[1])
                  print("loss : {} bitwise_avg_err:{}".format(loss.item(), bitwise_avg_err))

          for child in self.encoder.children():
              print("encoder", child)
              for param in child.parameters():
                  param.requires_grad = False
          for child in self.decoder.children():
              print("decoder", child)
              for param in child.parameters():
                  param.requires_grad = False