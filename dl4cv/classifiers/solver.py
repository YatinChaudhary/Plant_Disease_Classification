from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func()

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        
        #optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        optim = self.optim([p for p in model.parameters() if p.requires_grad], **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        # filter(lambda p: p.requires_grad, model.parameters())
        use_gpu = torch.cuda.is_available()


        print('START TRAIN.')

        for epoch in range(num_epochs):
            total_train  = 0
            correct_train = 0

            train_loss = 0
            val_loss = 0

            if use_gpu:
                model = model.cuda()
            
            #Training Loop
            for i, data in enumerate(train_loader, 0):
                
                # get the inputs, wrap them in Variable
                input, label = data
                if use_gpu:
                    input = input.cuda()
                    label = label.cuda()
                inputs, labels = Variable(input), Variable(label)
                
                # zero the parameter gradients
                optim.zero_grad()

                # forward pass
                outputs = model(inputs)

                # calculation of loss
                loss = self.loss_func(outputs, labels)

                # backpropagation
                loss.backward()

                # single gradient step
                optim.step()
                
                # calculating correct predictions
                _,predicted_train = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == label).sum()

                train_loss += loss.data[0]
                # Storing values
                #self.train_loss_history.append(loss.data[0])
                #if (i+1) % log_nth == 0:
                #    print('[Iteration:', i, '/', iter_per_epoch, ']Train loss:', \
                #            self.train_loss_history[-1])

                #if (i+1) % iter_per_epoch == 0:
            self.train_acc_history.append(correct_train/float(total_train))
            self.train_loss_history.append(train_loss / float(len(train_loader)))
            print('[Epoch:', epoch, '/', num_epochs, '] Train acc/loss:', \
                    self.train_acc_history[-1], '/', self.train_loss_history[-1])


            #Validation Loop
            correct_val = 0
            val_size = 0
            #loss_val = 0

            for i, data_val in enumerate(val_loader, 0):
                
                # get the inputs, wrap them in Variable
                input_val, label_val = data_val
                if use_gpu:
                    input_val = input_val.cuda()
                    label_val = label_val.cuda()
                inputs_val, labels_val = Variable(input_val), Variable(label_val)

                # forward pass
                output_val = model(inputs_val)

                # calculating loss
                loss_val = self.loss_func(output_val, labels_val)

                # calculating correct predictions
                _,predicted_val = torch.max(output_val.data, 1)
                val_size += label_val.size(0)
                correct_val += (predicted_val == label_val).sum()

                val_loss += loss_val.data[0]

            # storing values
            self.val_acc_history.append(correct_val/float(val_size))
            self.val_loss_history.append(val_loss / float(len(val_loader)))
            print('[Epoch:', epoch, '/', num_epochs, '] Val acc/loss:', \
                    self.val_acc_history[-1], '/', self.val_loss_history[-1])

        print('FINISH TRAIN.')
        return self.train_loss_history, self.val_loss_history
