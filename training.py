# -*- coding: utf-8 -*-
#! /usr/bin/python3
"""
Created on Thu Sep 21 16:15:53 2017

@author: cx
"""

from util import *
from cell import ConvLSTMCell
import util

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()

###declare some parameters that might be used 
        self.conv_pad = 0
        self.conv_kernel_size = 3
        self.conv_stride = 1
        self.pool_pad = 0
        self.pool_kernel_size = 3
        self.pool_stride = 3
        self.hidden_size = 64
        self.size = int((args.img_size+2*self.conv_pad-(self.conv_kernel_size-1)-1)/self.conv_stride+1)
        self.size1 = int((self.size+2*self.pool_pad-(self.pool_kernel_size-1)-1)/self.pool_stride+1)
###define layers
        self.conv = nn.Conv2d(
             in_channels=1,
             out_channels=8,
             kernel_size=3,
             stride=1,
             padding=0)
        self.pool = nn.MaxPool2d(
                     kernel_size=3
                     )
        self.convlstm1 = ConvLSTMCell(
                        shape=[self.size1,self.size1], 
                        input_channel=8, 
                        filter_size=3,
                        hidden_size=self.hidden_size)
        self.convlstm2 = ConvLSTMCell(
                        shape=[self.size1,self.size1], 
                        input_channel=self.hidden_size, 
                        filter_size=3,
                        hidden_size=self.hidden_size)
        self.deconv = nn.ConvTranspose2d(
                        in_channels=self.hidden_size , 
                        out_channels=1, 
                        kernel_size=6,
                        stride=3,
                        padding=0, 
                        output_padding=1, 
                        )
        self.relu = func.relu


    def forward(self,X):
        X_chunked = torch.chunk(X,args.seq_start,dim=1)
        X = None
        output = [None]*args.seq_length
        state_size = [args.batch_size, self.hidden_size]+[self.size1,self.size1]
        hidden1 = Variable(torch.zeros(state_size)).cuda()
        cell1 = Variable(torch.zeros(state_size)).cuda()
        hidden2 = Variable(torch.zeros(state_size)).cuda()
        cell2 = Variable(torch.zeros(state_size)).cuda()
        
        for i in range(args.seq_start):
                                                        
            output[i] = self.conv(X_chunked[i])     
            output[i] = self.pool(output[i])
            hidden1, cell1 = self.convlstm1(output[i],(hidden1,cell1))
            hidden2, cell2 = self.convlstm2(hidden1,(hidden2,cell2))
            output[i] = self.deconv(hidden2)
            output[i] = self.relu(output[i])
        
        for i in range(args.seq_start,args.seq_length):                                                 
            output[i] = self.conv(output[i-1])    
            output[i] = self.pool(output[i])
            hidden1, cell1 = self.convlstm1(output[i],(hidden1,cell1))
            hidden2, cell2 = self.convlstm2(hidden1,(hidden2,cell2))
            output[i] = self.deconv(hidden2)
            output[i] = self.relu(output[i])
            
        return output[args.seq_start:]


def run_training(args,reload=False):     

    #Initialize model
    if reload:
        model_list = []
        print("Reloading exsiting model")
        for model in os.listdir(args.model_dir):
            model_list.append(model)
        model = torch.load(model_list[-1])

    print('Initiating new model')

    summary = open(args.logs_train_dir+"5_10_2ly.txt","w") ## you can change the name of your summary. 
    self_built_dataset = util.Dataloader0(args.data_dir+args.trainset_name,
                                          args.seq_start,
                                          args.seq_length-args.seq_start)
    trainloader = DataLoader(
        self_built_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last = True)

    torch.manual_seed(1)
    model = Model()
    model = model.cuda()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    loss_ave = 0

######Train the model#######
    for epoch in range(args.epoches):

        print("--------------------------------------------")
        print("EPOCH:",epoch)
        t = time.time()

        for iteration,data in enumerate(trainloader,0):
            loss = 0
            # X is the given data while the Y is the real output
            X, Y = data
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
            optimizer.zero_grad()         

            output_list = model(X)
            for i in range(args.seq_length-args.seq_start):
                loss += criterion(output_list[i], Y[:,i,:,:])

            loss_ave += loss.data/100
            loss.backward()
            optimizer.step()
            
            if iteration%100==0 and iteration!=0:
 
                elapsed = time.time()-t
                t = time.time()

                print("EPOCH: %d, Iteration: %s, Duration %d s, Loss: %f" %(epoch,iteration,elapsed,loss_ave[0]))
                summary.write("Epoch: %d ,Iteration: %s, Duration %d s, Loss: %f \n" %(epoch,iteration,elapsed,loss_ave[0]))
                loss_ave = 0

        print("Finished an epoch.Saving the net....... ")
        torch.save(model,args.model_dir+"model_{0}.pkl".format(epoch))    

    summary.close()

if __name__=="__main__":

    run_training(args)
