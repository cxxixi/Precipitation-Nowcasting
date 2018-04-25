from util import *
from training import *

 
def restore_net(idx):

    print("Reloading previous model")
    net = torch.load(args.model_dir+"model_{0}.pkl".format(idx))
    return net


def test(idx,model,reload=False):

    if reload:
        model = restore_net(idx)
        model.eval()
### loading validation dataset
    self_built_dataset = util.Dataloader0(args.data_dir+args.testset_name,
                                          args.seq_start,
                                          args.seq_length-args.seq_start,
                                          rot=False)
    name_list = self_built_dataset.all_list  
    trainloader = DataLoader(
        self_built_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True) 

    for iteration,valid_data in enumerate(trainloader,0):
        
        print(iteration)
        valid_X,valid_Y = valid_data
        valid_X = Variable(valid_X, requires_grad=False).cuda()

        output_list = model(valid_X)

        for j in range(args.batch_size):

            start_time = name_list[iteration*args.batch_size+j][args.seq_start-1]
            time_list = [name_list[iteration*args.batch_size+j][i] for i in range(args.seq_start,args.seq_length)]
            A = valid_X[j][-1].data.cpu().numpy().reshape(args.img_size,args.img_size)
            A = (A+0.5).astype(np.uint8)
            A = Image.fromarray(A)
            path = args.img_dir+start_time.split("/")[-1]
            A.save(path)

            for k in range(args.seq_length-args.seq_start):
                A = output_list[k][j,0,:,:].data.cpu().numpy().reshape(args.img_size,args.img_size)
                A = (A+0.5).astype(np.uint8)
                A = Image.fromarray(A)
                path = args.img_dir+time_list[k][:-4].split("/")[-1]+"_{}.png".format(k)
                A.save(path)

        output_list = None

if __name__== "__main__":

    test(42,None,reload=True)




