import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader





######################## define two different transforms for train and test ##############


data_transforms = {"train":
    transforms.Compose([transforms.Resize((28,28)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]),
    "test":
    transforms.Compose([transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])} 

########################  download MNIST dataset  #################
mnist_train = dataset.MNIST(root='./data', 
                            train=True, 
                            transform=data_transforms["train"],
                            download=True)
test_set = dataset.MNIST(root='./data', 
                            train=False, 
                            transform=data_transforms["test"],
                            download=True)



########################  divide the train data into val and train  ######

train_set,val_set =  torch.utils.data.random_split(mnist_train, [40000, 20000])

######################## define dataloader for train, val and test   #####################

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False)




#####################  define LSTM model #################################

class myLSTM(nn.Module):
    def __init__(self, input_dim,hidden_dim,num_layers,output_dim):
        
        super(myLSTM,self).__init__()
        
        self.input_dim =  input_dim #28*28
        self.hidden_dim = hidden_dim #100
        self.num_layers = num_layers #2
        self.output_dim = output_dim #10


        self.model = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.classifier = nn.Linear(hidden_dim, output_dim)


    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to('cuda')
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to('cuda')
        out, (hn, cn) = self.model(x, (h0.detach(), c0.detach()))
        # only the last h from sequence is important for evaluation (delete the rest and keep the last h)
        y = self.classifier(out[:,-1,:])

        return y




def Train(train_dataloader,val_dataloader,num_epochs = 100):

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1


    input_dim = 28
    hidden_dim = 100
    num_layers = 2  # Stacked LSTM
    output_dim = 10

    

    model = myLSTM(input_dim, hidden_dim, num_layers, output_dim)
    model = model.to('cuda')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

    for e in range(num_epochs):
        train_acc = 0
        train_total = 0
        train_loss = 0


        


        for i, (images,labels) in enumerate(train_dataloader):
            images  = images[:,-1,:,:].requires_grad_()
            images = images.to('cuda')
            labels = labels.to('cuda')

            optimizer.zero_grad()

            out = model(images)

            loss = criterion(out,labels)

            loss.backward()

            optimizer.step()

            _,index = torch.max(out.data,1)


            train_acc += len(labels[index== labels])
            train_total += labels.size(0)
            train_loss += loss.item()

        print("loss",train_loss/train_total) 
        print("train acc =", train_acc/train_total) 
        val_loss, val_acc = Test(val_dataloader,model)
        print("val loss", val_loss)
        print("val acc", val_acc)

        

    return model
    


def Test(dataloader,model):
    test_acc = 0
    test_total = 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss()

    for i, (images,labels) in enumerate(dataloader):
        images  = images[:,-1,:,:].requires_grad_()
        images = images.to('cuda')
        labels = labels.to('cuda')

        
        out = model(images)

        loss = criterion(out,labels)

        
        _,index = torch.max(out.data,1)


        test_acc += len(labels[index== labels])
        test_total += labels.size(0)
        test_loss += loss.item()

    return test_loss/test_total, test_acc/test_total


model = Train(train_dataloader,val_dataloader)
test_loss,test_acc = Test(test_data_loader,model)

print("test accuracy = test_acc)







