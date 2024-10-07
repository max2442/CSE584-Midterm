import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ax.service.managed_loop import optimize
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
data = pd.read_csv("total_data.csv")
data = data.sample(frac=1).reset_index(drop=True)
training = data.loc[:int(len(data)*.72)]
x = []
y = []
for a in range(len(training)):
    x += tokenizer(training.at[a,"Text"],padding='max_length',max_length=128)['input_ids']
    temp = [0.,0.,0.,0.,0.,0.,0.,0.]
    temp[training.at[a,"Label"]]=1.
    y += temp
x = torch.from_numpy(np.array(x).reshape((len(x)//128,128)))
y = torch.from_numpy(np.array(y).reshape((len(y)//8,8)))
train_dataset = TensorDataset(x,y)
validation = data.loc[int(len(data)*.72):int(len(data)*.9)].reset_index(drop=True)
x = []
y = []
for a in range(len(validation)):
    x += tokenizer(validation.at[a,"Text"],padding='max_length',max_length=128)['input_ids']
    temp = [0., 0., 0., 0., 0., 0., 0., 0.]
    temp[validation.at[a, "Label"]] = 1.
    y += temp
x = torch.from_numpy(np.array(x).reshape((len(x)//128,128)))
y = torch.from_numpy(np.array(y).reshape((len(y)//8,8)))
val_dataset = TensorDataset(x,y)
test = data.loc[int(len(data)*.9):].reset_index(drop=True)
x = []
y = []
for a in range(len(test)):
    x += tokenizer(test.at[a,"Text"],padding='max_length',max_length=128)['input_ids']
    temp = [0., 0., 0., 0., 0., 0., 0., 0.]
    temp[test.at[a, "Label"]] = 1.
    y += temp
x = torch.from_numpy(np.array(x).reshape((len(x)//128,128)))
y = torch.from_numpy(np.array(y).reshape((len(y)//8,8)))
test_dataset = TensorDataset(x,y)

class CustomBlock(nn.Module):
    def __init__(self,inp,p,rat,last=False,first=False):
        super(CustomBlock, self).__init__()
        self.layer0_bn = nn.BatchNorm1d(inp)
        self.layer0_do=nn.Dropout(p)
        self.layer1=nn.Linear(inp, int(inp*rat))
        self.layer1_bn=nn.BatchNorm1d(int(inp*rat))
        self.layer1_do=nn.Dropout(p)
        self.layer2=nn.Linear(int(inp*rat), inp)
        self.layer2_bn=nn.BatchNorm1d(inp)
        self.layer2_do=nn.Dropout(p)
#         self.layer3 = nn.Linear(inp,inp//2)
        self.layer3 = nn.Linear(inp,inp)
        self.skip = nn.Identity()
        self.last=last
        self.first = first
    def forward(self, x):
        out1 = self.layer1_do(self.layer1_bn(nn.functional.relu(self.layer1(self.layer0_do(self.layer0_bn(x))))))
        out2 = self.layer2_do(self.layer2_bn(nn.functional.relu(self.layer2(out1)) + self.skip(x)))
        if self.last:
            out3 = nn.functional.relu(self.layer3(out2))
        else:
            out3 = nn.functional.relu(self.layer3(out2))
        return out3

class CustomNet(nn.Module):
    def __init__(self,inp,p,num_block,rat):
        super(CustomNet, self).__init__()
        self.blocks = []
        curr = inp
        for a in range(num_block):
            if a==num_block-1:
                self.blocks.append(CustomBlock(curr,p,rat,True))
            elif a==0:
                self.blocks.append(CustomBlock(curr, p, rat,first=True))
            else:
                self.blocks.append(CustomBlock(curr,p,rat))
            #curr = curr//2
        self.fc_layer1=nn.Linear(curr, 128)
        self.fc_layer1_bn=nn.BatchNorm1d(128)
        self.fc_layer1_do=nn.Dropout(.2)
        self.fc_layer2=nn.Linear(128, 32)
        self.fc_layer2_bn=nn.BatchNorm1d(32)
        self.fc_layer2_do=nn.Dropout(.2)
        self.fc_layer3=nn.Linear(32, 8)

    def forward(self, x):
        out=x
        for a in range(len(self.blocks)):
            out = self.blocks[a](out)
        out = self.fc_layer1_do(self.fc_layer1_bn(nn.functional.relu(self.fc_layer1(out))))
        out = self.fc_layer2_do(self.fc_layer2_bn(nn.functional.relu(self.fc_layer2(out))))
        out = nn.functional.softmax(self.fc_layer3(out))
        return out

    def eval_mode(self):
        for a in self.blocks:
            a.eval()
        self.eval()

    def train_mode(self):
        for a in self.blocks:
            a.train()
        self.train()

def init_net(parameterization):
    model = CustomNet(128,parameterization.get("dropout_val", 0.3),parameterization.get("num_block", 2),parameterization.get("node_ratio", 1.33))
    return model

def net_train(net, train_loader, parameters,dtype, device,n_epochs=100,val_loader=None):
    net.to(dtype=dtype, device=device)
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    model_params = []
    for a in net.blocks:
        model_params+=list(a.parameters())
    model_params+=list(net.parameters())
    optimizer = optim.SGD(model_params, # or any optimizer you prefer
                        lr=parameters.get("lr", 0.001), # 0.001 is used if no lr is specified
                        momentum=parameters.get("momentum", 0.9)
    )


    n_epochs = n_epochs#parameters.get("num_epochs", 100) # Play around with epoch number
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=n_epochs*int(len(train_dataset)/parameters.get("batch_size", 32)))
    best_accuracy = 0
    count =0
    for _ in range(n_epochs):
        for inputs, labels in train_loader:
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            #labels = labels.reshape((len(labels),1))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        if val_loader!=None:
            epoch_acc = 0
            for inputs, labels in val_loader:
                # move data to proper dtype and device
                inputs = inputs.to(dtype=dtype, device=device)
                labels = labels.to(device=device)
                # labels = labels.reshape((len(labels),1))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                epoch_acc += torch.sum(torch.argmax(outputs,dim=1)==torch.argmax(labels,dim=1)).item()/len(outputs)
            if epoch_acc/len(val_loader)>best_accuracy:
                best_accuracy=epoch_acc/len(val_loader)
                torch.save(net.state_dict(), "best_model"+str(count))
                count+=1
    if val_loader!=None:
        net.load_state_dict(torch.load("best_model"+str(count-1)))
    return net

def evaluate_custom(
net: nn.Module, data_loader: DataLoader, dtype: torch.dtype, device: torch.device, post_train=False
) -> float:
    net.eval_mode()
    score = 0
    los = 0
    target_los ={0:[0,0],1:[0,0],2:[0,0],3:[0,0],4:[0,0],5:[0,0],6:[0,0],7:[0,0]}
    fin_target_los = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0}
    store_pred = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
    store_labels = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
    cc=0
    with torch.no_grad():
        for inputs, labels in data_loader:
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            #labels = labels.reshape((len(labels),1))
            outputs = net(inputs)
            if post_train:
                for a in range(8):
                    store_pred[a] += outputs[:,a]
                    store_labels[a] +=labels[:,a]
                for a in range(len(inputs)):
                    target_los[torch.argmax(labels[a]).item()][1]+=1
                    if torch.argmax(outputs[a]) == torch.argmax(labels[a]):
                        target_los[torch.argmax(labels[a]).item()][0] += 1
            score += torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).item()/len(outputs)
            sim_funct = nn.functional.cross_entropy
            sim = sim_funct(outputs,labels)
            los +=sim.item()
            cc+=1
    if post_train:
        for a in range(8):
            fpr, tpr, _ = roc_curve(store_labels[a], store_pred[a])
            roc_auc = roc_auc_score(store_labels[a], store_pred[a])
            plt.figure(1)
            plt.plot([0, 1], [0, 1], 'k-')
            plt.plot(fpr, tpr, label='LMM'+str(a)+'(area={: .3f})'.format(roc_auc))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
            #plt.show()
        plt.savefig('ROC Curve LLMs.png')
        for a in target_los.keys():
            fin_target_los[a]=target_los[a][0]/target_los[a][1]
    return score/cc,los/cc,fin_target_los#1/(score/cc)

def train_evaluate(parameterization):

    # Get neural net
    untrained_net = init_net(parameterization)

    if len(train_dataset)%parameterization.get("batch_size", 32)==1:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                batch_size=parameterization.get("batch_size", 32),
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True,drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=parameterization.get("batch_size", 32),
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    # train
    trained_net = net_train(net=untrained_net, train_loader=train_loader,
                            parameters=parameterization,dtype=dtype, device=device)
    trained_net.eval_mode()

    # return the accuracy of the model as it was trained in this run
    acc,loss,target_los = evaluate_custom(
        net=trained_net,
        data_loader=val_loader,
        dtype=dtype,
        device=device,
    )
    trained_net.train_mode()
    return acc

dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-6, 1.0], "log_scale": True},
        {"name": "batch_size", "type": "range", "bounds": [16, 256]},
        {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "num_block", "type": "range", "bounds": [1, 4]},
        {"name": "dropout_val", "type": "range", "bounds": [0.0,1.0]},
        {"name": "node_ratio", "type": "range", "bounds": [.1,3.0]},
        #{"name": "num_epochs", "type": "range", "bounds": [1, 200]},
        #{"name": "stepsize", "type": "range", "bounds": [20, 40]},
    ],

    evaluation_function=train_evaluate,
    objective_name='accuracy',
    total_trials=300
)

print(best_parameters)
means, covariances = values
print(means)
print(covariances)

#best_parameters={'lr': 0.20900230593060978, 'batch_size': 129, 'momentum': 0.6520050118441927, 'num_block': 1, 'dropout_val': 0.27172874727196, 'node_ratio': 0.3916524610093416}
#
model = CustomNet(128,best_parameters["dropout_val"],best_parameters["num_block"],best_parameters["node_ratio"])
#
if len(train_dataset) % best_parameters["batch_size"] == 1:
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=best_parameters["batch_size"],
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True, drop_last=True)
else:
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=best_parameters["batch_size"],
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=64,
                                         shuffle=True,
                                         num_workers=0,
                                         pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=64,
                                         shuffle=True,
                                         num_workers=0,
                                         pin_memory=True)

trained_model = net_train(net=model, train_loader=train_loader,
                            parameters=best_parameters,dtype=dtype, device=device,n_epochs=1000,val_loader=val_loader)


trained_model.eval_mode()
acc,loss,target_loss = evaluate_custom(
        net=trained_model,
        data_loader=test_loader,
        dtype=dtype,
        device=device,
        post_train=True
)
print(acc)
print(loss)
print(target_loss)
