import torch    
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

print(device)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn1= nn.BatchNorm2d(64)
        self.bn2= nn.BatchNorm2d(256)
        self.bn3= nn.BatchNorm2d(128)
        self.bn4= nn.BatchNorm2d(512)
        self.bn5= nn.BatchNorm2d(1024)
        self.bn6= nn.BatchNorm2d(2048)
        self.conv1=nn.Conv2d(3,64,3,stride=2,padding=1,bias=False)
        self.conv2a1=nn.Conv2d(64,64,1,bias=False)
        self.conv2a2=nn.Conv2d(64,64,3,padding=1,bias=False)
        self.conv2a3=nn.Conv2d(64,256,1,bias=False)
        self.conv2a4=nn.Conv2d(64,256,1,bias=False)
        self.conv3a1=nn.Conv2d(256,128,1,stride=2,bias=False)
        self.conv3a2=nn.Conv2d(128,128,3,padding=1,bias=False)
        self.conv3a3=nn.Conv2d(128,512,1,bias=False)
        self.conv3a4=nn.Conv2d(256,512,1,stride=2,bias=False)
        self.conv4a1=nn.Conv2d(512,256,1,stride=2,bias=False)
        self.conv4a2=nn.Conv2d(256,256,3,padding=1,bias=False)
        self.conv4a3=nn.Conv2d(256,1024,1,bias=False)
        self.conv4a4=nn.Conv2d(512,1024,1,stride=2,bias=False)
        self.conv5a1=nn.Conv2d(1024,512,1,stride=2,bias=False)
        self.conv5a2=nn.Conv2d(512,512,3,padding=1,bias=False)
        self.conv5a3=nn.Conv2d(512,2048,1,bias=False)
        self.conv5a4=nn.Conv2d(1024,2048,1,stride=2,bias=False)
        self.conv2b1 = nn.ModuleList([nn.Conv2d(256,64,1,bias=False) for i in range(2)])
        self.conv2b2 = nn.ModuleList([nn.Conv2d(64,64,3,padding=1,bias=False) for i in range(2)])
        self.conv2b3 = nn.ModuleList([nn.Conv2d(64,256,1,bias=False) for i in range(2)])
        self.conv3b1 = nn.ModuleList([nn.Conv2d(512,128,1,bias=False) for i in range(3)])
        self.conv3b2 = nn.ModuleList([nn.Conv2d(128,128,3,padding=1,bias=False) for i in range(3)])
        self.conv3b3 = nn.ModuleList([nn.Conv2d(128,512,1,bias=False) for i in range(3)])
        self.conv4b1 = nn.ModuleList([nn.Conv2d(1024,256,1,bias=False) for i in range(22)])
        self.conv4b2 = nn.ModuleList([nn.Conv2d(256,256,3,padding=1,bias=False) for i in range(22)])
        self.conv4b3 = nn.ModuleList([nn.Conv2d(256,1024,1,bias=False) for i in range(22)])
        self.conv5b1 = nn.ModuleList([nn.Conv2d(2048,512,1,bias=False) for i in range(2)])
        self.conv5b2 = nn.ModuleList([nn.Conv2d(512,512,3,padding=1,bias=False) for i in range(2)])
        self.conv5b3 = nn.ModuleList([nn.Conv2d(512,2048,1,bias=False) for i in range(2)])       
        self.pool=nn.AvgPool2d(2)
        self.fc1=nn.Linear(2048,512)
        self.fc2=nn.Linear(512,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x=self.conv1(x)
        x = F.relu(self.bn1(x), True)
        
        out=self.conv2a1(x)
        out = F.relu(self.bn1(out), True)
        out=self.conv2a2(out)
        out = F.relu(self.bn1(out), True)
        out=self.conv2a3(out)
        out =self.bn2(out)
        x=self.bn2(self.conv2a4(x))
        x=F.relu(x+out)

        for i in range(2):
            out=self.conv2b1[i](x)
            out = F.relu(self.bn1(out), True)
            out=self.conv2b2[i](out)
            out = F.relu(self.bn1(out), True)
            out=self.conv2b3[i](out)
            out =self.bn2(out)
            x=F.relu(x+out)

        out=self.conv3a1(x)
        out = F.relu(self.bn3(out), True)
        out=self.conv3a2(out)
        out = F.relu(self.bn3(out), True)
        out=self.conv3a3(out)
        out =self.bn4(out)
        x=self.bn4(self.conv3a4(x))
        x=F.relu(x+out)

        for i in range(3):
            out=self.conv3b1[i](x)
            out = F.relu(self.bn3(out), True)
            out=self.conv3b2[i](out)
            out = F.relu(self.bn3(out), True)
            out=self.conv3b3[i](out)
            out =self.bn4(out)
            x=F.relu(x+out)

        out=self.conv4a1(x)
        out = F.relu(self.bn2(out), True)
        out=self.conv4a2(out)
        out = F.relu(self.bn2(out), True)
        out=self.conv4a3(out)
        out =self.bn5(out)
        x=self.bn5(self.conv4a4(x))
        x=F.relu(x+out)

        for i in range(22):
            out=self.conv4b1[i](x)
            out = F.relu(self.bn2(out), True)
            out=self.conv4b2[i](out)
            out = F.relu(self.bn2(out), True)
            out=self.conv4b3[i](out)
            out =self.bn5(out)
            x=F.relu(x+out)

        out=self.conv5a1(x)
        out = F.relu(self.bn4(out), True)
        out=self.conv5a2(out)
        out = F.relu(self.bn4(out), True)
        out=self.conv5a3(out)
        out =self.bn6(out)
        x=self.bn6(self.conv5a4(x))
        x=F.relu(x+out)

        for i in range(2):
            out=self.conv5b1[i](x)
            out = F.relu(self.bn4(out), True)
            out=self.conv5b2[i](out)
            out = F.relu(self.bn4(out), True)
            out=self.conv5b3[i](out)
            out =self.bn6(out)
            x=F.relu(x+out)

        x=self.pool(x)
        x=x.view(-1,2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.to(device)


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for j in range(65):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs=outputs.to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
            #if i % 2000 == 1999: 
                #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                #running_loss = 0.0

    print('The %d th Finished Training'% (j+1))
    print('loss:%.3f'%(running_loss / 12000))



    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(4):
                label=labels[i]
                total+=1
                class_total[label]+=1
                if predicted[i]==label:
                    correct+=1
                    class_correct[label]+=1
    print('Accuracy of the network on the 10000 train images: %d %%' % (100 * correct / total))
    #for i in range(10):
        #print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(4):
                label=labels[i]
                total+=1
                class_total[label]+=1
                if predicted[i]==label:
                    correct+=1
                    class_correct[label]+=1
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    #for i in range(10):
        #print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print('\n')
