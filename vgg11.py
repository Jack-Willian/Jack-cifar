# Code from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html



import torch    
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


print(device)



import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drop=nn.Dropout
        self.pool=nn.MaxPool2d(2)
        self.conv1=nn.Conv2d(3,64,3,padding=1)
        self.bn1= nn.BatchNorm2d(64)
        self.bn2= nn.BatchNorm2d(128)
        self.bn3= nn.BatchNorm2d(256)
        self.bn4= nn.BatchNorm2d(1024)
        self.conv2=nn.Conv2d(64,128,3,padding=1)
        self.conv3=nn.Conv2d(128,256,3,padding=1)
        self.conv4=nn.Conv2d(256,256,3,padding=1)
        self.conv5=nn.Conv2d(256,1024,3,padding=1)
        self.conv6=nn.Conv2d(1024,1024,3,padding=1)
        self.conv7=nn.Conv2d(1024,1024,3,padding=1)
        self.conv8=nn.Conv2d(1024,1024,3,padding=1)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 64)  
        self.fc3 = nn.Linear(64,10)  
    def forward(self, x):
        x=self.conv1(x)
        x = F.relu(self.bn1(x), True)
        self.drop()
        x=self.pool(x)
        x=self.conv2(x)
        x = F.relu(self.bn2(x), True)
        self.drop()
        x=self.pool(x)
        x=self.conv3(x)
        x = F.relu(self.bn3(x), True)
        self.drop()
        x=self.conv4(x)
        x = F.relu(self.bn3(x), True)
        self.drop()
        x=self.pool(x)
        x=self.conv5(x)
        x = F.relu(self.bn4(x), True)
        self.drop()
        x=self.conv6(x)
        x = F.relu(self.bn4(x), True)
        self.drop()
        x=self.pool(x)
        x=self.conv7(x)
        x = F.relu(self.bn4(x), True)
        self.drop()
        x=self.conv8(x)
        x = F.relu(self.bn4(x), True)
        self.drop()
        x=self.pool(x)
        x=x.view(-1,1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
net.to(device)


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for j in range(20):
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

    print('The %d th Finished Training'% j)



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
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
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
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print('\n')



