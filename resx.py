import torch    
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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
        self.conv1=nn.Conv2d(3,64,3,padding=1)
        self.conv2=nn.Conv2d(64,64,1,bias=False)
        self.conv3=nn.Conv2d(64,64,3,padding=1,bias=False)
        self.conv4=nn.Conv2d(64,256,1,bias=False)
        self.conv5=nn.Conv2d(64,256,1,bias=False)
        self.conv6=nn.Conv2d(256,64,1,bias=False)
        self.conv7=nn.Conv2d(64,64,3,padding=1,bias=False)
        self.conv8=nn.Conv2d(64,256,1,bias=False)
        self.conv9=nn.Conv2d(256,64,1,bias=False)
        self.conv10=nn.Conv2d(64,64,3,padding=1,bias=False)
        self.conv11=nn.Conv2d(64,256,1,bias=False)
        self.conv12=nn.Conv2d(256,128,1,stride=2,bias=False)
        self.conv13=nn.Conv2d(128,128,3,padding=1,bias=False)
        self.conv14=nn.Conv2d(128,512,1,bias=False)
        self.conv15=nn.Conv2d(256,512,1,stride=2,bias=False)
        self.conv16=nn.Conv2d(512,128,1,bias=False)
        self.conv17=nn.Conv2d(128,128,3,padding=1,bias=False)
        self.conv18=nn.Conv2d(128,512,1,bias=False)
        self.conv19=nn.Conv2d(512,128,1,bias=False)
        self.conv20=nn.Conv2d(128,128,3,padding=1,bias=False)
        self.conv21=nn.Conv2d(128,512,1,bias=False)
        self.conv22=nn.Conv2d(512,128,1,bias=False)
        self.conv23=nn.Conv2d(128,128,3,padding=1,bias=False)
        self.conv24=nn.Conv2d(128,512,1,bias=False)
        self.conv25=nn.Conv2d(512,256,1,stride=2,bias=False)
        self.conv26=nn.Conv2d(256,256,3,padding=1,bias=False)
        self.conv27=nn.Conv2d(256,1024,1,bias=False)
        self.conv28=nn.Conv2d(512,1024,1,stride=2,bias=False)
        self.conv29=nn.Conv2d(1024,256,1,bias=False)
        self.conv30=nn.Conv2d(256,256,3,padding=1,bias=False)
        self.conv31=nn.Conv2d(256,1024,1,bias=False)
        self.conv32=nn.Conv2d(1024,256,1,bias=False)
        self.conv33=nn.Conv2d(256,256,3,padding=1,bias=False)
        self.conv34=nn.Conv2d(256,1024,1,bias=False)
        self.conv35=nn.Conv2d(1024,256,1,bias=False)
        self.conv36=nn.Conv2d(256,256,3,padding=1,bias=False)
        self.conv37=nn.Conv2d(256,1024,1,bias=False)
        self.conv38=nn.Conv2d(1024,256,1,bias=False)
        self.conv39=nn.Conv2d(256,256,3,padding=1,bias=False)
        self.conv40=nn.Conv2d(256,1024,1,bias=False)
        self.conv41=nn.Conv2d(1024,256,1,bias=False)
        self.conv42=nn.Conv2d(256,256,3,padding=1,bias=False)
        self.conv43=nn.Conv2d(256,1024,1,bias=False)
        self.conv44=nn.Conv2d(1024,512,1,stride=2,bias=False)
        self.conv45=nn.Conv2d(512,512,3,padding=1,bias=False)
        self.conv46=nn.Conv2d(512,2048,1,bias=False)
        self.conv47=nn.Conv2d(1024,2048,1,stride=2,bias=False)
        self.conv48=nn.Conv2d(2048,512,1,bias=False)
        self.conv49=nn.Conv2d(512,512,3,padding=1,bias=False)
        self.conv50=nn.Conv2d(512,2048,1,bias=False)
        self.conv51=nn.Conv2d(2048,512,1,bias=False)
        self.conv52=nn.Conv2d(512,512,3,padding=1,bias=False)
        self.conv53=nn.Conv2d(512,2048,1,bias=False)
        self.pool1=nn.MaxPool2d(2,2)
        self.pool2=nn.AvgPool2d(2,2)
        self.fc1=nn.Linear(2048,512)
        self.fc2=nn.Linear(512,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x=self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x=self.pool1(x)

        out=self.conv2(x)
        out = F.relu(self.bn1(out), True)
        out=self.conv3(out)
        out = F.relu(self.bn1(out), True)
        out=self.conv4(out)
        out = self.bn2(out)
        x=self.bn2(self.conv5(x))
        x=F.relu(x+out)

        out=self.conv6(x)
        out = F.relu(self.bn1(out), True)
        out=self.conv7(out)
        out = F.relu(self.bn1(out), True)
        out=self.conv8(out)
        out = self.bn2(out)
        x=F.relu(x+out)

        out=self.conv9(x)
        out = F.relu(self.bn1(out), True)
        out=self.conv10(out)
        out = F.relu(self.bn1(out), True)
        out=self.conv11(out)
        out = self.bn2(out)
        x=F.relu(x+out)

        out=self.conv12(x)
        out = F.relu(self.bn3(out), True)
        out=self.conv13(out)
        out = F.relu(self.bn3(out), True)
        out=self.conv14(out)
        out = self.bn4(out)
        x=self.bn4(self.conv15(x))
        x=F.relu(x+out)

        out=self.conv16(x)
        out = F.relu(self.bn3(out), True)
        out=self.conv17(out)
        out = F.relu(self.bn3(out), True)
        out=self.conv18(out)
        out = self.bn4(out)
        x=F.relu(x+out)

        out=self.conv19(x)
        out = F.relu(self.bn3(out), True)
        out=self.conv20(out)
        out = F.relu(self.bn3(out), True)
        out=self.conv21(out)
        out = self.bn4(out)
        x=F.relu(x+out)

        out=self.conv22(x)
        out = F.relu(self.bn3(out), True)
        out=self.conv23(out)
        out = F.relu(self.bn3(out), True)
        out=self.conv24(out)
        out = self.bn4(out)
        x=F.relu(x+out)

        out=self.conv25(x)
        out = F.relu(self.bn2(out), True)
        out=self.conv26(out)
        out = F.relu(self.bn2(out), True)
        out=self.conv27(out)
        out = self.bn5(out)
        x=self.bn5(self.conv28(x))
        x=F.relu(x+out)

        out=self.conv29(x)
        out = F.relu(self.bn2(out), True)
        out=self.conv30(out)
        out = F.relu(self.bn2(out), True)
        out=self.conv31(out)
        out = self.bn5(out)
        x=F.relu(x+out)

        out=self.conv32(x)
        out = F.relu(self.bn2(out), True)
        out=self.conv33(out)
        out = F.relu(self.bn2(out), True)
        out=self.conv34(out)
        out = self.bn5(out)
        x=F.relu(x+out)

        out=self.conv35(x)
        out = F.relu(self.bn2(out), True)
        out=self.conv36(out)
        out = F.relu(self.bn2(out), True)
        out=self.conv37(out)
        out = self.bn5(out)
        x=F.relu(x+out)

        out=self.conv38(x)
        out = F.relu(self.bn2(out), True)
        out=self.conv39(out)
        out = F.relu(self.bn2(out), True)
        out=self.conv40(out)
        out = self.bn5(out)
        x=F.relu(x+out)

        out=self.conv41(x)
        out = F.relu(self.bn2(out), True)
        out=self.conv42(out)
        out = F.relu(self.bn2(out), True)
        out=self.conv43(out)
        out = self.bn5(out)
        x=F.relu(x+out)

        out=self.conv44(x)
        out = F.relu(self.bn4(out), True)
        out=self.conv45(out)
        out = F.relu(self.bn4(out), True)
        out=self.conv46(out)
        out = self.bn6(out)
        x=self.bn6(self.conv47(x))
        x=F.relu(x+out)

        out=self.conv48(x)
        out = F.relu(self.bn4(out), True)
        out=self.conv49(out)
        out = F.relu(self.bn4(out), True)
        out=self.conv50(out)
        out = self.bn6(out)
        x=F.relu(x+out)

        out=self.conv51(x)
        out = F.relu(self.bn4(out), True)
        out=self.conv52(out)
        out = F.relu(self.bn4(out), True)
        out=self.conv53(out)
        out = self.bn6(out)
        x=F.relu(x+out)

        x=self.pool2(x)
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

for j in range(130):
    #running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs=outputs.to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #running_loss += loss.item()
            #if i % 2000 == 1999: 
                #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                #running_loss = 0.0

    print('The %d th Finished Training'% (j+1))



    correct = 0
    total = 0
    #class_correct = list(0. for i in range(10))
    #class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(4):
                label=labels[i]
                total+=1
                #class_total[label]+=1
                if predicted[i]==label:
                    correct+=1
                    #class_correct[label]+=1
    print('Accuracy of the network on the 10000 train images: %d %%' % (100 * correct / total))
    #for i in range(10):
        #print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    #print('\n')
    correct = 0
    total = 0
    #class_correct = list(0. for i in range(10))
    #class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(4):
                label=labels[i]
                total+=1
                #class_total[label]+=1
                if predicted[i]==label:
                    correct+=1
                    #class_correct[label]+=1
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    #for i in range(10):
        #print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print('\n')


