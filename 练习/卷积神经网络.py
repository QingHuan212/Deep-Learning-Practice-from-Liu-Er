import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. 硬件设备配置 (GPU/CPU)
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. 数据准备与预处理
# ==========================================
batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集的均值和标准差
])

# 下载并加载 MNIST 训练集与测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=4)

# ==========================================
# 3. 定义 CNN 网络模型
# ==========================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=2)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3)
        self.pooling1 = nn.MaxPool2d(2)
        self.pooling2 = nn.MaxPool2d(2)
        self.pooling3 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(120, 60)
        self.linear2 = nn.Linear(60, 30)
        self.linear3 = nn.Linear(30, 10)

    def forward(self, x):
        # 提取 Batch 维度
        batch_size = x.size(0)
        x = self.pooling1(F.relu(self.conv1(x)))
        x = self.pooling2(F.relu(self.conv2(x)))
        x = self.pooling3(F.relu(self.conv3(x)))
        x = x.view(batch_size,-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x =self.linear3(x)

        return x

model = Net()
model.to(device)

# ==========================================
# 4. 定义损失函数与优化器
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# ==========================================
# 5. 训练函数
# ==========================================
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        # 将数据和标签转移到 GPU 设备
        inputs, target = inputs.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播 + 反向传播 + 优化更新
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 300:.3f}')
            running_loss = 0.0

# ==========================================
# 6. 测试/评估函数
# ==========================================
def test():
    correct = 0
    total = 0
    # 测试阶段不计算梯度，节省内存和 GPU 计算资源
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            # 将测试数据和标签转移到 GPU 设备
            inputs, target = inputs.to(device), target.to(device)
            
            outputs = model(inputs)
            # 获取预测的类别索引 (沿维度 1 求最大值)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    print(f'Accuracy on test set: {100 * correct / total:.2f}% [{correct}/{total}]')

# ==========================================
# 7. 主程序循环
# ==========================================
if __name__ == '__main__':
    epochs = 15
    for epoch in range(epochs):
        train(epoch)
        test()