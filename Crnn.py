import torch.nn as nn
import torch



class Convmodule(nn.Module):
    def __init__(self):
        super().__init__()
     
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) 
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) 
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        

        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,1), padding=(0,1)) 
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=1)
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=1)
        self.batchnorm2 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        
        self.pool6 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,1), padding=(0,1))
        
        self.conv7 = nn.Conv2d(512, 512, kernel_size=(2,2), stride=(1,1), padding=0)

    def forward(self, input):
        x = self.pool1(self.relu1(self.conv1(input)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        x = self.relu3(self.conv3(x))
        x = self.pool4(self.relu4(self.conv4(x))) 
        
        x = self.batchnorm1(self.conv5(x))
        x = self.relu5(x)
        
        x = self.batchnorm2(self.conv6(x))
        x = self.relu6(x)
        x = self.pool6(x) 
        
        x = self.conv7(x)
        return x


class CRNN(nn.Module):
    def __init__(self, num_labels, hidden_size):
        super(CRNN, self).__init__()
        
        self.cnn = Convmodule()
        
        self.rnn = nn.LSTM(
            input_size=512, 
            hidden_size=hidden_size, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True
        )
        

        self.fc = nn.Linear(hidden_size * 2, num_labels + 1) #numlables +1 for blank tokens

    def forward(self, x):
        features = self.cnn(x)  
        
        B, C, H, W = features.size() #=> [B,512,1,seq]
        
        features = features.squeeze(2) #=>[B,512,seq]
        
        features = features.permute(0,2,1) # Shape: [B, SeqLen, 512]
        rnn_out, _ = self.rnn(features) # Shape: [B, SeqLen, 512]
        
        output = self.fc(rnn_out) # Shape: [B, SeqLen, 82]
        
        return output
        

if __name__ == "__main__":
    model = Convmodule()
    input_tensor = torch.randn(1, 3, 32, 124)
    output = model(input_tensor)

    print(f"Output Size: {output.size()}")
