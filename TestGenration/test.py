import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt


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
        self.fc = nn.Linear(hidden_size * 2, num_labels + 1) 

    def forward(self, x):
        features = self.cnn(x)  
        features = features.squeeze(2) 
        features = features.permute(0,2,1) 
        rnn_out, _ = self.rnn(features) 
        output = self.fc(rnn_out) 
        return output

class LabelEncoderDecoder():
    def __init__(self, vocab):
        self.vocab = vocab
        self.char2int = {char: i + 1 for i, char in enumerate(vocab)}
        self.int2char = {i + 1: char for i, char in enumerate(vocab)}
        self.blank_idx = 0

    def decode(self, preds):
        pred_indices = torch.argmax(preds, dim=2) 
        
        decoded_strings = []
        for sequence in pred_indices:
            decoded_chars = []
            prev_char = -1
            for idx in sequence:
                idx = idx.item()
                # CTC Logic: Merge repeated characters and ignore blanks (0)
                if idx != self.blank_idx and idx != prev_char:
                    decoded_chars.append(self.int2char[idx])
                prev_char = idx
            decoded_strings.append("".join(decoded_chars))
        return decoded_strings


#Got this from notebook
VOCAB = "01289ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

MODEL_PATH = "/home/saurav/Desktop/PrecorgTask/models/Task2Genration.pth" #model path
DATASET_ROOT = "test_dataset"  #test dataset 

model = CRNN(len(VOCAB), 256)
weights = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(weights)



model.eval()
encoder = LabelEncoderDecoder(VOCAB)

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



subsets = ["easy", "hard", "bonus"]

fig = plt.figure(figsize=(20, 10))
plt.subplots_adjust(hspace=0.6, wspace=0.3)

plot_idx = 1

for row_idx, subset in enumerate(subsets):
    subset_path = os.path.join(DATASET_ROOT, subset)
    csv_path = os.path.join(subset_path, "labels.csv")
    
        
    #each subset has its own label.csv file
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        
    data_pairs = []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            data_pairs.append((parts[0], parts[1])) # (filename, label)
 
        
    print(f"\nProcessing {subset.upper()} set ({len(data_pairs)} images)...")

    # Limit to 10 images per subset for plotting
    data_pairs = data_pairs[:10]
    
    for i, (filename, true_label) in enumerate(data_pairs):
        img_path = os.path.join(subset_path, filename)
        

            # 1. Load and Preprocess Image
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0) # [1, 3, 32, 128] adds batch dimension
            
            # 2. Inference
        with torch.no_grad():
            preds = model(img_tensor) # [1, SeqLen, NumClasses]
            
            # 3. Decode
        predicted_text = encoder.decode(preds)[0]
            
            # 4. Plotting
        ax = plt.subplot(3, 10, plot_idx)
        ax.imshow(image)
            
        #Green if correct, Red if wrong
        text_color = 'green' if predicted_text == true_label else 'red'
            
        title_text = f"Pred: {predicted_text}\nTrue: {true_label}"
        ax.set_title(title_text, color=text_color, fontsize=9)
        ax.axis('off')
            
        plot_idx += 1


plt.suptitle("Model Predictions on Test Dataset (Green=Correct, Red=Wrong)", fontsize=16)
plt.show()