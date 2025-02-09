import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import copy
import os
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.ReLU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
        
    def forward(self, x):
        x = x.permute(1, 0, 2)
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.norm(x + attn_out)
        ffn_out = self.ffn(x)
        return ffn_out.permute(1, 0, 2)

class RegNetWithAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.regnet_y_16gf(pretrained=True)
        self.backbone.fc = nn.Identity()  # in_features kaldırıldı
        self.attention = MultiHeadAttention()
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.unsqueeze(1)
        attn_features = self.attention(features)
        return self.classifier(attn_features.squeeze(1))

def get_data_loaders(data_dir, batch_size=32):
    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    datasets = {
        phase: datasets.ImageFolder(
            os.path.join(data_dir, phase), 
            transform[phase]
        )
        for phase in ['train', 'val', 'test']
    }
    
    loaders = {
        phase: DataLoader(
            datasets[phase],
            batch_size=batch_size,
            shuffle=(phase == 'train'),
            num_workers=4
        )
        for phase in ['train', 'val', 'test']
    }
    
    return loaders, datasets

def get_model(num_classes):
    model = RegNetWithAttention(num_classes=num_classes)
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, data_dir, num_epochs=25):
    loaders, _ = get_data_loaders(data_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in loaders[phase]:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(loaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()
    
    model.load_state_dict(best_weights)
    return model

def evaluate_model(model, data_dir):
    loaders, datasets = get_data_loaders(data_dir)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loaders['test']:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    correct_predictions = sum(p == label for p, label in zip(all_preds, all_labels))
    test_acc = correct_predictions / len(all_labels)
    
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'F1 Score: {f1_score(all_labels, all_preds, average="weighted"):.4f}')
    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=datasets['test'].classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()



def main():
    parser = argparse.ArgumentParser(description='BI-RADS Classification Training')
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='Path to dataset directory (should contain train/val/test subfolders)')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Input batch size for training')
    parser.add_argument('--eval_only', action='store_true',
                       help='Skip training and only evaluate pre-trained model')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='Path to pre-trained model weights')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not args.eval_only:
        # Training and validation
        model = get_model(args.data_dir)
        model = train_model(model, args.data_dir, args.epochs)
    else:
        # Evaluation only
        model = RegNetWithAttention(num_classes=len(datasets['test'].classes))  # Update class count
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(device)

    # Evaluation on test set
    evaluate_model(model, args.data_dir)

if __name__ == '__main__':
    main()
    

# örnek eğitim
# python script.py --data_dir ./preprocessed --epochs 30 --batch_size 64

# örnek test
# python script.py --data_dir ./preprocessed --eval_only --model_path best_model.pth
