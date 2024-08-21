import torch
import torch.nn as nn
import torch.optim as optim
from dataset import TextDataset
from torch.utils.data import DataLoader
import pandas as pd

from model import TextClassificationModel
from utils import collate_fn, set_seed

from tqdm import tqdm
from sklearn.metrics import f1_score
import argparse

set_seed(42)
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    all_predictions = []
    all_labels = []
    total_losss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_losss = loss.item() + total_losss
        
        _, predicted = torch.max(outputs, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    f1_macro = f1_score(all_predictions, all_labels, average='macro')
    f1_micro = f1_score(all_predictions, all_labels, average='micro')
    
    return f1_macro, f1_micro, total_losss

def evaluate(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1_macro = f1_score(all_predictions, all_labels, average='macro')
    f1_micro = f1_score(all_predictions, all_labels, average='micro')
    
    return f1_macro, f1_micro


def _main():
    # 创建数据加载器, 模型等基本配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = TextDataset("./data/train.tsv", mode="train")
    vocab_size = len(train_dataset.vocab)

    val_dataset = TextDataset("./data/val.tsv")
    test_datataset = TextDataset("./data/test.tsv")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_datataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = TextClassificationModel(vocab_size=vocab_size, embedding_dim=300, num_classes=9)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 训练部分
    best_f1_macro = 0
    best_f1_micro = 0
    num_epochs = 100
    best_f1_macro_model_path = 'best_f1_macro_model.pth'
    best_f1_micro_model_path = 'best_f1_micro_model.pth'

    train_f1_macro_all = []
    train_f1_micro_all = []
    val_f1_macro_all = []
    val_f1_micro_all = []
    train_loss = []
    test_f1_micro = 0
    test_f1_macro = 0
    for epoch in range(num_epochs):
        train_f1_macro, train_f1_micro, epoch_loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch+1}/{num_epochs} - Train F1-macro: {train_f1_macro:.4f}, Train F1-micro: {train_f1_micro:.4f}')
        #保存训练过程指标
        train_f1_macro_all.append(train_f1_macro)
        train_f1_micro_all.append(train_f1_micro)
        train_loss.append(epoch_loss)
        
        val_f1_macro, val_f1_micro = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{num_epochs} - Val F1-macro: {val_f1_macro:.4f}, Val F1-micro: {val_f1_micro:.4f}')
        val_f1_macro_all.append(val_f1_macro)
        val_f1_micro_all.append(val_f1_micro)
        
        if val_f1_macro > best_f1_macro:
            best_f1_macro = val_f1_macro
            test_f1_macro, test_f1_micro = evaluate(model, test_loader, device)
            print(f'Epoch {epoch+1}/{num_epochs} - Test F1-macro: {test_f1_macro:.4f}, Test F1-micro: {test_f1_micro:.4f}')
            torch.save(model.state_dict(), best_f1_macro_model_path)
            
        if val_f1_micro > best_f1_micro:
            best_f1_micro = val_f1_micro
            test_f1_macro, test_f1_micro = evaluate(model, test_loader, device)
            print(f'Epoch {epoch+1}/{num_epochs} - Test F1-macro: {test_f1_macro:.4f}, Test F1-micro: {test_f1_micro:.4f}')
            torch.save(model.state_dict(), best_f1_micro_model_path)
            
        print(f'Epoch {epoch+1}/{num_epochs} - Best F1-macro on Test: {test_f1_macro:.4f}, Best F1-micro on Test: {test_f1_micro:.4f}')
    pd.DataFrame({'train_f1_macro': train_f1_macro_all, 'train_f1_micro': train_f1_micro_all, 'train_loss': train_loss, "val_f1_macro": val_f1_macro_all, "val_f1_micro": val_f1_micro_all}).to_csv('train_result.csv', index=False)


if __name__ == "__main__":
    _main()