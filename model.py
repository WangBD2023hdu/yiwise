import torch.nn as nn
import torch.nn.functional as F

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_heads=2, transformer_hidden=256, cnn_out_channels=128, lstm_hidden=128):
        super(TextClassificationModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=transformer_hidden)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden, num_layers=2, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(lstm_hidden * 2, num_classes) 
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.embedding(x)  #[batch_size, seq_len, embed_size]
        x = x.permute(1, 0, 2) 
        x = self.transformer_encoder(x)  #[seq_len, batch_size, embed_size]
        x = x.permute(1, 2, 0)  # [batch_size, embed_size, seq_len] for CNN
        x = F.relu(self.conv1(x))  # [batch_size, cnn_out_channels, seq_len]
        x = F.relu(self.conv2(x))  
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, cnn_out_channels] for LSTM
        x, (hn, cn) = self.lstm(x)  
    
        x = x[:, -1, :]  
        x = self.relu(self.fc(x))  
        
        return x
# 模型二
class ClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取 LSTM 输出的最后一个时间步
        x = self.fc(x)
        return F.log_softmax(x, dim=1)



