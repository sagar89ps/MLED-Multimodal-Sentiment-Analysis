import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
import cv2
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from typing import Dict, List, Tuple

class MELDDataPreprocessor:
    """Handles data preprocessing and balancing for the MELD dataset"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        
    def load_meld_data(self, data_path: str) -> pd.DataFrame:
        """Load MELD dataset and perform initial preprocessing"""
        data_path="dataset\dev_sent_emo.csv"
        df = pd.read_csv(data_path)
        df['sentiment_encoded'] = self.label_encoder.fit_transform(df['sentiment'])
        return df
    
    def random_undersample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform random undersampling to balance the dataset"""
        class_counts = Counter(df['sentiment_encoded'])
        min_class_count = min(class_counts.values())
        
        balanced_dfs = []
        for class_label in class_counts.keys():
            class_df = df[df['sentiment_encoded'] == class_label]
            if len(class_df) > min_class_count:
                class_df = class_df.sample(n=min_class_count, 
                                         random_state=self.random_state)
            balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs, axis=0).reset_index(drop=True)

class MELDFeatureExtractor:
    """Extracts features from multimodal MELD data"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
    def extract_audio_features(self, audio_path: str) -> np.ndarray:
        """Extract MFCC and other audio features"""
        y, sr = librosa.load(audio_path)
        features = []
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.append(mfcc)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(spectral_centroids)
        
        # Chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(chroma)
        
        return np.concatenate(features, axis=0)
    
    def extract_video_features(self, video_path: str) -> np.ndarray:
        """Extract visual features including facial expressions"""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        cap = cv2.VideoCapture(video_path)
        frames_features = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                # Extract face embeddings or features here
                # For now, we'll use basic statistics of the face region
                face_features = cv2.resize(face_roi, (64, 64)).flatten()
                frames_features.append(face_features)
        
        cap.release()
        return np.array(frames_features)

class MultimodalTransformer(nn.Module):
    """Enhanced Transformer model for multimodal sentiment analysis"""
    
    def __init__(self, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        
        # Audio encoder with attention
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(20, 64, kernel_size=3),  # Increased input channels for more features
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Video encoder with 3D convolutions
        self.video_encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Dropout(dropout_rate),
            nn.Conv3d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            dropout=dropout_rate
        )
        
        # Final classification
        self.classifier = nn.Sequential(
            nn.Linear(768 + 256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, text_ids, text_mask, audio_features, video_features):
        # Process text
        text_output = self.text_encoder(text_ids, attention_mask=text_mask)
        text_embeddings = text_output.last_hidden_state[:, 0, :]
        
        # Process audio and video
        audio_embeddings = self.audio_encoder(audio_features)
        video_embeddings = self.video_encoder(video_features)
        
        # Cross-modal attention between audio and video
        av_features, _ = self.cross_attention(
            audio_embeddings, 
            video_embeddings, 
            video_embeddings
        )
        
        # Concatenate all features
        combined = torch.cat([text_embeddings, av_features], dim=1)
        
        return self.classifier(combined)

class SentimentTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Use weighted cross entropy for class imbalance
        class_weights = torch.FloatTensor([1.0, 1.0, 1.0]).to(device)  # Adjust based on class distribution
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            # Move batch to device
            text_ids = batch['text_ids'].to(self.device)
            text_mask = batch['text_mask'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            video_features = batch['video_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(text_ids, text_mask, audio_features, video_features)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                text_ids = batch['text_ids'].to(self.device)
                text_mask = batch['text_mask'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                video_features = batch['video_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(text_ids, text_mask, audio_features, video_features)
                loss = self.criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += loss.item()
        
        return {
            'accuracy': correct / total,
            'val_loss': val_loss / len(val_loader)
        }