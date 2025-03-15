import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

class LipReadingDataset(Dataset):
    def __init__(self, metadata_df, preprocessed_dir, tokenizer, split='train'):
        self.metadata = metadata_df[metadata_df['split'] == split].reset_index(drop=True)
        self.preprocessed_dir = preprocessed_dir
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        # Get metadata
        row = self.metadata.iloc[idx]
        video_id = os.path.basename(row['video_path']).split('.')[0]
        text = row['text']
        
        # Load preprocessed mouth regions
        mouth_path = os.path.join(self.preprocessed_dir, f"{video_id}_mouth.npy")
        
        try:
            mouth_regions = np.load(mouth_path)
            
            # Convert to tensor and normalize
            video_tensor = torch.FloatTensor(mouth_regions) / 255.0
            video_tensor = video_tensor.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
            
            # Tokenize text
            tokens = torch.LongTensor(self.tokenizer.encode(text))
            
            return video_tensor, tokens
        except Exception as e:
            print(f"Error loading {mouth_path}: {e}")
            # Return a dummy sample
            dummy_video = torch.zeros((1, 3, CONFIG['mouth_height'], CONFIG['mouth_width']))
            dummy_text = torch.LongTensor([0])  # PAD token
            return dummy_video, dummy_text

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    # Filter out any bad samples
    batch = [item for item in batch if item[0].size(0) > 0 and item[1].size(0) > 0]
    
    if not batch:
        # Return empty tensors if batch is empty
        return (torch.tensor([]), torch.tensor([]), 
                torch.tensor([]), torch.tensor([]))
    
    # Sort by video length for packing
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    videos, texts = zip(*batch)
    
    # Get lengths
    video_lengths = torch.LongTensor([len(video) for video in videos])
    text_lengths = torch.LongTensor([len(text) for text in texts])
    
    # Pad videos
    max_video_len = max(video_lengths).item()
    padded_videos = []
    for video in videos:
        T, C, H, W = video.size(0), video.size(1), video.size(2), video.size(3)
        padding = torch.zeros((max_video_len - T, C, H, W))
        padded_video = torch.cat([video, padding], dim=0)
        padded_videos.append(padded_video)
    
    # Stack
    videos_tensor = torch.stack(padded_videos)
    texts_tensor = pad_sequence(texts, batch_first=True, padding_value=0)
    
    return videos_tensor, texts_tensor, text_lengths, video_lengths
