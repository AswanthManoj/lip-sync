# data/dataset.py
import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
import cv2

class LRS2Dataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None, max_seq_length=75, tokenizer=None):
        """
        Args:
            root_dir: Root directory of the preprocessed LRS2 dataset
            subset: 'pretrain', 'train', 'val', or 'test'
            transform: Optional transforms to apply to mouth ROIs
            max_seq_length: Maximum sequence length
            tokenizer: Character tokenizer for text processing
        """
        self.root_dir = root_dir
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        
        # Choose the appropriate folder based on subset
        if subset == 'pretrain':
            self.data_dir = os.path.join(root_dir, 'pretrain')
        else:
            self.data_dir = os.path.join(root_dir, 'main')
        
        # Load the label file (assuming you merged labels as instructed in preprocessing)
        if subset == 'pretrain':
            label_file = os.path.join(root_dir, f'pretrain_labels.txt')
        elif subset == 'train':
            label_file = os.path.join(root_dir, f'train_labels.txt')
        elif subset == 'val':
            label_file = os.path.join(root_dir, f'val_labels.txt')
        else:  # test
            label_file = os.path.join(root_dir, f'test_labels.txt')
        
        # Parse the label file
        self.samples = []
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        path = parts[0]  # Format: speakerID/clipID
                        text = ' '.join(parts[1:])
                        self.samples.append((path, text))
        except FileNotFoundError:
            raise ValueError(f"Label file {label_file} not found. Make sure to run preprocessing and merging first.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, text = self.samples[idx]
        speaker_id, clip_id = path.split('/')
        
        # Load the landmark pickle files
        pickle_dir = os.path.join(self.data_dir, speaker_id)
        pickle_path = os.path.join(pickle_dir, f"{clip_id}.pkl")
        
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
                
            # Landmarks format should contain mouth ROIs
            mouth_rois = data.get('mouth_rois', [])
            
            # Convert to tensor and normalize (assuming values 0-255)
            frames = []
            for roi in mouth_rois:
                # Convert BGR to RGB if needed
                if roi.shape[-1] == 3:
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Resize to fixed dimensions (96x96 is common for mouth ROIs)
                roi = cv2.resize(roi, (96, 96))
                
                # Normalize to 0-1
                roi = roi.astype(np.float32) / 255.0
                
                # Apply transforms if any
                if self.transform:
                    roi = self.transform(roi)
                
                frames.append(roi)
            
            # Convert to tensor
            frames = np.array(frames)
            frames = torch.FloatTensor(frames)
            
            # Add channel dimension if missing
            if frames.dim() == 3:  # [T, H, W]
                frames = frames.unsqueeze(1)  # [T, C, H, W]
            
            # Tokenize the text if tokenizer is provided
            if self.tokenizer:
                text_tokens = self.tokenizer.encode(text)
                text_tokens = torch.LongTensor(text_tokens)
                return frames, text_tokens, len(text_tokens), frames.size(0)
            else:
                return frames, text, None, frames.size(0)
                
        except Exception as e:
            print(f"Error loading {pickle_path}: {e}")
            # Return empty tensors in case of error
            return torch.zeros(1, 1, 96, 96), "", 0, 1
    
    def collate_fn(self, batch):
        """Custom collate function to handle variable length sequences"""
        # Sort by sequence length (descending) for packing
        batch.sort(key=lambda x: x[3], reverse=True)
        
        # Separate data
        frames, texts, text_lengths, frame_lengths = zip(*batch)
        
        # Determine max sequence length in this batch
        max_frame_len = max(frame_lengths)
        
        # Pad video frames
        padded_frames = []
        for i, frame_seq in enumerate(frames):
            # Padding for frames
            if frame_seq.size(0) < max_frame_len:
                padding = torch.zeros(max_frame_len - frame_seq.size(0), 
                                     frame_seq.size(1), 
                                     frame_seq.size(2), 
                                     frame_seq.size(3))
                padded_frames.append(torch.cat([frame_seq, padding], dim=0))
            else:
                padded_frames.append(frame_seq)
        
        # Stack tensors
        padded_frames = torch.stack(padded_frames)
        
        # If tokenized text is available
        if isinstance(texts[0], torch.Tensor):
            # Pad text sequences
            max_text_len = max(text_lengths)
            padded_texts = []
            for i, text_seq in enumerate(texts):
                if text_seq.size(0) < max_text_len:
                    padding = torch.zeros(max_text_len - text_seq.size(0), dtype=torch.long)
                    padded_texts.append(torch.cat([text_seq, padding], dim=0))
                else:
                    padded_texts.append(text_seq)
            
            padded_texts = torch.stack(padded_texts)
            text_lengths = torch.LongTensor(text_lengths)
        else:
            # For raw text strings
            padded_texts = texts
            text_lengths = None
        
        frame_lengths = torch.LongTensor(frame_lengths)
        
        return padded_frames, padded_texts, text_lengths, frame_lengths
    