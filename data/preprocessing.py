import os
import cv2
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG, MOUTH_INDICES

def format_dataset():
    """Create metadata DataFrame from LRS2 dataset"""
    # Define paths
    lrs2_video_dir = CONFIG['raw_data_dir']
    landmarks_dir = CONFIG['landmarks_dir']
    
    # Create metadata list
    metadata = []
    
    # Process LRS2 directory structure
    splits = ['pretrain', 'train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(lrs2_video_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        # Get text files with transcriptions
        txt_files = []
        for root, _, files in os.walk(split_dir):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        
        for txt_path in tqdm(txt_files, desc=f'Processing {split}'):
            # Get video path from text path
            video_path = txt_path.replace('.txt', '.mp4')
            if not os.path.exists(video_path):
                continue
                
            # Get video ID for finding landmarks
            video_id = os.path.relpath(video_path, lrs2_video_dir).replace('/', '_').replace('.mp4', '')
            
            # Check if landmarks exist
            landmark_path = os.path.join(landmarks_dir, f"{video_id}.pkl")
            if not os.path.exists(landmark_path):
                continue
                
            # Read the transcript
            with open(txt_path, 'r') as f:
                text = f.read().strip()
            
            # Add to metadata
            metadata.append({
                'video_path': video_path,
                'landmark_path': landmark_path,
                'text': text,
                'split': 'train' if split == 'pretrain' else split  # Combine pretrain with train
            })
    
    # Convert to DataFrame
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv('lrs2_metadata.csv', index=False)
    print(f"Created metadata with {len(metadata_df)} entries")
    
    return metadata_df

def extract_mouth_regions():
    """Extract mouth regions using landmarks"""
    metadata_df = pd.read_csv('lrs2_metadata.csv')
    
    # Create output directory
    os.makedirs(CONFIG['preprocessed_dir'], exist_ok=True)
    
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Extracting mouths"):
        video_id = os.path.basename(row['video_path']).split('.')[0]
        video_path = row['video_path']
        landmark_path = row['landmark_path']
        
        # Output path for mouth regions
        output_path = os.path.join(CONFIG['preprocessed_dir'], f"{video_id}_mouth.npy")
        
        # Skip if already processed
        if os.path.exists(output_path):
            continue
            
        try:
            # Load video
            cap = cv2.VideoCapture(video_path)
            frames = []
            success = True
            while success and len(frames) < CONFIG['max_frames']:
                success, frame = cap.read()
                if success:
                    frames.append(frame)
            cap.release()
            
            # Load landmarks
            with open(landmark_path, 'rb') as f:
                landmarks = pickle.load(f)
            
            # Process each frame to extract mouth region
            mouth_regions = []
            for i, frame in enumerate(frames):
                if i >= len(landmarks):
                    break
                
                # Get mouth landmarks for this frame
                mouth_points = np.array([landmarks[i][point] for point in MOUTH_INDICES])
                
                # Calculate bounding box with padding
                x_min = np.min(mouth_points[:, 0])
                y_min = np.min(mouth_points[:, 1])
                x_max = np.max(mouth_points[:, 0])
                y_max = np.max(mouth_points[:, 1])
                
                # Add padding
                padding = int(max(x_max - x_min, y_max - y_min) * 0.2)
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(frame.shape[1], x_max + padding)
                y_max = min(frame.shape[0], y_max + padding)
                
                # Crop and resize
                mouth = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                mouth = cv2.resize(mouth, (CONFIG['mouth_width'], CONFIG['mouth_height']))
                
                mouth_regions.append(mouth)
            
            # Save as numpy array
            if mouth_regions:
                np.save(output_path, np.array(mouth_regions))
        
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
