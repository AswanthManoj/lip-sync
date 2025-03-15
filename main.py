import os
import sys
import torch
from torch.utils.data import DataLoader

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from config import CONFIG
from data.preprocessing import format_dataset, extract_mouth_regions
from data.tokenizer import CharacterTokenizer
from data.dataset import LipReadingDataset, collate_fn
from model.architecture import LightweightLipReader
from model.training import train_model

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(CONFIG['preprocessed_dir'], exist_ok=True)
    os.makedirs(CONFIG['checkpoints_dir'], exist_ok=True)
    
    print("Step 1: Preparing dataset metadata...")
    metadata_df = format_dataset()
    
    print("Step 2: Creating tokenizer...")
    tokenizer = CharacterTokenizer(metadata_df['text'].tolist())
    CONFIG['vocab_size'] = tokenizer.vocab_size()
    
    print("Step 3: Extracting mouth regions...")
    extract_mouth_regions()
    
    print("Step 4: Creating datasets...")
    train_dataset = LipReadingDataset(metadata_df, CONFIG['preprocessed_dir'], tokenizer, 'train')
    val_dataset = LipReadingDataset(metadata_df, CONFIG['preprocessed_dir'], tokenizer, 'val')
    
    print("Step 5: Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    print("Step 6: Creating model...")
    model = LightweightLipReader(CONFIG['vocab_size']).to(device)
    
    print("Step 7: Training model...")
    history = train_model(model, train_loader, val_loader, device, CONFIG)
    
    print("Training complete!")

if __name__ == "__main__":
    main()