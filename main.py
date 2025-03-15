# main.py
import os
import torch
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms

from data.dataset import LRS2Dataset
from data.tokenizer import CharacterTokenizer
from model.architecture import LightweightLipReader
from model.training import train_model, train_val_split
from config import CONFIG

def main():
    parser = argparse.ArgumentParser(description='Train a lip reading model on LRS2 dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to LRS2_landmarks directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--subset', type=str, default='train', choices=['pretrain', 'train'], help='Data subset to use')
    args = parser.parse_args()

    # Update config with command line arguments
    CONFIG.update({
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'checkpoints_dir': args.checkpoints_dir,
    })
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(CONFIG['checkpoints_dir'], exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Create dataset and tokenizer
    # First, create and fit tokenizer from the label files
    all_texts = []
    label_file = os.path.join(args.data_dir, f'{args.subset}_labels.txt')
    try:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    text = ' '.join(parts[1:])
                    all_texts.append(text)
    except FileNotFoundError:
        print(f"Label file {label_file} not found. Make sure to run preprocessing and merging first.")
        return
    
    # Create and fit tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.fit(all_texts)
    
    # Create dataset
    dataset = LRS2Dataset(
        root_dir=args.data_dir,
        subset=args.subset,
        transform=transform,
        tokenizer=tokenizer
    )
    
    # Split into train and validation
    train_dataset, val_dataset = train_val_split(dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = LightweightLipReader(vocab_size=tokenizer.vocab_size())
    model = model.to(device)
    
    # Print model summary
    print(model)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Train model
    history = train_model(model, train_loader, val_loader, device, CONFIG)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(CONFIG['checkpoints_dir'], 'lip_reading_final.pt'))
    print("Training completed!")

if __name__ == '__main__':
    main()