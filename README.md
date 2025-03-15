# lip-sync

## Folder Structure
```
lip_reading_project/
│
├── main.py                  # Main execution script
├── config.py                # Configuration settings
│
├── data/
│   ├── __init__.py
│   ├── dataset.py           # Dataset and DataLoader classes
│   └── tokenizer.py         # Character tokenizer implementation
│
├── model/
│   ├── __init__.py
│   ├── architecture.py      # LightweightLipReader model definition
│   └── training.py          # Training and validation functions
│
└── LRS2_landmarks/          # Extracted LRS2_landmarks
```


```plaintext
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
matplotlib>=3.7.0
mediapipe>=0.10.3
```


## Training Run:

Run the code `python main.py --data_dir /path/to/LRS2_landmarks --batch_size 8 --epochs 20` for training