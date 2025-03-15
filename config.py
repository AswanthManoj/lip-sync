# Configuration settings
CONFIG = {
    'batch_size': 8,
    'epochs': 10,
    'learning_rate': 3e-4,
    'weight_decay': 1e-5,
    'hidden_size': 256,
    'feature_dim': 128,
    'mouth_height': 96,
    'mouth_width': 96,
    'max_frames': 100,
    'vocab_size': 40,  # Will be updated after tokenizer creation
    'preprocessed_dir': 'preprocessed_mouths',
    'checkpoints_dir': 'checkpoints',
    'raw_data_dir': 'raw_data/mvlrs_v1/main',
    'landmarks_dir': 'landmarks'
}

# MediaPipe mouth landmark indices
MOUTH_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]
