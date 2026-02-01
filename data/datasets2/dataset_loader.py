"""
Dataset 2 Loader: Interface for Training

This file provides a PyTorch-compatible Dataset class for loading the
high-quality osu!mania 4K data processed by osz_to_dataset.py.
"""

import json
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset

class OsuManiaDataset2(Dataset):
    def __init__(self, dataset_dir, sample_rate=22050, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.annotations_dir = self.dataset_dir / "annotations"
        self.audio_dir = self.dataset_dir / "audio"
        self.sample_rate = sample_rate
        self.transform = transform
        
        self.samples = sorted(list(self.annotations_dir.glob("*.json")))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        annotation_path = self.samples[idx]
        
        # Load annotation
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Load audio
        audio_path = self.audio_dir / data['audio_file']
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Prepare labels (HitObjects)
        # In a real training scenario, you'd convert these to a time-series mask
        hit_objects = data['hit_objects']
        
        return {
            'waveform': waveform,
            'metadata': {
                'title': data['title'],
                'version': data['version'],
                'song_id': data['song_id']
            },
            'hit_objects': hit_objects,
            'timing_points': data['timing_points']
        }

if __name__ == "__main__":
    # Quick verification
    dataset = OsuManiaDataset2(Path(__file__).resolve().parent)
    print(f"Dataset 2 initialized with {len(dataset)} samples.")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Verified sample: {sample['metadata']['title']} [{sample['metadata']['version']}]")
        print(f"Waveform shape: {sample['waveform'].shape}")
        print(f"Detected {len(sample['hit_objects'])} notes.")
