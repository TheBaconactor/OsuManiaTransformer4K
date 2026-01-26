# Future Plans: Sub-Frame Beat Interpolation

## Overview
The current `beat_to_osu.py` script uses raw beat timestamps from Beat This! at **50 FPS (20ms resolution)**. This document outlines a planned enhancement to achieve **1-2ms precision** using interpolation techniques.

## Problem
At high OD (Overall Difficulty 8-10), osu! timing windows can be as tight as **±16ms**. A 20ms grid may cause audible/playable misalignment on precise songs.

## Solution: Peak Interpolation
The model outputs a **probability curve**, not just binary "beat/no-beat". By fitting a curve to the peak, we can find the true beat position *between* frames.

### Quadratic Interpolation (Recommended)
```python
def refine_peak(probs, frame_idx, fps=50):
    """Refine beat position using parabolic interpolation."""
    if frame_idx == 0 or frame_idx >= len(probs) - 1:
        return frame_idx / fps
    
    y0, y1, y2 = probs[frame_idx - 1], probs[frame_idx], probs[frame_idx + 1]
    delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
    return (frame_idx + delta) / fps
```

### Expected Precision
| Method | Resolution |
|--------|------------|
| Raw Frame | 20ms |
| Quadratic Interpolation | ~2-5ms |
| Gaussian Fitting | ~1-2ms |

## Implementation Plan
1. Add `--refine` flag to `beat_to_osu.py`
2. Access raw probability logits from the model (requires modifying inference call)
3. Apply quadratic interpolation to each detected beat
4. Optionally upgrade to Gaussian fitting for maximum precision

## Status
- [ ] Not yet implemented
