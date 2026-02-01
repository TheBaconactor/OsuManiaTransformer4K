# OsuManiaTransformer4K Roadmap: Hybrid Architecture

## The Four Foundations
This is our project anchor for what “good mapping” (and a good mapping model) must support:

1.  **The Ear (Audio Analysis):** Strong musical understanding (timing, rhythm, phrasing, emphasis, and when *not* to place notes).
2.  **The Eye (Review + Taste):** Ability to critique and justify style choices (“why this pattern here?”) and match patterns to mood/intent.
3.  **The Brain (Contextual Generation):** Long-range coherence and contextual reasoning across sections (transformer-style planning, motifs, density ramps).
4.  **The Hands (Playability + Iteration):** Ergonomics, readability, difficulty fairness, and an iterative playtest/feedback loop.

---

## 1. Strategy Overview: "The Hybrid Approach"
We separate the problem into two distinct tasks handled by specialized AI components.

> Training note: For AMD GPUs (RX 7900 XTX), we plan to do training in a dedicated “training site” (WSL2/Linux) and export ONNX back for Windows inference. See `docs/TRAINING_SITE.md`.

### **Stage 1: The Ear (Timing & Rhythm)**
-   **Goal:** Detect BPM, Offset, and Time Signature (The Red Lines/Timing Board).
-   **Engine:** `Beat This!` (Pre-trained SOTA model).
-   **Why:** Timing detection requires massive generalization across all music genres. `Beat This!` is already trained on standard datasets (GTZAN/Ballroom/etc.) and outperforms custom models trained on small datasets.
-   **Status:** ✅ **DONE (stable tempo)** / 🚧 **Needs work (live/rubato)**
    -   Sub-frame precision (1-2ms) implemented via Quadratic Interpolation.
    -   AMD GPU (DirectML) support added.
    -   Wrapper script `beat_to_osu.py` operational.
    -   Current gap on live piano (e.g., Rousseau): mapper redlines snap at ~<1ms p95, while the current automatic timing can be ~10–15ms p95 (needs adaptation).

#### Stage 1.1: Ear Calibration (osu-supervised, no human feedback)
Goal: fine-tune timing perception using *existing ranked maps* as supervision (their redlines define the beat grid).

- Build a Beat This training dataset from `data/datasets2/`:
    - `python training/scripts/build_beat_this_osu_dataset.py --val-artist Rousseau`
- Fine-tune from a pretrained Beat This checkpoint:
    - `python training/scripts/finetune_beat_this_osu.py --init-checkpoint final0 --tune heads`
- Use the resulting `.ckpt` during timing inference:
    - `python modules/beat_this/beat_to_osu.py <audio> --checkpoint <path_to_ckpt> --timing-profile live -o out.timing.txt`

### **Stage 2: The Brain (Mapping & Patterns)**
-   **Goal:** Place 4K HitObjects (Notes, Sliders, Rice) onto the timing grid.
-   **Engine:** Custom Transformer (OsuManiaTransformer4K).
-   **Why:** `Beat This!` doesn't understand "mapping." Our model will learn *patterning* (jacks, streams, LNs) from correct ranked map data.
-   **Status:** 🚧 **NEXT STEP**

---

## 2. Infrastructure & Data
### **Dataset 2 (High Quality)**
-   **Source:** Manual selection of high-quality Ranked/Loved maps (e.g., "The Last Page").
-   **Pipeline:**
    -   `osz_to_dataset.py`: Extracts Audio + JSON Annotations + Timing Points.
    -   **Refinement:** Run Stage 1 (`beat_to_osu.py`) on all audio to ensure we have a perfect "machine-readable" grid to train against, rather than relying solely on human timing which might vary.
-   **Status:** ✅ **Infrastructure Ready** (Processing "The Last Page" verified).

---

## 3. Development Plan (The Next Steps)

### **Phase A: Data Prep (Timing Injection)**
1.  [ ] **Batch Process Dataset 2:** Run `beat_to_osu.py` on all 6 songs in `data/datasets2/`.
2.  [ ] **Grid Alignment:** Create a training script that "snaps" the human HitObjects to our AI-generated Timing Points. This ensures the model learns to place notes *relative to the AI's perceived beat*.

### **Phase B: The Mapping Model (Architecture)**
1.  [ ] **Design "Mapper" Transformer:**
    -   **Input 1:** Audio Features (Mel Spectrogram + Onset Strengths).
    -   **Input 2:** Timing Grid (Phase, Tempo).
    -   **Output:** 4-channel classification (Column 0-3 activation) + Note Type (Tap vs Hold).
2.  [ ] **Data Loader:** Update `dataset_loader.py` to feed this aligned grid data.

### **Phase C: Training & Inference**
1.  [ ] **Training Loop:** Train on Dataset 2 (overfit test first to verify learning).
2.  [ ] **Inference Script:** `generate_map.py`
    -   Step 1: Run `beat_to_osu.py` -> Get Timing.
    -   Step 2: Run `Mapper Model` -> Get Notes.
    -   Step 3: Combine into final `.osu` file.
