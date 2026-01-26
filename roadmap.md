# OsuManiaTransformer4K Roadmap: Hybrid Architecture

## 1. Strategy Overview: "The Hybrid Approach"
We separate the problem into two distinct tasks handled by specialized AI components.

### **Stage 1: The Ear (Timing & Rhythm)**
-   **Goal:** Detect BPM, Offset, and Time Signature (The Red Lines/Timing Board).
-   **Engine:** `Beat This!` (Pre-trained SOTA model).
-   **Why:** Timing detection requires massive generalization across all music genres. `Beat This!` is already trained on standard datasets (GTZAN/Ballroom/etc.) and outperforms custom models trained on small datasets.
-   **Status:** ✅ **DONE**
    -   Sub-frame precision (1-2ms) implemented via Quadratic Interpolation.
    -   AMD GPU (DirectML) support added.
    -   Wrapper script `beat_to_osu.py` operational.

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
1.  [ ] **Batch Process Dataset 2:** Run `beat_to_osu.py` on all 6 songs in `Datasets2/`.
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
