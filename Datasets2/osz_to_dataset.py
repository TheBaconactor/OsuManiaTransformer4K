r"""
Dataset 2 Processor: OSZ to Training Data

This script processes .osz files from the osu! Exports folder and organizes them
into a structured dataset for training the OsuManiaTransformer4K model.

Input: .osz files in C:\Users\troll\AppData\Local\osu!\Exports
Output: Organized data in Datasets2/ with audio, annotations, and metadata

Usage:
    python osz_to_dataset.py                     # Process all .osz files
    python osz_to_dataset.py --file song.osz     # Process single file
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Optional


# Defaults (override via CLI flags)
DEFAULT_OSZ_INPUT_DIR = Path(r"C:/Users/troll/AppData/Local/osu!/Exports")
DEFAULT_DATASET_OUTPUT_DIR = Path(r"c:/Users/troll/Desktop/OsuManiaTransformer4K/Datasets2")


def parse_osu_file(content: str) -> dict:
    """
    Parse an .osu file and extract relevant sections.
    
    Returns dict with: general, metadata, difficulty, timing_points, hit_objects
    """
    sections = {}
    current_section = None
    current_lines = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('[') and line.endswith(']'):
            if current_section:
                sections[current_section] = current_lines
            current_section = line[1:-1]
            current_lines = []
        else:
            current_lines.append(line)
    
    if current_section:
        sections[current_section] = current_lines
    
    result = {
        'general': {},
        'metadata': {},
        'difficulty': {},
        'timing_points': [],
        'hit_objects': []
    }
    
    # Parse General section
    for line in sections.get('General', []):
        if ':' in line:
            key, value = line.split(':', 1)
            result['general'][key.strip()] = value.strip()
    
    # Parse Metadata section
    for line in sections.get('Metadata', []):
        if ':' in line:
            key, value = line.split(':', 1)
            result['metadata'][key.strip()] = value.strip()
    
    # Parse Difficulty section
    for line in sections.get('Difficulty', []):
        if ':' in line:
            key, value = line.split(':', 1)
            result['difficulty'][key.strip()] = value.strip()
    
    # Parse TimingPoints
    for line in sections.get('TimingPoints', []):
        parts = line.split(',')
        if len(parts) >= 8:
            result['timing_points'].append({
                'offset': float(parts[0]),
                'beat_length': float(parts[1]),  # ms per beat (negative = inherited)
                'meter': int(parts[2]),
                'sample_set': int(parts[3]),
                'sample_index': int(parts[4]),
                'volume': int(parts[5]),
                'uninherited': int(parts[6]) == 1,
                'effects': int(parts[7])
            })
    
    # Parse HitObjects (Mania format)
    for line in sections.get('HitObjects', []):
        parts = line.split(',')
        if len(parts) >= 5:
            x = int(parts[0])
            time = int(parts[2])
            obj_type = int(parts[3])
            
            # Convert x position to column (4K Mania)
            # 0-127 = col 0, 128-255 = col 1, 256-383 = col 2, 384-511 = col 3
            column = x // 128
            
            # Check if it's a hold note (type & 128)
            is_hold = (obj_type & 128) != 0
            end_time = None
            
            if is_hold and len(parts) >= 6:
                # End time is in the last part before the colon-separated extras
                end_part = parts[5].split(':')[0]
                try:
                    end_time = int(end_part)
                except ValueError:
                    end_time = time
            
            result['hit_objects'].append({
                'time': time,
                'column': column,
                'is_hold': is_hold,
                'end_time': end_time
            })
    
    return result


def is_mania_4k(parsed: dict) -> bool:
    """Check if the beatmap is osu!mania 4K."""
    mode = parsed['general'].get('Mode', '0')
    circle_size = parsed['difficulty'].get('CircleSize', '4')
    return mode == '3' and circle_size == '4'


def process_osz(osz_path: Path, output_dir: Path, overwrite: bool = False) -> Optional[dict]:
    """
    Process a single .osz file and extract its contents.
    
    Returns metadata dict on success, None on skip/failure.
    """
    print(f"[INFO] Processing: {osz_path.name}")
    
    try:
        with zipfile.ZipFile(osz_path, 'r') as zf:
            # Find all .osu files
            osu_files = [f for f in zf.namelist() if f.endswith('.osu')]
            
            if not osu_files:
                print(f"  [WARN] No .osu files found, skipping")
                return None
            
            # Find audio files for each difficulty
            difficulties = []
            for osu_file in osu_files:
                content = zf.read(osu_file).decode('utf-8', errors='ignore')
                parsed = parse_osu_file(content)
                
                if not is_mania_4k(parsed):
                    print(f"  [SKIP] {osu_file} is not 4K Mania")
                    continue
                
                audio_name = parsed['general'].get('AudioFilename')
                if not audio_name:
                    print(f"  [WARN] No AudioFilename in {osu_file}, skipping")
                    continue
                
                difficulties.append((osu_file, parsed, audio_name))
            
            if not difficulties:
                print(f"  [WARN] No valid 4K Mania difficulties found")
                return None
            
            # Process each difficulty
            processed_count = 0
            already_exists_count = 0
            song_metadata = None
            
            for osu_file, parsed, audio_name in difficulties:
                # Find the actual case-insensitive file in the archive
                actual_audio_file = None
                for name in zf.namelist():
                    if name.lower() == audio_name.lower():
                        actual_audio_file = name
                        break
                
                if not actual_audio_file:
                    print(f"  [WARN] Audio file {audio_name} not found in zip. Files: {zf.namelist()}")
                    continue
                
                print(f"  [DEBUG] Extracting {actual_audio_file} for {osu_file}")
                
                # Generate unique ID
                beatmap_set_id = parsed['metadata'].get('BeatmapSetID', 'unknown')
                beatmap_id = parsed['metadata'].get('BeatmapID', 'unknown')
                version = parsed['metadata'].get('Version', 'Unknown')
                safe_version = "".join(c for c in version if c.isalnum() or c in ' _-').strip()
                
                song_id = f"{beatmap_set_id}_{beatmap_id}_{safe_version}"
                song_id = song_id.replace(' ', '_')
                
                # Create output paths
                audio_out = output_dir / 'audio' / f"{song_id}.mp3"
                annotation_out = output_dir / 'annotations' / f"{song_id}.json"
                
                if audio_out.exists() and annotation_out.exists() and not overwrite:
                    print(f"  [SKIP] {song_id} already exists")
                    already_exists_count += 1
                    if song_metadata is None:
                        song_metadata = {
                            'title': parsed['metadata'].get('Title', 'Unknown'),
                            'artist': parsed['metadata'].get('Artist', 'Unknown'),
                            'beatmap_set_id': beatmap_set_id
                        }
                    continue
                
                # Extract audio (only once per song)
                if not audio_out.exists() or overwrite:
                    audio_out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(actual_audio_file) as src, open(audio_out, 'wb') as dst:
                        dst.write(src.read())
                
                # Create annotation file
                annotation_data = {
                    'song_id': song_id,
                    'title': parsed['metadata'].get('Title', 'Unknown'),
                    'artist': parsed['metadata'].get('Artist', 'Unknown'),
                    'creator': parsed['metadata'].get('Creator', 'Unknown'),
                    'version': version,
                    'beatmap_set_id': beatmap_set_id,
                    'beatmap_id': beatmap_id,
                    'difficulty': {
                        'hp': float(parsed['difficulty'].get('HPDrainRate', 5)),
                        'od': float(parsed['difficulty'].get('OverallDifficulty', 5)),
                    },
                    'timing_points': parsed['timing_points'],
                    'hit_objects': parsed['hit_objects'],
                    'audio_file': audio_out.name
                }
                
                annotation_out.parent.mkdir(parents=True, exist_ok=True)
                with open(annotation_out, 'w', encoding='utf-8') as f:
                    json.dump(annotation_data, f, indent=2, ensure_ascii=False)
                
                print(f"  [OK] Extracted: {song_id}")
                processed_count += 1
                
                if song_metadata is None:
                    song_metadata = {
                        'title': parsed['metadata'].get('Title', 'Unknown'),
                        'artist': parsed['metadata'].get('Artist', 'Unknown'),
                        'beatmap_set_id': beatmap_set_id
                    }
            
            if processed_count == 0 and already_exists_count == 0:
                print(f"  [WARN] No 4K Mania difficulties found")
                return None
            
            if processed_count == 0 and already_exists_count > 0:
                print(f"  [DONE] All {already_exists_count} difficulties already extracted")
            else:
                print(f"  [DONE] Processed {processed_count} difficulties")
            return song_metadata
            
    except zipfile.BadZipFile:
        print(f"  [ERROR] Invalid .osz file")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Process .osz files into Dataset 2")
    parser.add_argument('--file', type=str, help="Process a single .osz file")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing files")
    parser.add_argument('--input-dir', type=str, default=str(DEFAULT_OSZ_INPUT_DIR), help="Directory containing .osz files")
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_DATASET_OUTPUT_DIR), help="Output dataset directory")
    args = parser.parse_args()

    osz_input_dir = Path(args.input_dir)
    dataset_output_dir = Path(args.output_dir)
    
    # Ensure output directories exist
    (dataset_output_dir / 'audio').mkdir(parents=True, exist_ok=True)
    (dataset_output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
    (dataset_output_dir / 'metadata').mkdir(parents=True, exist_ok=True)
    
    if args.file:
        # Process single file
        osz_path = Path(args.file)
        if not osz_path.exists():
            osz_path = osz_input_dir / args.file
        if not osz_path.exists():
            print(f"[ERROR] File not found: {args.file}")
            return
        process_osz(osz_path, dataset_output_dir, args.overwrite)
    else:
        # Process all .osz files in input directory
        osz_files = list(osz_input_dir.glob('*.osz'))
        
        if not osz_files:
            print(f"[INFO] No .osz files found in {osz_input_dir}")
            return
        
        print(f"[INFO] Found {len(osz_files)} .osz files")
        
        processed = []
        for osz_path in osz_files:
            result = process_osz(osz_path, dataset_output_dir, args.overwrite)
            if result:
                processed.append(result)
        
        # Write manifest
        manifest_path = dataset_output_dir / 'metadata' / 'manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_songs': len(processed),
                'songs': processed
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUMMARY] Processed {len(processed)} songs")
        print(f"[SUMMARY] Output directory: {dataset_output_dir}")


if __name__ == "__main__":
    main()
