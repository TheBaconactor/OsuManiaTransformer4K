"""
osu! Beatmap Audio Downloader for Osu2MIR Dataset

This script downloads beatmap sets from osu.ppy.sh using the osu! API v2
and extracts the audio files to match with the Osu2MIR annotations.

Setup:
1. Go to https://osu.ppy.sh/home/account/edit
2. Scroll down to "OAuth" section
3. Click "New OAuth Application"
4. Name: "Osu2MIR Downloader" (or any name)
5. Callback URL: http://localhost (doesn't matter for this use case)
6. Copy the Client ID and Client Secret
7. Set them as environment variables or pass to this script

Usage:
    python download_beatmaps.py --client-id YOUR_ID --client-secret YOUR_SECRET
"""

import os
import sys
import json
import time
import hashlib
import zipfile
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


class OsuAPIClient:
    """osu! API v2 client for downloading beatmaps."""
    
    BASE_URL = "https://osu.ppy.sh/api/v2"
    AUTH_URL = "https://osu.ppy.sh/oauth/token"
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expires = 0
        
    def authenticate(self):
        """Get OAuth2 access token."""
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
            "scope": "public"
        }
        
        response = requests.post(self.AUTH_URL, data=data)
        if response.status_code != 200:
            raise Exception(f"Authentication failed: {response.text}")
        
        token_data = response.json()
        self.access_token = token_data["access_token"]
        self.token_expires = time.time() + token_data["expires_in"] - 60
        print("✓ Authenticated with osu! API")
        
    def _ensure_authenticated(self):
        """Ensure we have a valid token."""
        if not self.access_token or time.time() > self.token_expires:
            self.authenticate()
            
    def _headers(self):
        """Get request headers with auth."""
        self._ensure_authenticated()
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }
    
    def get_beatmapset(self, beatmapset_id: int) -> dict:
        """Get beatmapset metadata."""
        url = f"{self.BASE_URL}/beatmapsets/{beatmapset_id}"
        response = requests.get(url, headers=self._headers())
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


def download_beatmapset(beatmapset_id: int, output_dir: Path, session: requests.Session) -> Path:
    """
    Download a beatmapset .osz file from osu! mirror.
    
    Note: The official API doesn't provide direct download links.
    We use the osu! direct download which requires being logged in via browser cookies,
    OR we can use a mirror service.
    """
    # Try official download (requires authentication via browser session)
    # Using the direct download URL pattern
    download_url = f"https://osu.ppy.sh/beatmapsets/{beatmapset_id}/download"
    
    output_path = output_dir / f"{beatmapset_id}.osz"
    
    if output_path.exists():
        print(f"  ⏭ {beatmapset_id}.osz already exists, skipping")
        return output_path
    
    # Try catboy.best mirror (commonly used for bulk downloads)
    mirror_url = f"https://catboy.best/d/{beatmapset_id}"
    
    try:
        response = session.get(mirror_url, stream=True, timeout=60)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  ✓ Downloaded {beatmapset_id}.osz")
            return output_path
    except Exception as e:
        print(f"  ⚠ Mirror failed for {beatmapset_id}: {e}")
    
    # Try chimu.moe mirror as fallback
    chimu_url = f"https://api.chimu.moe/v1/download/{beatmapset_id}"
    try:
        response = session.get(chimu_url, stream=True, timeout=60)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  ✓ Downloaded {beatmapset_id}.osz (chimu)")
            return output_path
    except Exception as e:
        print(f"  ⚠ Chimu mirror failed for {beatmapset_id}: {e}")
    
    print(f"  ✗ Failed to download {beatmapset_id}")
    return None


def extract_audio_from_osz(osz_path: Path, output_dir: Path, expected_md5: str = None) -> Path:
    """Extract audio file from .osz archive."""
    try:
        with zipfile.ZipFile(osz_path, 'r') as zf:
            # Find audio files (mp3, ogg, wav)
            audio_files = [f for f in zf.namelist() 
                          if f.lower().endswith(('.mp3', '.ogg', '.wav'))]
            
            if not audio_files:
                print(f"  ⚠ No audio found in {osz_path.name}")
                return None
            
            # Usually there's one main audio file, or pick the largest
            for audio_file in audio_files:
                info = zf.getinfo(audio_file)
                audio_data = zf.read(audio_file)
                
                # Calculate MD5
                md5 = hashlib.md5(audio_data).hexdigest()
                
                # If we have an expected MD5, check it
                if expected_md5 and md5 == expected_md5:
                    output_path = output_dir / f"{md5}{Path(audio_file).suffix}"
                    with open(output_path, 'wb') as f:
                        f.write(audio_data)
                    print(f"  ✓ Extracted {audio_file} → {output_path.name}")
                    return output_path
                elif not expected_md5:
                    # Just extract the first audio file
                    output_path = output_dir / f"{md5}{Path(audio_file).suffix}"
                    with open(output_path, 'wb') as f:
                        f.write(audio_data)
                    print(f"  ✓ Extracted {audio_file} → {output_path.name}")
                    return output_path
                    
            # If expected MD5 not found, extract all and warn
            if expected_md5:
                print(f"  ⚠ Expected MD5 {expected_md5} not found in {osz_path.name}")
                # Extract first one anyway
                audio_file = audio_files[0]
                audio_data = zf.read(audio_file)
                md5 = hashlib.md5(audio_data).hexdigest()
                output_path = output_dir / f"{md5}{Path(audio_file).suffix}"
                with open(output_path, 'wb') as f:
                    f.write(audio_data)
                return output_path
                
    except zipfile.BadZipFile:
        print(f"  ✗ Corrupt archive: {osz_path.name}")
        return None
    except Exception as e:
        print(f"  ✗ Error extracting {osz_path.name}: {e}")
        return None


def parse_annotation_filename(filename: str) -> tuple:
    """Parse MD5 and BeatmapSetID from annotation filename."""
    # Format: <MD5>_<BeatmapSetID>_beats_metered.txt
    parts = filename.replace("_beats_metered.txt", "").split("_")
    if len(parts) >= 2:
        md5 = parts[0]
        beatmapset_id = int(parts[1])
        return md5, beatmapset_id
    return None, None


def get_unique_beatmapsets(annotations_dir: Path) -> dict:
    """Get unique beatmapset IDs and their expected MD5s from annotations."""
    beatmapsets = {}  # beatmapset_id -> list of expected MD5s
    
    for anno_file in annotations_dir.glob("*.txt"):
        md5, beatmapset_id = parse_annotation_filename(anno_file.name)
        if beatmapset_id:
            if beatmapset_id not in beatmapsets:
                beatmapsets[beatmapset_id] = []
            beatmapsets[beatmapset_id].append(md5)
    
    return beatmapsets


def main():
    parser = argparse.ArgumentParser(description="Download osu! beatmap audio for Osu2MIR")
    parser.add_argument("--client-id", type=str, 
                       default=os.environ.get("OSU_CLIENT_ID"),
                       help="osu! API Client ID")
    parser.add_argument("--client-secret", type=str,
                       default=os.environ.get("OSU_CLIENT_SECRET"),
                       help="osu! API Client Secret")
    parser.add_argument("--annotations-dir", type=str,
                       default="annotations",
                       help="Directory containing annotation files")
    parser.add_argument("--output-dir", type=str,
                       default="audio",
                       help="Directory to save audio files")
    parser.add_argument("--osz-dir", type=str,
                       default="osz_cache",
                       help="Directory to cache .osz files")
    parser.add_argument("--max-workers", type=int, default=3,
                       help="Number of parallel downloads")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of beatmaps to download (for testing)")
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    annotations_dir = script_dir / args.annotations_dir
    output_dir = script_dir / args.output_dir
    osz_dir = script_dir / args.osz_dir
    
    output_dir.mkdir(exist_ok=True)
    osz_dir.mkdir(exist_ok=True)
    
    # Get beatmapsets to download
    print(f"📂 Scanning annotations in {annotations_dir}")
    beatmapsets = get_unique_beatmapsets(annotations_dir)
    print(f"   Found {len(beatmapsets)} unique beatmapsets")
    
    if args.limit:
        beatmapset_ids = list(beatmapsets.keys())[:args.limit]
        beatmapsets = {k: beatmapsets[k] for k in beatmapset_ids}
        print(f"   Limited to {len(beatmapsets)} beatmapsets")
    
    # Initialize API client (optional, for metadata)
    api_client = None
    if args.client_id and args.client_secret:
        try:
            api_client = OsuAPIClient(args.client_id, args.client_secret)
            api_client.authenticate()
        except Exception as e:
            print(f"⚠ API authentication failed: {e}")
            print("  Continuing without API (using mirrors only)")
    
    # Download and extract
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Osu2MIR-Downloader/1.0"
    })
    
    successful = 0
    failed = 0
    
    for i, (beatmapset_id, md5_list) in enumerate(beatmapsets.items(), 1):
        print(f"\n[{i}/{len(beatmapsets)}] Beatmapset {beatmapset_id}")
        
        # Download .osz
        osz_path = download_beatmapset(beatmapset_id, osz_dir, session)
        
        if osz_path:
            # Extract audio for each expected MD5
            for md5 in md5_list:
                audio_path = extract_audio_from_osz(osz_path, output_dir, md5)
                if audio_path:
                    successful += 1
                else:
                    failed += 1
        else:
            failed += len(md5_list)
        
        # Rate limiting
        time.sleep(0.5)
    
    print(f"\n{'='*50}")
    print(f"✓ Successfully extracted: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"📁 Audio files saved to: {output_dir}")


if __name__ == "__main__":
    main()
