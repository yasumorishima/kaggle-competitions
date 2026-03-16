"""Google Drive sync utility for GitHub Actions.

Downloads EXP results from Google Drive to local filesystem.
Used by GitHub Actions workflows to fetch experiment outputs.

Requires:
  - GOOGLE_SERVICE_ACCOUNT_KEY secret (JSON key)
  - Drive folder shared with service account email

Usage:
    python scripts/drive_sync.py --folder-id <DRIVE_FOLDER_ID> --local-dir ./exp-results
    python scripts/drive_sync.py --comp s6e3-churn --exp EXP001 --child child-exp005
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


def get_drive_service():
    """Create Drive API service from environment variable or key file."""
    key_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
    if key_json:
        info = json.loads(key_json)
    elif os.path.exists("service-account-key.json"):
        with open("service-account-key.json") as f:
            info = json.load(f)
    else:
        print("ERROR: No service account key found.")
        print("Set GOOGLE_SERVICE_ACCOUNT_KEY env var or provide service-account-key.json")
        sys.exit(1)

    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)


def find_folder_by_path(service, path_parts, parent_id="root"):
    """Navigate folder hierarchy: kaggle/s6e3-churn/EXP/EXP001/output/child-exp005"""
    current_id = parent_id
    for part in path_parts:
        query = (
            f"name='{part}' and '{current_id}' in parents "
            f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get("files", [])
        if not files:
            print(f"Folder not found: {part} (in {current_id})")
            return None
        current_id = files[0]["id"]
    return current_id


def download_folder(service, folder_id, local_dir):
    """Recursively download all files from a Drive folder."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query, fields="files(id, name, mimeType, size)"
    ).execute()

    for item in results.get("files", []):
        if item["mimeType"] == "application/vnd.google-apps.folder":
            download_folder(service, item["id"], local_dir / item["name"])
        else:
            local_path = local_dir / item["name"]
            print(f"  Downloading: {local_path}")
            request = service.files().get_media(fileId=item["id"])
            with open(local_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()


def download_result(service, comp, exp, child, shared_folder_id):
    """Download a specific child-exp result from Drive."""
    path_parts = ["kaggle", comp, "EXP", exp, "output", child]
    folder_id = find_folder_by_path(service, path_parts, shared_folder_id)
    if not folder_id:
        print(f"Result not found: {'/'.join(path_parts)}")
        return None

    local_dir = Path(f"exp-results/{comp}/{exp}/{child}")
    download_folder(service, folder_id, local_dir)
    return local_dir


def list_completed_experiments(service, comp, exp, shared_folder_id):
    """List all child-exp results that have result.json."""
    path_parts = ["kaggle", comp, "EXP", exp, "output"]
    output_folder_id = find_folder_by_path(service, path_parts, shared_folder_id)
    if not output_folder_id:
        return []

    query = (
        f"'{output_folder_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()

    completed = []
    for folder in results.get("files", []):
        # Check if result.json exists
        rq = (
            f"name='result.json' and '{folder['id']}' in parents and trashed=false"
        )
        r = service.files().list(q=rq, fields="files(id)").execute()
        if r.get("files"):
            completed.append(folder["name"])

    return sorted(completed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp", help="Competition slug (e.g., s6e3-churn)")
    parser.add_argument("--exp", default="EXP001", help="EXP number")
    parser.add_argument("--child", help="child-exp to download (e.g., child-exp005)")
    parser.add_argument("--list", action="store_true", help="List completed experiments")
    parser.add_argument("--folder-id", help="Shared Drive folder ID")
    parser.add_argument("--local-dir", default="exp-results", help="Local download dir")
    args = parser.parse_args()

    service = get_drive_service()
    shared_id = args.folder_id or os.environ.get("DRIVE_SHARED_FOLDER_ID", "root")

    if args.list:
        completed = list_completed_experiments(service, args.comp, args.exp, shared_id)
        print(f"Completed experiments for {args.comp}/{args.exp}:")
        for c in completed:
            print(f"  {c}")
        return

    if args.child:
        local = download_result(service, args.comp, args.exp, args.child, shared_id)
        if local:
            result_path = local / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                print(f"\nResult: cv_score={result.get('cv_score')}")
    else:
        # Download all completed experiments
        completed = list_completed_experiments(service, args.comp, args.exp, shared_id)
        for child in completed:
            download_result(service, args.comp, args.exp, child, shared_id)


if __name__ == "__main__":
    main()
