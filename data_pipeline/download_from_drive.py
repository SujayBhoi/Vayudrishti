"""Downloads GEE batch-export outputs from Google Drive via the Drive API,
using the same OAuth credentials already authenticated for Earth Engine
(scopes include drive access - confirmed in the original credentials file).

No Google Drive Desktop sync is installed on this machine, so this is the
direct-API alternative to load_hcho_rasters.py's assumption of a locally
synced Drive folder - either this script's output directory or an actual
Drive Desktop sync path can feed that loader.
"""

import argparse
import os

import ee
import google.auth.transport.requests
import requests

DOWNLOAD_ROOT = os.path.join(os.path.dirname(__file__), "drive_downloads")


def get_drive_credentials():
    ee.Initialize(project="isro-bah-2026")
    creds = ee.data.get_persistent_credentials()
    creds.refresh(google.auth.transport.requests.Request())
    return creds


def get_drive_headers():
    return {"Authorization": f"Bearer {get_drive_credentials().token}"}


def list_files_in_folder(folder_name: str, headers: dict) -> list[dict]:
    # Find the folder ID first
    q = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    r = requests.get("https://www.googleapis.com/drive/v3/files", headers=headers, params={"q": q, "fields": "files(id,name)"})
    r.raise_for_status()
    folders = r.json().get("files", [])
    if not folders:
        print(f"  [warn] folder '{folder_name}' not found in Drive")
        return []
    folder_id = folders[0]["id"]

    q2 = f"'{folder_id}' in parents and trashed=false"
    files = []
    page_token = None
    while True:
        params = {"q": q2, "fields": "nextPageToken, files(id,name,size)", "pageSize": 1000}
        if page_token:
            params["pageToken"] = page_token
        r = requests.get("https://www.googleapis.com/drive/v3/files", headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
        files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return files


def download_file(file_id: str, dest_path: str, creds):
    # BUG (hit mid-download on a 27-file, multi-GB batch): the OAuth access
    # token is only valid ~1hr. The original version fetched headers ONCE at
    # the start of the whole folder download; anything running longer than
    # that hit a 401 partway through. Refreshing before every file (cheap -
    # google-auth no-ops if the token still has a healthy remaining lifetime)
    # avoids this regardless of how long the overall download takes.
    if os.path.exists(dest_path):
        return
    creds.refresh(google.auth.transport.requests.Request())
    headers = {"Authorization": f"Bearer {creds.token}"}
    tmp_path = dest_path + ".part"
    r = requests.get(f"https://www.googleapis.com/drive/v3/files/{file_id}", headers=headers, params={"alt": "media"}, stream=True)
    r.raise_for_status()
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    os.replace(tmp_path, dest_path)  # atomic - a partial file from a crash mid-download never looks "complete"


def download_folder(folder_name: str) -> str:
    creds = get_drive_credentials()
    files = list_files_in_folder(folder_name, {"Authorization": f"Bearer {creds.token}"})
    out_dir = os.path.join(DOWNLOAD_ROOT, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading {len(files)} files from Drive folder '{folder_name}'...")
    for i, f in enumerate(files):
        dest = os.path.join(out_dir, f["name"])
        try:
            download_file(f["id"], dest, creds)
        except Exception as e:
            print(f"  [FAILED, will need retry] {f['name']}: {e}")
            continue
        if (i + 1) % 5 == 0 or i == len(files) - 1:
            print(f"  {i+1}/{len(files)} downloaded")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Drive folder name, e.g. ps3_obj2_hcho")
    args = parser.parse_args()
    out_dir = download_folder(args.folder)
    print(f"Saved to {out_dir}")
