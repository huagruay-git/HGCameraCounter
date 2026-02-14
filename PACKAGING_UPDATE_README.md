Remote Update Workflow (minimal)

1) Metadata endpoint
- Provide a JSON metadata URL that describes the latest release. Example schema:
{
  "version": "1.2.3",
  "notes": "Changelog",
  "assets": [
    {"name": "HGCameraCounter-1.2.3.zip", "url": "https://.../HGCameraCounter-1.2.3.zip", "sha256": "..."}
  ]
}

2) Controller UI
- Uses `shared.updater.Updater` to fetch metadata, download primary asset, verify sha256 and stage under `updates/`.
- The UI does NOT auto-install; operator reviews and runs installer or deploy steps.

3) Staging
- Downloaded files placed in `updates/` (configurable via `updates.download_dir`).

4) Installation (operator)
- If the asset is an installer (`.exe` on Windows), run it manually:
  updates\\HGCameraCounter-1.2.3.exe
- If the asset is a zip/package, extract to a deployment workspace and run tests before swapping binaries.

5) Security
- Host metadata and assets over HTTPS.
- Sign packages and/or publish SHA256 in metadata. Verify SHA256 before trusting.

6) Future improvements
- Add signed metadata (e.g., detached signature or JWT) and verify signatures.
- Add atomic deployment: download -> verify -> extract to new folder -> swap symlink or service restart.
- Add auto-install option with rollback support.

