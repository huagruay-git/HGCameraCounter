# Model / Config OTA — Deploy Guide (Mechanism B)

Push a new YOLO model and/or runtime config from Supabase to edge devices, by branch.
The **client is implemented and unit-tested** (`runtime/model_ota.py`,
`shared/supabase_client.py`); the backend below must be deployed to *your* Supabase
because edge devices authenticate with the **anon key + device_token** (not a logged-in
user), so the original user-RLS bootstrap/storage cannot serve them.

## How it works
1. Device calls `get_cctv_runtime_bootstrap(device_token)` (SECURITY DEFINER) → returns
   the active `config` (jsonb) + active `model` metadata for the device's branch.
2. **Config** is written to `runtime/runtime_settings.override.json` → the running
   runtime hot-applies known keys live (no restart). Put runtime-tunable keys in
   `program_configs.config_data` (e.g. `event_max_detection_age_sec`,
   `chair_service_classifier_min_conf`, `dwell_time.*` style keys the runtime reads).
3. **Model**: if `model_version` differs from the local marker, the file is downloaded,
   **sha256-verified**, swapped into `models/<model>.pt` (old kept as `.bak`), and the
   version recorded in `models/<model>.pt.version`. It becomes active on the **next
   runtime restart**. A row is written to `model_download_logs`.

## Deploy steps (one-time backend)
1. **Apply the prerequisite migrations** (if not already): the three `20260224_*` files
   (program_configs, yolo_models, storage bucket `yolo-models`, model_download_logs).
2. **Adapt + apply** `supabase/migrations/20260616_cctv_device_ota.sql`. Fix the two
   `TODO(device-auth)` blocks so `_cctv_branch_for_token()` validates a token and returns
   `branch_id` using the SAME table your `ingest_cctv_*` RPCs already use (placeholder:
   `public.cctv_devices(device_token, branch_id, is_active)`), and match
   `model_download_logs` column names.
3. **Pick a storage-access option** (in that migration, bottom):
   - *Option 1* — uncomment the `anon read yolo-models` policy (simplest; model files become
     anon-readable by path; sha256 still verified). 
   - *Option 2* — keep the bucket private, add an Edge Function that returns a short-lived
     signed URL, and include it as `signed_url` in the bootstrap model object.

## Publish a model
1. Upload the file to the `yolo-models` bucket, e.g. `branch/<branch_id>/<version>/best.pt`
   (or `global/<version>/best.pt`).
2. `sha256sum best.pt` to get the hash.
3. Insert/activate a registry row:
   ```sql
   insert into public.yolo_models
     (branch_id, model_name, model_version, storage_bucket, storage_path, sha256, is_active)
   values (<branch_id>, 'salon', '2.0.0', 'yolo-models',
           'branch/<branch_id>/2.0.0/best.pt', '<sha256>', true);
   -- deactivate older rows for that branch as needed (is_active=false)
   ```
4. (Optional) publish config:
   ```sql
   insert into public.program_configs (branch_id, config_name, version_no, is_active, config_data)
   values (<branch_id>, 'default', 2, true, '{"chair_service_classifier_min_conf": 0.45}'::jsonb);
   ```

## Run on the device
```bash
# dry-run: show what would change, no download/swap
python runtime/model_ota.py --dry-run
# apply: pull config + model
python runtime/model_ota.py
```
Schedule it (Windows Task Scheduler, off-hours) or call it just before starting the
runtime service. Config applies live; a downloaded model needs a runtime restart.

## Verify
- `logs/model_ota.log` and console JSON result.
- `models/best.pt.version` shows the applied version.
- `select * from public.model_download_logs order by created_at desc limit 5;`

## Security
- The device uses the **anon key** (already in `config.yaml`) + `device_token`. Treat the
  device_token like a credential. Option 1 makes model files anon-readable — only use it
  if your models are not sensitive.
