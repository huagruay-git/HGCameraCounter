-- HG Camera Counter - Supabase migration
-- Date: 2026-06-16
-- Purpose: DEVICE-TOKEN authenticated OTA for edge devices.
--
-- Why this exists:
--   The earlier model/config OTA objects (program_configs, yolo_models,
--   get_runtime_bootstrap, yolo-models bucket) are guarded by RLS that uses
--   auth.uid() + public.user_branch_access -> they only work for a logged-in
--   DASHBOARD user. Edge devices authenticate with the anon key + a device_token
--   (the ingest_cctv_* RPCs already follow this pattern), so they cannot read
--   those tables/bucket directly. This migration adds SECURITY DEFINER RPCs that
--   validate the device_token and return the active config + model for the
--   device's branch, plus a download-log writer.
--
-- ⚠️ ADAPT BEFORE DEPLOY:
--   The two helper blocks marked `TODO(device-auth)` must be aligned with the
--   ACTUAL table your existing ingest_cctv_* RPCs use to validate device_token
--   and resolve branch_id (referred to here as public.cctv_devices with columns
--   device_token / branch_id / is_active). Rename/adjust to match your schema.

begin;

-- ---------------------------------------------------------------------------
-- Helper: resolve a valid device_token -> branch_id (raises if invalid)
-- TODO(device-auth): point this at your real device table / validation logic.
-- ---------------------------------------------------------------------------
create or replace function public._cctv_branch_for_token(p_device_token text)
returns bigint
language plpgsql
security definer
set search_path = public
as $$
declare
  v_branch_id bigint;
begin
  if p_device_token is null or length(trim(p_device_token)) = 0 then
    raise exception 'missing device token';
  end if;

  select d.branch_id
    into v_branch_id
  from public.cctv_devices d                       -- TODO(device-auth): your table
  where d.device_token = trim(p_device_token)
    and coalesce(d.is_active, true) = true
  limit 1;

  if v_branch_id is null then
    raise exception 'invalid or inactive device token';
  end if;
  return v_branch_id;
end;
$$;

-- ---------------------------------------------------------------------------
-- RPC: get_cctv_runtime_bootstrap(device_token) -> {config, model}
-- SECURITY DEFINER so it bypasses the user-scoped RLS on program_configs /
-- yolo_models, AFTER validating the device token.
-- ---------------------------------------------------------------------------
create or replace function public.get_cctv_runtime_bootstrap(p_device_token text)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  v_branch_id bigint := public._cctv_branch_for_token(p_device_token);
begin
  return jsonb_build_object(
    'branch_id', v_branch_id,
    'config', (
      select pc.config_data
      from public.program_configs pc
      where pc.branch_id = v_branch_id
        and pc.config_name = 'default'
        and pc.is_active = true
      order by pc.version_no desc
      limit 1
    ),
    'model', (
      select jsonb_build_object(
        'id', ym.id,
        'branch_id', ym.branch_id,
        'model_name', ym.model_name,
        'model_version', ym.model_version,
        'storage_bucket', ym.storage_bucket,
        'storage_path', ym.storage_path,
        'sha256', ym.sha256,
        'metrics', ym.metrics
        -- NOTE: no signed_url here (SQL cannot mint storage signed URLs).
        -- The device downloads via the storage read policy below, OR add an
        -- Edge Function that returns a signed URL and include it here.
      )
      from public.yolo_models ym
      where (ym.branch_id = v_branch_id or ym.branch_id is null)
        and ym.is_active = true
        and ym.is_deleted = false
      order by case when ym.branch_id = v_branch_id then 0 else 1 end, ym.created_at desc
      limit 1
    )
  );
end;
$$;

-- ---------------------------------------------------------------------------
-- RPC: log_cctv_model_download(...) -> records an OTA download outcome.
-- SECURITY DEFINER so the anon device can write its own audit row after
-- validating the device token.
-- TODO(device-auth): match column names to your public.model_download_logs.
-- ---------------------------------------------------------------------------
create or replace function public.log_cctv_model_download(
  p_device_token text,
  p_model_id uuid,
  p_model_version text,
  p_status text,
  p_message text default null,
  p_sha256_ok boolean default null
)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare
  v_branch_id bigint := public._cctv_branch_for_token(p_device_token);
begin
  insert into public.model_download_logs
    (branch_id, model_id, model_version, status, message, sha256_ok, created_at)
  values
    (v_branch_id, p_model_id, p_model_version, p_status, p_message, p_sha256_ok, now());
exception when undefined_column or undefined_table then
  -- Keep OTA resilient if the audit table differs; do not fail the device.
  raise notice 'log_cctv_model_download: audit table mismatch, skipped';
end;
$$;

-- ---------------------------------------------------------------------------
-- Allow the anon (device) role to EXECUTE the device RPCs.
-- ---------------------------------------------------------------------------
grant execute on function public.get_cctv_runtime_bootstrap(text) to anon;
grant execute on function public.log_cctv_model_download(text, uuid, text, text, text, boolean) to anon;

-- ---------------------------------------------------------------------------
-- STORAGE ACCESS for the device (choose ONE option, then uncomment):
--
-- Option 1 (simplest): allow anon to READ the yolo-models bucket. Model files
--   become readable by anyone holding the public anon key + path. Acceptable if
--   models are not sensitive. The runtime still verifies sha256 before use.
--
--   drop policy if exists "anon read yolo models" on storage.objects;
--   create policy "anon read yolo models"
--   on storage.objects for select to anon
--   using (bucket_id = 'yolo-models');
--
-- Option 2 (more secure): keep the bucket private and add a Supabase Edge
--   Function `cctv_model_signed_url(device_token, storage_path)` that validates
--   the token (via _cctv_branch_for_token) and returns a short-lived signed URL;
--   include that URL as `signed_url` in get_cctv_runtime_bootstrap's model object.
-- ---------------------------------------------------------------------------

commit;
