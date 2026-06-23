-- HG Camera Counter - Supabase migration
-- Date: 2026-06-16 (adapted to real schema 2026-06-23)
-- Purpose: DEVICE-TOKEN authenticated OTA for edge devices.
--
-- Edge devices authenticate with the anon key + a device_token (the ingest_cctv_*
-- RPCs already follow this pattern), so they cannot read the user-scoped
-- program_configs / yolo_models tables directly. This migration adds SECURITY
-- DEFINER RPCs that validate the device_token and return the active config + model
-- for the device's branch, plus a download-log writer.
--
-- Adapted to the real schema (verified 2026-06-23):
--   * device token is validated by SHA-256 hash against cctv_devices.device_token_hash
--     (with a legacy plaintext fallback) — identical to ingest_cctv_heartbeat.
--   * model_download_logs has columns (branch_id, model_id, status, detail, downloaded_at).

begin;

-- ---------------------------------------------------------------------------
-- Helper: resolve a valid device_token -> branch_id (raises if invalid).
-- Mirrors the hash check used by ingest_cctv_heartbeat / ingest_cctv_realtime.
-- ---------------------------------------------------------------------------
create or replace function public._cctv_branch_for_token(p_device_token text)
returns bigint
language plpgsql
security definer
set search_path = public, extensions
as $$
declare
  v_branch_id bigint;
  v_token_hash text;
begin
  if p_device_token is null or length(trim(p_device_token)) = 0 then
    raise exception 'missing device token' using errcode = '22023';
  end if;

  v_token_hash := encode(digest(trim(p_device_token), 'sha256'), 'hex');

  select d.branch_id
    into v_branch_id
  from public.cctv_devices d
  where coalesce(d.is_active, true) = true
    and (d.device_token_hash = v_token_hash
         or d.device_token = trim(p_device_token))   -- legacy fallback
  limit 1;

  if v_branch_id is null then
    raise exception 'invalid or inactive device token' using errcode = '42501';
  end if;
  return v_branch_id;
end;
$$;

-- ---------------------------------------------------------------------------
-- RPC: get_cctv_runtime_bootstrap(device_token) -> {branch_id, config, model}
-- SECURITY DEFINER so it bypasses the user-scoped RLS on program_configs /
-- yolo_models, AFTER validating the device token.
-- ---------------------------------------------------------------------------
create or replace function public.get_cctv_runtime_bootstrap(p_device_token text)
returns jsonb
language plpgsql
security definer
set search_path = public, extensions
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
-- Adapted to real public.model_download_logs columns
-- (branch_id, model_id, status, detail, downloaded_at).
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
set search_path = public, extensions
as $$
declare
  v_branch_id bigint := public._cctv_branch_for_token(p_device_token);
begin
  insert into public.model_download_logs
    (branch_id, model_id, status, detail, downloaded_at)
  values (
    v_branch_id,
    p_model_id,
    p_status,
    concat_ws(' | ',
      nullif(p_message, ''),
      'version=' || coalesce(p_model_version, '?'),
      'sha256_ok=' || coalesce(p_sha256_ok::text, '?')),
    now()
  );
exception when undefined_column or undefined_table then
  raise notice 'log_cctv_model_download: audit table mismatch, skipped';
end;
$$;

-- ---------------------------------------------------------------------------
-- Allow the anon (device) role to EXECUTE the device RPCs.
-- ---------------------------------------------------------------------------
grant execute on function public.get_cctv_runtime_bootstrap(text) to anon;
grant execute on function public.log_cctv_model_download(text, uuid, text, text, text, boolean) to anon;

commit;

-- ---------------------------------------------------------------------------
-- STORAGE ACCESS for model-FILE download (NOT enabled here — decide first):
-- get_cctv_runtime_bootstrap returns model metadata but not the file. To let a
-- device download the model from the private `yolo-models` bucket with just the
-- anon key + device_token, EITHER:
--
-- Option 1 (simplest) allow anon to READ the bucket (model files become readable
--   by anyone holding the public anon key + path; runtime still verifies sha256):
--     create policy "anon read yolo models" on storage.objects
--       for select to anon using (bucket_id = 'yolo-models');
--
-- Option 2 (more secure) keep the bucket private; add an Edge Function
--   cctv_model_signed_url(device_token, storage_path) that validates the token and
--   returns a short-lived signed URL, surfaced as model.signed_url in the bootstrap.
-- Config OTA (config_data) already works without any storage change.
-- ---------------------------------------------------------------------------
