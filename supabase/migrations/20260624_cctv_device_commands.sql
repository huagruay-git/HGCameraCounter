-- HG Camera Counter - Supabase migration
-- Date: 2026-06-24
-- Purpose: REMOTE DEVICE COMMANDS (reboot / restart_app / shutdown / power_cycle / wake).
--
-- The edge device already heartbeats every ~minute (proven E2E). This adds a tiny
-- command queue the device polls on the same device-token channel, so an operator can
-- remotely reboot/restart a branch mini-PC without remoting in. SECURITY DEFINER RPCs
-- mirror the device-token auth used by ingest_cctv_heartbeat / get_cctv_runtime_bootstrap.
--
-- NOTE on scope: software commands (reboot/restart_app/shutdown) only run while the PC is
-- ON and the app is alive. 'power_cycle'/'wake' are delivered the same way but require the
-- device side to drive a smart-plug/WOL helper (config-driven); a fully-OFF PC cannot be
-- woken by its own software.

begin;

-- ---------------------------------------------------------------------------
-- Queue table
-- ---------------------------------------------------------------------------
create table if not exists public.cctv_device_commands (
  id          uuid primary key default gen_random_uuid(),
  device_id   bigint references public.cctv_devices(id) on delete cascade,
  branch_id   bigint,
  command     text not null,                       -- reboot|restart_app|shutdown|power_cycle|wake|update_now|ping
  args        jsonb not null default '{}'::jsonb,
  status      text not null default 'pending',     -- pending|acked|done|failed|expired
  detail      text,
  created_by  uuid default auth.uid(),
  created_at  timestamptz not null default now(),
  acked_at    timestamptz,
  done_at     timestamptz,
  expires_at  timestamptz not null default now() + interval '1 hour'
);

create index if not exists idx_cctv_cmd_device_pending
  on public.cctv_device_commands (device_id, status, expires_at);

alter table public.cctv_device_commands enable row level security;
-- Access is via the SECURITY DEFINER RPCs below; no broad table policies are granted.

-- ---------------------------------------------------------------------------
-- Helper: device_token -> device id (mirrors _cctv_branch_for_token's hash check)
-- ---------------------------------------------------------------------------
create or replace function public._cctv_device_for_token(p_device_token text)
returns bigint
language plpgsql
security definer
set search_path = public, extensions
as $$
declare
  v_id   bigint;
  v_hash text;
begin
  if p_device_token is null or length(trim(p_device_token)) = 0 then
    raise exception 'missing device token' using errcode = '22023';
  end if;
  v_hash := encode(digest(trim(p_device_token), 'sha256'), 'hex');
  select d.id into v_id
  from public.cctv_devices d
  where coalesce(d.is_active, true) = true
    and (d.device_token_hash = v_hash or d.device_token = trim(p_device_token))
  limit 1;
  if v_id is null then
    raise exception 'invalid or inactive device token' using errcode = '42501';
  end if;
  return v_id;
end;
$$;

-- ---------------------------------------------------------------------------
-- RPC: device pulls its pending commands (and atomically marks them 'acked' so
-- they are never served twice). Expires stale ones first. anon-executable.
-- ---------------------------------------------------------------------------
create or replace function public.get_cctv_device_commands(p_device_token text)
returns jsonb
language plpgsql
security definer
set search_path = public, extensions
as $$
declare
  v_device bigint := public._cctv_device_for_token(p_device_token);
  v_branch bigint;
  v_out    jsonb;
begin
  select branch_id into v_branch from public.cctv_devices where id = v_device;

  update public.cctv_device_commands
     set status = 'expired'
   where status = 'pending'
     and expires_at < now()
     and (device_id = v_device or (device_id is null and branch_id = v_branch));

  with picked as (
    update public.cctv_device_commands c
       set status = 'acked', acked_at = now()
     where c.status = 'pending'
       and c.expires_at >= now()
       and (c.device_id = v_device or (c.device_id is null and c.branch_id = v_branch))
     returning c.id, c.command, c.args, c.created_at
  )
  select coalesce(
           jsonb_agg(jsonb_build_object(
             'id', id, 'command', command, 'args', args, 'created_at', created_at
           ) order by created_at),
           '[]'::jsonb)
    into v_out
  from picked;

  return v_out;
end;
$$;

-- ---------------------------------------------------------------------------
-- RPC: device reports the outcome of a command. anon-executable.
-- ---------------------------------------------------------------------------
create or replace function public.ack_cctv_device_command(
  p_device_token text,
  p_command_id   uuid,
  p_status       text,
  p_detail       text default null
)
returns void
language plpgsql
security definer
set search_path = public, extensions
as $$
declare
  v_device bigint := public._cctv_device_for_token(p_device_token);
  v_branch bigint;
begin
  select branch_id into v_branch from public.cctv_devices where id = v_device;
  update public.cctv_device_commands
     set status  = case when p_status in ('done', 'failed') then p_status else 'done' end,
         detail  = p_detail,
         done_at = now()
   where id = p_command_id
     and (device_id = v_device or (device_id is null and branch_id = v_branch));
end;
$$;

-- ---------------------------------------------------------------------------
-- RPC: operator enqueues a command. Authenticated users only (the dashboard /
-- a logged-in CLI). Returns the new command id.
-- ---------------------------------------------------------------------------
create or replace function public.admin_enqueue_cctv_command(
  p_device_id          bigint,
  p_command            text,
  p_args               jsonb default '{}'::jsonb,
  p_expires_in_minutes int default 60
)
returns uuid
language plpgsql
security definer
set search_path = public, extensions
as $$
declare
  v_id     uuid;
  v_branch bigint;
begin
  -- allow logged-in dashboard users AND service-role admin tools (CLI); block anon/device
  if coalesce(auth.role(), 'anon') not in ('authenticated', 'service_role') then
    raise exception 'authentication required' using errcode = '42501';
  end if;
  if p_command is null or length(trim(p_command)) = 0 then
    raise exception 'command is required' using errcode = '22023';
  end if;
  select branch_id into v_branch from public.cctv_devices where id = p_device_id;
  if v_branch is null then
    raise exception 'unknown device %', p_device_id using errcode = '22023';
  end if;
  insert into public.cctv_device_commands (device_id, branch_id, command, args, expires_at, created_by)
  values (p_device_id, v_branch, trim(p_command), coalesce(p_args, '{}'::jsonb),
          now() + make_interval(mins => greatest(1, p_expires_in_minutes)), auth.uid())
  returning id into v_id;
  return v_id;
end;
$$;

grant execute on function public.get_cctv_device_commands(text) to anon;
grant execute on function public.ack_cctv_device_command(text, uuid, text, text) to anon;
grant execute on function public.admin_enqueue_cctv_command(bigint, text, jsonb, int) to authenticated;

commit;
