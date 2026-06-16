-- HG Camera Counter - Supabase migration
-- Date: 2026-02-24
-- Purpose:
--   1) Program config storage (versioned, branch-scoped)
--   2) YOLO model registry metadata (branch/global)
--   3) RLS policies using existing public.user_branch_access
--   4) Helper RPC for runtime bootstrap
--
-- Safe to re-run: mostly idempotent (uses IF NOT EXISTS / DROP POLICY IF EXISTS).

begin;

create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- Table: public.program_configs
-- ---------------------------------------------------------------------------
create table if not exists public.program_configs (
  id uuid primary key default gen_random_uuid(),
  branch_id bigint not null references public.branches(id) on delete cascade,
  config_name text not null default 'default',
  version_no int not null,
  is_active boolean not null default false,
  config_data jsonb not null,
  note text,
  created_by uuid references auth.users(id),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint program_configs_branch_config_version_key unique (branch_id, config_name, version_no)
);

create index if not exists idx_program_configs_branch_active
  on public.program_configs(branch_id, config_name, is_active);

create index if not exists idx_program_configs_config_data_gin
  on public.program_configs using gin (config_data);

-- ---------------------------------------------------------------------------
-- Table: public.yolo_models
-- ---------------------------------------------------------------------------
create table if not exists public.yolo_models (
  id uuid primary key default gen_random_uuid(),
  branch_id bigint references public.branches(id) on delete cascade, -- null = global model
  model_name text not null,
  model_version text not null,
  task_type text not null default 'detect',
  classes jsonb,
  metrics jsonb,
  storage_bucket text not null default 'yolo-models',
  storage_path text not null,
  file_size_bytes bigint,
  sha256 text,
  is_active boolean not null default false,
  is_deleted boolean not null default false,
  note text,
  created_by uuid references auth.users(id),
  created_at timestamptz not null default now(),
  constraint yolo_models_model_version_storage_path_key unique (model_version, storage_path)
);

create index if not exists idx_yolo_models_branch_active
  on public.yolo_models(branch_id, is_active);

create index if not exists idx_yolo_models_metrics_gin
  on public.yolo_models using gin (metrics);

-- ---------------------------------------------------------------------------
-- Trigger: updated_at for program_configs
-- ---------------------------------------------------------------------------
create or replace function public.set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trg_program_configs_updated_at on public.program_configs;
create trigger trg_program_configs_updated_at
before update on public.program_configs
for each row execute function public.set_updated_at();

-- ---------------------------------------------------------------------------
-- RLS Enable
-- ---------------------------------------------------------------------------
alter table public.program_configs enable row level security;
alter table public.yolo_models enable row level security;

-- ---------------------------------------------------------------------------
-- Policies: program_configs (using public.user_branch_access)
-- ---------------------------------------------------------------------------
drop policy if exists "read config by branch access" on public.program_configs;
create policy "read config by branch access"
on public.program_configs
for select
to authenticated
using (
  exists (
    select 1
    from public.user_branch_access uba
    where uba.user_id = auth.uid()
      and uba.branch_id = program_configs.branch_id
      and uba.role in ('STAFF','BARBER','BRANCH_STAFF','BRANCH_ADMIN','COMPANY_ADMIN')
  )
);

drop policy if exists "insert config by admin roles" on public.program_configs;
create policy "insert config by admin roles"
on public.program_configs
for insert
to authenticated
with check (
  exists (
    select 1
    from public.user_branch_access uba
    where uba.user_id = auth.uid()
      and uba.branch_id = program_configs.branch_id
      and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
  )
);

drop policy if exists "update config by admin roles" on public.program_configs;
create policy "update config by admin roles"
on public.program_configs
for update
to authenticated
using (
  exists (
    select 1
    from public.user_branch_access uba
    where uba.user_id = auth.uid()
      and uba.branch_id = program_configs.branch_id
      and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
  )
)
with check (
  exists (
    select 1
    from public.user_branch_access uba
    where uba.user_id = auth.uid()
      and uba.branch_id = program_configs.branch_id
      and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
  )
);

drop policy if exists "delete config by admin roles" on public.program_configs;
create policy "delete config by admin roles"
on public.program_configs
for delete
to authenticated
using (
  exists (
    select 1
    from public.user_branch_access uba
    where uba.user_id = auth.uid()
      and uba.branch_id = program_configs.branch_id
      and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
  )
);

-- ---------------------------------------------------------------------------
-- Policies: yolo_models (branch-scoped + optional global models)
-- ---------------------------------------------------------------------------
drop policy if exists "read models by branch access" on public.yolo_models;
create policy "read models by branch access"
on public.yolo_models
for select
to authenticated
using (
  not is_deleted
  and (
    branch_id is null
    or exists (
      select 1
      from public.user_branch_access uba
      where uba.user_id = auth.uid()
        and uba.branch_id = yolo_models.branch_id
        and uba.role in ('STAFF','BARBER','BRANCH_STAFF','BRANCH_ADMIN','COMPANY_ADMIN')
    )
  )
);

drop policy if exists "insert models by admin roles" on public.yolo_models;
create policy "insert models by admin roles"
on public.yolo_models
for insert
to authenticated
with check (
  (
    branch_id is null
    and exists (
      select 1
      from public.user_branch_access uba
      where uba.user_id = auth.uid()
        and uba.role = 'COMPANY_ADMIN'
    )
  )
  or
  (
    branch_id is not null
    and exists (
      select 1
      from public.user_branch_access uba
      where uba.user_id = auth.uid()
        and uba.branch_id = yolo_models.branch_id
        and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
    )
  )
);

drop policy if exists "update models by admin roles" on public.yolo_models;
create policy "update models by admin roles"
on public.yolo_models
for update
to authenticated
using (
  (
    branch_id is null
    and exists (
      select 1
      from public.user_branch_access uba
      where uba.user_id = auth.uid()
        and uba.role = 'COMPANY_ADMIN'
    )
  )
  or
  (
    branch_id is not null
    and exists (
      select 1
      from public.user_branch_access uba
      where uba.user_id = auth.uid()
        and uba.branch_id = yolo_models.branch_id
        and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
    )
  )
)
with check (
  (
    branch_id is null
    and exists (
      select 1
      from public.user_branch_access uba
      where uba.user_id = auth.uid()
        and uba.role = 'COMPANY_ADMIN'
    )
  )
  or
  (
    branch_id is not null
    and exists (
      select 1
      from public.user_branch_access uba
      where uba.user_id = auth.uid()
        and uba.branch_id = yolo_models.branch_id
        and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
    )
  )
);

drop policy if exists "delete models by admin roles" on public.yolo_models;
create policy "delete models by admin roles"
on public.yolo_models
for delete
to authenticated
using (
  (
    branch_id is null
    and exists (
      select 1
      from public.user_branch_access uba
      where uba.user_id = auth.uid()
        and uba.role = 'COMPANY_ADMIN'
    )
  )
  or
  (
    branch_id is not null
    and exists (
      select 1
      from public.user_branch_access uba
      where uba.user_id = auth.uid()
        and uba.branch_id = yolo_models.branch_id
        and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
    )
  )
);

-- ---------------------------------------------------------------------------
-- Optional helper RPC: runtime bootstrap (active config + active model)
-- ---------------------------------------------------------------------------
create or replace function public.get_runtime_bootstrap(p_branch_id bigint)
returns jsonb
language sql
stable
as $$
select jsonb_build_object(
  'config', (
    select pc.config_data
    from public.program_configs pc
    where pc.branch_id = p_branch_id
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
    where (ym.branch_id = p_branch_id or ym.branch_id is null)
      and ym.is_active = true
      and ym.is_deleted = false
    order by case when ym.branch_id = p_branch_id then 0 else 1 end, ym.created_at desc
    limit 1
  )
);
$$;

commit;

-- ---------------------------------------------------------------------------
-- Optional manual snippets (reference only)
-- ---------------------------------------------------------------------------
-- List active config for a branch
-- select * from public.program_configs
-- where branch_id = 1 and config_name = 'default' and is_active = true;
--
-- List models for a branch + global
-- select * from public.yolo_models
-- where (branch_id = 1 or branch_id is null) and is_deleted = false
-- order by is_active desc, created_at desc;
