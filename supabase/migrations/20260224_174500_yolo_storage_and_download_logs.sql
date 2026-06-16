-- HG Camera Counter - Supabase migration
-- Date: 2026-02-24
-- Purpose:
--   1) Create storage bucket for YOLO model files
--   2) Add storage policies (using public.user_branch_access roles)
--   3) Add model_download_logs table + RLS
--
-- Notes:
--   - Requires previous migration that creates public.yolo_models.
--   - Storage objects are protected (public = false).
--   - Global model files (branch_id null) are manageable by COMPANY_ADMIN only.

begin;

-- ---------------------------------------------------------------------------
-- Storage bucket: yolo-models
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public)
values ('yolo-models', 'yolo-models', false)
on conflict (id) do nothing;

-- ---------------------------------------------------------------------------
-- Storage policies for bucket yolo-models
-- Path convention (recommended):
--   global/<version>/best.pt
--   branch/<branch_id>/<version>/best.pt
--
-- Access model:
--   - Read: any authenticated user with any branch access role
--   - Insert/Update/Delete branch files: BRANCH_ADMIN or COMPANY_ADMIN for that branch
--   - Insert/Update/Delete global files: COMPANY_ADMIN only
-- ---------------------------------------------------------------------------

drop policy if exists "read yolo model files by branch access" on storage.objects;
create policy "read yolo model files by branch access"
on storage.objects
for select
to authenticated
using (
  bucket_id = 'yolo-models'
  and (
    -- global/*
    (
      (storage.foldername(name))[1] = 'global'
      and exists (
        select 1
        from public.user_branch_access uba
        where uba.user_id = auth.uid()
          and uba.role in ('STAFF','BARBER','BRANCH_STAFF','BRANCH_ADMIN','COMPANY_ADMIN')
      )
    )
    or
    -- branch/<branch_id>/*
    (
      (storage.foldername(name))[1] = 'branch'
      and (storage.foldername(name))[2] ~ '^[0-9]+$'
      and exists (
        select 1
        from public.user_branch_access uba
        where uba.user_id = auth.uid()
          and uba.branch_id = ((storage.foldername(name))[2])::bigint
          and uba.role in ('STAFF','BARBER','BRANCH_STAFF','BRANCH_ADMIN','COMPANY_ADMIN')
      )
    )
  )
);

drop policy if exists "insert yolo model files by admin roles" on storage.objects;
create policy "insert yolo model files by admin roles"
on storage.objects
for insert
to authenticated
with check (
  bucket_id = 'yolo-models'
  and (
    -- global/*
    (
      (storage.foldername(name))[1] = 'global'
      and exists (
        select 1
        from public.user_branch_access uba
        where uba.user_id = auth.uid()
          and uba.role = 'COMPANY_ADMIN'
      )
    )
    or
    -- branch/<branch_id>/*
    (
      (storage.foldername(name))[1] = 'branch'
      and (storage.foldername(name))[2] ~ '^[0-9]+$'
      and exists (
        select 1
        from public.user_branch_access uba
        where uba.user_id = auth.uid()
          and uba.branch_id = ((storage.foldername(name))[2])::bigint
          and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
      )
    )
  )
);

drop policy if exists "update yolo model files by admin roles" on storage.objects;
create policy "update yolo model files by admin roles"
on storage.objects
for update
to authenticated
using (
  bucket_id = 'yolo-models'
  and (
    (
      (storage.foldername(name))[1] = 'global'
      and exists (
        select 1
        from public.user_branch_access uba
        where uba.user_id = auth.uid()
          and uba.role = 'COMPANY_ADMIN'
      )
    )
    or
    (
      (storage.foldername(name))[1] = 'branch'
      and (storage.foldername(name))[2] ~ '^[0-9]+$'
      and exists (
        select 1
        from public.user_branch_access uba
        where uba.user_id = auth.uid()
          and uba.branch_id = ((storage.foldername(name))[2])::bigint
          and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
      )
    )
  )
)
with check (
  bucket_id = 'yolo-models'
  and (
    (
      (storage.foldername(name))[1] = 'global'
      and exists (
        select 1
        from public.user_branch_access uba
        where uba.user_id = auth.uid()
          and uba.role = 'COMPANY_ADMIN'
      )
    )
    or
    (
      (storage.foldername(name))[1] = 'branch'
      and (storage.foldername(name))[2] ~ '^[0-9]+$'
      and exists (
        select 1
        from public.user_branch_access uba
        where uba.user_id = auth.uid()
          and uba.branch_id = ((storage.foldername(name))[2])::bigint
          and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
      )
    )
  )
);

drop policy if exists "delete yolo model files by admin roles" on storage.objects;
create policy "delete yolo model files by admin roles"
on storage.objects
for delete
to authenticated
using (
  bucket_id = 'yolo-models'
  and (
    (
      (storage.foldername(name))[1] = 'global'
      and exists (
        select 1
        from public.user_branch_access uba
        where uba.user_id = auth.uid()
          and uba.role = 'COMPANY_ADMIN'
      )
    )
    or
    (
      (storage.foldername(name))[1] = 'branch'
      and (storage.foldername(name))[2] ~ '^[0-9]+$'
      and exists (
        select 1
        from public.user_branch_access uba
        where uba.user_id = auth.uid()
          and uba.branch_id = ((storage.foldername(name))[2])::bigint
          and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
      )
    )
  )
);

-- ---------------------------------------------------------------------------
-- Table: public.model_download_logs
-- ---------------------------------------------------------------------------
create table if not exists public.model_download_logs (
  id bigserial primary key,
  model_id uuid not null references public.yolo_models(id) on delete cascade,
  branch_id bigint references public.branches(id) on delete set null,
  device_id text,
  machine_name text,
  app_version text,
  downloaded_by uuid references auth.users(id),
  downloaded_at timestamptz not null default now(),
  status text not null default 'success'
    check (status in ('success', 'failed')),
  detail text
);

create index if not exists idx_model_download_logs_model_id
  on public.model_download_logs(model_id);

create index if not exists idx_model_download_logs_branch_id
  on public.model_download_logs(branch_id);

create index if not exists idx_model_download_logs_downloaded_by
  on public.model_download_logs(downloaded_by);

create index if not exists idx_model_download_logs_downloaded_at
  on public.model_download_logs(downloaded_at desc);

alter table public.model_download_logs enable row level security;

-- Read logs: branch access (or COMPANY_ADMIN for global/null branch logs)
drop policy if exists "read model download logs by branch access" on public.model_download_logs;
create policy "read model download logs by branch access"
on public.model_download_logs
for select
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
  exists (
    select 1
    from public.user_branch_access uba
    where uba.user_id = auth.uid()
      and uba.branch_id = model_download_logs.branch_id
      and uba.role in ('STAFF','BARBER','BRANCH_STAFF','BRANCH_ADMIN','COMPANY_ADMIN')
  )
);

-- Insert logs: any authenticated user with access to that branch, or COMPANY_ADMIN for global
drop policy if exists "insert model download logs by branch access" on public.model_download_logs;
create policy "insert model download logs by branch access"
on public.model_download_logs
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
  exists (
    select 1
    from public.user_branch_access uba
    where uba.user_id = auth.uid()
      and uba.branch_id = model_download_logs.branch_id
      and uba.role in ('STAFF','BARBER','BRANCH_STAFF','BRANCH_ADMIN','COMPANY_ADMIN')
  )
);

-- Optional: allow admins to update/delete logs (mostly for corrections)
drop policy if exists "update model download logs by admin roles" on public.model_download_logs;
create policy "update model download logs by admin roles"
on public.model_download_logs
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
  exists (
    select 1
    from public.user_branch_access uba
    where uba.user_id = auth.uid()
      and uba.branch_id = model_download_logs.branch_id
      and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
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
  exists (
    select 1
    from public.user_branch_access uba
    where uba.user_id = auth.uid()
      and uba.branch_id = model_download_logs.branch_id
      and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
  )
);

drop policy if exists "delete model download logs by admin roles" on public.model_download_logs;
create policy "delete model download logs by admin roles"
on public.model_download_logs
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
  exists (
    select 1
    from public.user_branch_access uba
    where uba.user_id = auth.uid()
      and uba.branch_id = model_download_logs.branch_id
      and uba.role in ('BRANCH_ADMIN','COMPANY_ADMIN')
  )
);

commit;

-- ---------------------------------------------------------------------------
-- Optional reference queries
-- ---------------------------------------------------------------------------
-- list files in yolo-models bucket (metadata)
-- select * from storage.objects where bucket_id = 'yolo-models' order by created_at desc;
--
-- recent download logs
-- select * from public.model_download_logs order by downloaded_at desc limit 100;
