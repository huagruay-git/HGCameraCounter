-- HG Camera Counter - Supabase migration
-- Date: 2026-02-24
-- Purpose:
--   Add transactional RPC to set one program_config row active and deactivate
--   sibling rows in same (branch_id, config_name) scope.

begin;

create or replace function public.set_active_program_config(p_config_id uuid)
returns public.program_configs
language plpgsql
security invoker
as $$
declare
  v_row public.program_configs%rowtype;
begin
  select *
  into v_row
  from public.program_configs
  where id = p_config_id;

  if not found then
    raise exception 'program_config not found: %', p_config_id;
  end if;

  -- RLS applies on these updates (security invoker).
  update public.program_configs
  set is_active = false
  where branch_id = v_row.branch_id
    and config_name = v_row.config_name
    and is_active = true;

  update public.program_configs
  set is_active = true
  where id = p_config_id
  returning * into v_row;

  return v_row;
end;
$$;

grant execute on function public.set_active_program_config(uuid) to authenticated;

commit;
