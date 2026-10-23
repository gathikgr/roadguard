-- Keep the SECURITY DEFINER helper outside the exposed public schema.
create schema if not exists private;
revoke all on schema private from public, anon;
grant usage on schema private to authenticated;

create or replace function private.is_admin()
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select auth.uid() is not null
    and exists (
      select 1
      from public.profiles
      where id = auth.uid() and role = 'admin'
    )
$$;

revoke all on function private.is_admin() from public, anon;
grant execute on function private.is_admin() to authenticated;

drop policy "profiles self or admin" on public.profiles;
create policy "profiles self" on public.profiles
  for select to authenticated using (id = (select auth.uid()));
create policy "admins view profiles" on public.profiles
  for select to authenticated using ((select private.is_admin()));

drop policy "published incidents public" on public.incidents;
create policy "published incidents public" on public.incidents
  for select to anon, authenticated using (status = 'PUBLISHED');
create policy "admins view incidents" on public.incidents
  for select to authenticated using ((select private.is_admin()));
drop policy "admins manage incidents" on public.incidents;
create policy "admins manage incidents" on public.incidents
  for update to authenticated using ((select private.is_admin()))
  with check ((select private.is_admin()));

drop policy "users create reports" on public.user_reports;
create policy "users create reports" on public.user_reports
  for insert to authenticated
  with check ((select auth.uid()) is not null and user_id = (select auth.uid()));
drop policy "users own reports admin all" on public.user_reports;
create policy "users own reports" on public.user_reports
  for select to authenticated using (user_id = (select auth.uid()));
create policy "admins view all reports" on public.user_reports
  for select to authenticated using ((select private.is_admin()));

drop policy "admins manage source data" on public.video_sources;
create policy "admins manage source data" on public.video_sources
  for all to authenticated using ((select private.is_admin()))
  with check ((select private.is_admin()));
drop policy "admins manage jobs" on public.processing_jobs;
create policy "admins manage jobs" on public.processing_jobs
  for all to authenticated using ((select private.is_admin()))
  with check ((select private.is_admin()));

drop policy "admins view detections" on public.detections;
create policy "admins view detections" on public.detections
  for select to authenticated using ((select private.is_admin()));
drop policy "admins view reviews" on public.incident_reviews;
create policy "admins view reviews" on public.incident_reviews
  for select to authenticated using ((select private.is_admin()));
drop policy "admins view evidence" on public.incident_evidence;
create policy "admins view evidence" on public.incident_evidence
  for select to authenticated using ((select private.is_admin()));
drop policy "admins view context" on public.context_logs;
create policy "admins view context" on public.context_logs
  for select to authenticated using ((select private.is_admin()));
drop policy "admins view risk zones" on public.risk_zones;
create policy "admins view risk zones" on public.risk_zones
  for select to authenticated using ((select private.is_admin()));
drop policy "admins view automation events" on public.automation_events;
create policy "admins view automation events" on public.automation_events
  for select to authenticated using ((select private.is_admin()));
drop policy "admins view audit" on public.audit_logs;
create policy "admins view audit" on public.audit_logs
  for select to authenticated using ((select private.is_admin()));

drop function public.is_admin();
