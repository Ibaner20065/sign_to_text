-- AuraCare Supabase Setup Script
-- Run this in the Supabase SQL Editor to enable the AI Orchestrator's query capability.

-- Part 1: SQL Execution RPC
-- This allows the backend (using service_role) to execute dynamic clinical queries.
create or replace function execute_sql(query text)
returns json
language plpgsql
security definer -- Critical: Executes with permissions of the creator (owner)
as $$
declare
  result json;
begin
  execute query into result;
  return result;
exception
  when others then
    return json_build_object('error', SQLERRM);
end;
$$;

-- Part 11: Heatmap Support (Example)
-- Function to get incident density for response time heatmaps
create or replace function get_incident_heatmap()
returns table (lat float, lon float, intensity int)
language sql
security definer
as $$
  select a.lat, a.lon, i.response_time_sec as intensity
  from incidents i
  join ambulances a on i.ambulance_id = a.id;
$$;
