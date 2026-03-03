# Fixes Log
> This file tracks all issues encountered and how they were resolved.
> Read this file first before debugging any issue to avoid repeating work.

---

## Fix #1 — Backend & Site Stop Working After Closing VS Code / Idle Time
**Date:** 2026-02-26  
**Severity:** Critical  
**Affected:** Entire app (backend + frontend)

### Problem
Every time the user closes VS Code or leaves the site idle, the backend on Render (free tier) spins down after 15 minutes of inactivity. The Vercel frontend stays up but all API calls fail because the backend is dead — making the whole app appear broken.

### Root Cause
Render's **free tier** automatically puts web services to sleep after 15 minutes of no incoming requests. Cold starts take 30-60 seconds, during which requests fail.

### Fix Applied
1. **Added `/health` endpoint** in `web_app/backend/main.py` (around line 182):
   ```python
   @app.get("/health")
   async def health_check():
       return {"status": "ok"}
   ```
2. **Set up external uptime monitor** (UptimeRobot, free) to ping `https://<app>.onrender.com/health` every 5 minutes, preventing Render from ever spinning down.

### How to Set Up UptimeRobot (if not done yet)
1. Go to https://uptimerobot.com → Sign up free
2. Add New Monitor → HTTP(s)
3. URL: `https://your-render-app.onrender.com/health`
4. Interval: 5 minutes
5. Save

### Files Changed
- `web_app/backend/main.py` — Added `/health` endpoint
- `render.yaml` — Kept on free plan (no `plan: starter`)

### Notes
- Vercel frontend hosting is always-on for static sites — no fix needed there.
- Alternative free ping services: cron-job.org, Better Stack, Freshping
- If budget allows later, Render Starter plan ($7/mo) eliminates this entirely.

---
