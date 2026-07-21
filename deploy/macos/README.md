# macOS deploy — LaunchAgents

Local automation (macOS `launchd`) that gets the Mac ready each morning so the `trading_predict`
retraining DAGs run without intervention:

| File | What it does | Local time |
|---|---|---|
| `ensure-docker-up.sh` | Opens Docker Desktop if needed, waits for the daemon, and brings up the stack with `docker compose --profile all up -d`. Idempotent. | — |
| `com.trading.ensure-docker.plist` | Triggers the script above. | 04:51 |
| `com.trading.morning-caffeinate.plist` | Keeps the Mac awake for 90 minutes (`caffeinate`) to cover retraining plus `daily_report`. | 04:52 |

Both fire 1–2 minutes after the scheduled `pmset` wake (04:50), so the machine is already awake
when they start.

## Prerequisites

- **Docker Desktop** installed.
- A **scheduled wake** via `pmset`, so the Mac is up before the agents fire:
  ```bash
  sudo pmset repeat wake MTWRFSU 04:50:00
  ```

## Installation

1. Copy the script to a stable local path (outside iCloud) and make it executable:
   ```bash
   sudo cp ensure-docker-up.sh /usr/local/bin/ensure-docker-up.sh
   sudo chmod +x /usr/local/bin/ensure-docker-up.sh
   ```

   The script **requires** the `TRADING_PREDICT_DIR` variable and fails without it. Since
   `launchd` runs with a minimal environment, it's defined inside the `ensure-docker` `.plist`,
   which ships with a placeholder — replace the value with your real path before copying:
   ```xml
   <key>EnvironmentVariables</key>
   <dict>
       <key>TRADING_PREDICT_DIR</key>
       <string>/path/to/your/trading_predict</string>
   </dict>
   ```

2. Copy the plists to `~/Library/LaunchAgents/` and load them:
   ```bash
   cp com.trading.ensure-docker.plist      ~/Library/LaunchAgents/
   cp com.trading.morning-caffeinate.plist ~/Library/LaunchAgents/

   launchctl load ~/Library/LaunchAgents/com.trading.ensure-docker.plist
   launchctl load ~/Library/LaunchAgents/com.trading.morning-caffeinate.plist
   ```

## Verifying

```bash
launchctl list | grep com.trading          # both labels should appear
launchctl start com.trading.ensure-docker  # manual test trigger
tail -f /tmp/ensure-docker-up.log          # follow the script log
```

## Uninstalling

```bash
launchctl unload ~/Library/LaunchAgents/com.trading.ensure-docker.plist
launchctl unload ~/Library/LaunchAgents/com.trading.morning-caffeinate.plist
rm ~/Library/LaunchAgents/com.trading.ensure-docker.plist
rm ~/Library/LaunchAgents/com.trading.morning-caffeinate.plist
sudo rm /usr/local/bin/ensure-docker-up.sh
```

## Notes

- Logs go to `/tmp/ensure-docker-up.log` and `/tmp/morning-caffeinate.log`. They're cleared on
  every reboot, which is enough for same-day diagnosis.
- `caffeinate -s` only prevents system sleep while the charger is connected.
