# macOS deploy — LaunchAgents

Automatización local (macOS `launchd`) que deja la Mac lista cada mañana para
que los DAGs de retrain de `trading_predict` corran sin intervención:

| Archivo | Qué hace | Hora local |
|---|---|---|
| `ensure-docker-up.sh` | Abre Docker Desktop (si hace falta), espera al daemon y levanta el stack `docker compose --profile all up -d`. Idempotente. | — |
| `com.trading.ensure-docker.plist` | Dispara el script de arriba. | 04:51 |
| `com.trading.morning-caffeinate.plist` | Mantiene la Mac despierta 90 min (`caffeinate`) para cubrir retrain + `daily_report`. | 04:52 |

Estos disparos van 1–2 min después del wake programado con `pmset` (04:50), de
modo que la máquina ya esté despierta cuando arrancan.

## Requisitos previos

- **Docker Desktop** instalado.
- **Wake programado** con `pmset` para que la Mac despierte antes de los agentes:
  ```bash
  sudo pmset repeat wake MTWRFSU 04:50:00
  ```

## Instalación

1. Copiar el script a una ruta local estable (fuera de iCloud) y hacerlo ejecutable:
   ```bash
   sudo cp ensure-docker-up.sh /usr/local/bin/ensure-docker-up.sh
   sudo chmod +x /usr/local/bin/ensure-docker-up.sh
   ```

   El script **requiere** la variable `TRADING_PREDICT_DIR` (falla si no está).
   Como `launchd` corre con un entorno mínimo, se define dentro del `.plist` de
   `ensure-docker`, que ya trae la clave con un placeholder — reemplazá el valor
   por tu ruta real antes de copiarlo:
   ```xml
   <key>EnvironmentVariables</key>
   <dict>
       <key>TRADING_PREDICT_DIR</key>
       <string>/ruta/a/tu/trading_predict</string>
   </dict>
   ```

2. Copiar los plists a `~/Library/LaunchAgents/` y cargarlos:
   ```bash
   cp com.trading.ensure-docker.plist    ~/Library/LaunchAgents/
   cp com.trading.morning-caffeinate.plist ~/Library/LaunchAgents/

   launchctl load ~/Library/LaunchAgents/com.trading.ensure-docker.plist
   launchctl load ~/Library/LaunchAgents/com.trading.morning-caffeinate.plist
   ```

## Verificar

```bash
launchctl list | grep com.trading                 # deben aparecer los dos labels
launchctl start com.trading.ensure-docker          # disparo manual de prueba
tail -f /tmp/ensure-docker-up.log                  # seguir el log del script
```

## Desinstalar

```bash
launchctl unload ~/Library/LaunchAgents/com.trading.ensure-docker.plist
launchctl unload ~/Library/LaunchAgents/com.trading.morning-caffeinate.plist
rm ~/Library/LaunchAgents/com.trading.ensure-docker.plist
rm ~/Library/LaunchAgents/com.trading.morning-caffeinate.plist
sudo rm /usr/local/bin/ensure-docker-up.sh
```

## Notas

- Los logs van a `/tmp/ensure-docker-up.log` y `/tmp/morning-caffeinate.log`
  (se limpian en cada reinicio; suficiente para diagnóstico del día).
- `caffeinate -s` solo evita el system sleep con el cargador conectado.
