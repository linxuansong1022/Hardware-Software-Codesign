# Face Frontend

React + Vite dashboard for the ESP32 face recognition demo.

## Run the frontend

```bash
cd /Users/cherie/dtu-project/Hardware-Software-Codesign/face/frontend
npm install
npm run dev
```

Open the Vite URL in Chrome or Edge.

## Flash the updated ESP32 firmware

The frontend now expects the ESP32 firmware to stream both recognition metadata and raw camera frames over USB serial.

```bash
cd /Users/cherie/dtu-project/Hardware-Software-Codesign/face/esp32
source /Users/cherie/esp/esp-idf/export.sh
idf.py flash
```

## Demo flow

1. Flash the ESP32 firmware.
2. Use `idf.py monitor` normally for text logs while debugging.
3. Close `idf.py monitor` when you want the browser to take over the port.
4. Start the frontend and open it in Chrome or Edge.
5. Click `Connect ESP32`.
6. The page sends `START_STREAM` and then renders the ESP32 camera feed and recognition results.

## Protocol

For every inference loop the device sends:

- `===METRICS===` + one line of JSON recognition data
- `===FRAME===` + one JSON header line (`length` + `crc32`) + one 320x240 RGB565 frame

The firmware stays in normal log mode until it receives `START_STREAM\n` over Web Serial.
