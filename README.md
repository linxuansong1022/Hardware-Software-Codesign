# Open-Set Face Recognition on ESP32-S3

This repository contains the final face-recognition project for DTU 02214 Hardware/Software Codesign.

The system recognizes three enrolled people and rejects unknown faces using a compact INT8 CNN deployed on an ESP32-S3 Sense board.

## Project Structure

- `face/esp32/` - ESP32-S3 firmware, camera pipeline, TFLite Micro inference, open-set rejection, and serial/Web Serial output.
- `face/frontend/` - Vite/React dashboard for displaying the ESP32 camera stream and recognition result.
- `face/python/` - Python utilities used by the face project.
- `face/*.md` - deployment notes, rejection notes, and generated evaluation summaries.
- `ml_share/ml_scripts/` - training, quantization, evaluation, and deployment-metrics scripts.
- `ml_share/models/` - trained/exported model artifacts used by the deployment pipeline.
- `ml_share/data/metadata/` - metadata and split files used by the ML scripts.

Raw face images are intentionally not included in the final branch.

## Main Results

The deployed INT8 model keeps high closed-set accuracy while adding conservative unknown rejection:

- INT8 closed-set accuracy: `98.70%`
- Softmax-only unknown false accept: `45/259`
- Two-stage rejection unknown false accept: `11/259`
- TFLite model size: `29.7 KB`
- ESP32 firmware app binary size: about `395-418 KB`, depending on serial streaming support

See:

- `face/int8_deployment_eval.md`
- `face/deployment_metrics.md`
- `face/OPENSET_REJECTION_NOTE.md`
- `face/DEPLOYMENT_OPTIMIZATION_PLAN.md`
- `face/DEPLOYMENT_WORK_EXECUTION_PLAN.md`

## Build ESP32 Firmware

```bash
cd face/esp32
source /Users/songlinxuan/esp/esp-idf/export.sh
export PATH=/Users/songlinxuan/.espressif/python_env/idf6.1_py3.14_env/bin:$PATH
idf.py build
```

Flash and monitor:

```bash
idf.py -p /dev/cu.usbmodemXXXX flash monitor
```

## Run Frontend

```bash
cd face/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 4174
```

The frontend connects to the ESP32 through Web Serial.
