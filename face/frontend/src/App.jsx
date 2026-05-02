import { useEffect, useMemo, useRef, useState } from 'react';

const FRAME_WIDTH = 320;
const FRAME_HEIGHT = 240;
const FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 2;
const MAX_LOGS = 80;
const METRICS_PREAMBLE = new TextEncoder().encode('\n===METRICS===\n');
const FRAME_PREAMBLE = new TextEncoder().encode('\n===FRAME===\n');
const START_STREAM_COMMAND = 'START_STREAM\n';

const initialMetrics = {
  frame: 'WAITING',
  frameConfidence: 0,
  vote: 'WAITING',
  voteCount: 0,
  voteWindow: 5,
  scores: { A: 0, B: 0, C: 0 },
};

function prettyName(name) {
  if (!name) return 'Unknown';
  if (name === 'UNKNOWN' || name === 'WAITING') return name;
  return name.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function concatUint8(a, b) {
  const merged = new Uint8Array(a.length + b.length);
  merged.set(a, 0);
  merged.set(b, a.length);
  return merged;
}

function indexOfSequence(buffer, sequence) {
  if (sequence.length === 0 || buffer.length < sequence.length) {
    return -1;
  }

  outer: for (let i = 0; i <= buffer.length - sequence.length; i += 1) {
    for (let j = 0; j < sequence.length; j += 1) {
      if (buffer[i + j] !== sequence[j]) {
        continue outer;
      }
    }
    return i;
  }

  return -1;
}

function decodeAscii(bytes) {
  return new TextDecoder().decode(bytes);
}

function rgb565ToImageData(frameBytes) {
  const imageData = new ImageData(FRAME_WIDTH, FRAME_HEIGHT);
  const rgba = imageData.data;

  for (let src = 0, dst = 0; src < frameBytes.length; src += 2, dst += 4) {
    const byte1 = frameBytes[src];
    const byte2 = frameBytes[src + 1];
    rgba[dst] = byte1 & 0xf8;
    rgba[dst + 1] = ((byte1 & 0x07) << 5) | ((byte2 & 0xe0) >> 3);
    rgba[dst + 2] = (byte2 & 0x1f) << 3;
    rgba[dst + 3] = 255;
  }

  return imageData;
}

export default function App() {
  const [serialSupported] = useState(() => 'serial' in navigator);
  const [port, setPort] = useState(null);
  const [reader, setReader] = useState(null);
  const [connecting, setConnecting] = useState(false);
  const [serialStatus, setSerialStatus] = useState('Disconnected');
  const [metrics, setMetrics] = useState(initialMetrics);
  const [logs, setLogs] = useState([]);
  const [errorMessage, setErrorMessage] = useState('');
  const [frameCounter, setFrameCounter] = useState(0);
  const [frameReceived, setFrameReceived] = useState(false);

  const canvasRef = useRef(null);
  const canvasContextRef = useRef(null);
  const serialBufferRef = useRef(new Uint8Array(0));
  const parserStateRef = useRef('seek');
  const readLoopActiveRef = useRef(false);
  const startStreamTimerRef = useRef(null);
  const frameReceivedRef = useRef(false);

  const accessGranted = useMemo(() => metrics.vote !== 'UNKNOWN' && metrics.vote !== 'WAITING', [metrics.vote]);

  useEffect(() => {
    return () => {
      if (startStreamTimerRef.current) {
        clearInterval(startStreamTimerRef.current);
      }
      if (reader) {
        reader.cancel().catch(() => {});
      }
      if (port) {
        port.close().catch(() => {});
      }
    };
  }, [reader, port]);

  useEffect(() => {
    if (canvasRef.current && !canvasContextRef.current) {
      canvasContextRef.current = canvasRef.current.getContext('2d');
    }
  }, []);

  function pushLog(line) {
    const normalized = line.trim();
    if (!normalized) return;
    setLogs((current) => [normalized, ...current].slice(0, MAX_LOGS));
  }

  async function sendStartStream(activePort) {
    if (!activePort?.writable) return;
    try {
      const writer = activePort.writable.getWriter();
      await writer.write(new TextEncoder().encode(START_STREAM_COMMAND));
      writer.releaseLock();
    } catch (error) {
      setErrorMessage(error.message || 'Failed to request START_STREAM.');
    }
  }

  function renderFrame(frameBytes) {
    if (!canvasContextRef.current) return;
    const imageData = rgb565ToImageData(frameBytes);
    canvasContextRef.current.putImageData(imageData, 0, 0);
    frameReceivedRef.current = true;
    setFrameReceived(true);
    setFrameCounter((count) => count + 1);
  }

  function consumeTextPrefix(prefixBytes) {
    const prefixText = decodeAscii(prefixBytes);
    if (prefixText.includes('===READY===') || prefixText.includes('START_STREAM')) {
      setSerialStatus('ESP32 ready, requesting stream');
      sendStartStream(port);
    }
    prefixText
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .forEach(pushLog);
  }

  function parseSerialBuffer() {
    while (true) {
      const buffer = serialBufferRef.current;

      if (parserStateRef.current === 'metrics') {
        const newlineIndex = buffer.indexOf(0x0a);
        if (newlineIndex === -1) return;

        const lineBytes = buffer.slice(0, newlineIndex);
        serialBufferRef.current = buffer.slice(newlineIndex + 1);
        parserStateRef.current = 'seek';

        const line = decodeAscii(lineBytes).trim();
        if (!line) {
          continue;
        }

        try {
          const parsed = JSON.parse(line);
          setMetrics(parsed);
        } catch {
          pushLog(`Invalid metrics JSON: ${line}`);
        }
        continue;
      }

      if (parserStateRef.current === 'frame') {
        if (buffer.length < FRAME_SIZE) return;

        const frameBytes = buffer.slice(0, FRAME_SIZE);
        serialBufferRef.current = buffer.slice(FRAME_SIZE);
        parserStateRef.current = 'seek';
        renderFrame(frameBytes);
        continue;
      }

      const metricsIndex = indexOfSequence(buffer, METRICS_PREAMBLE);
      const frameIndex = indexOfSequence(buffer, FRAME_PREAMBLE);
      const candidateIndexes = [metricsIndex, frameIndex].filter((value) => value >= 0);

      if (candidateIndexes.length === 0) {
        if (buffer.length > 2048) {
          consumeTextPrefix(buffer.slice(0, buffer.length - 256));
          serialBufferRef.current = buffer.slice(buffer.length - 256);
        }
        return;
      }

      const nextIndex = Math.min(...candidateIndexes);
      if (nextIndex > 0) {
        consumeTextPrefix(buffer.slice(0, nextIndex));
      }

      if (metricsIndex === nextIndex) {
        serialBufferRef.current = buffer.slice(nextIndex + METRICS_PREAMBLE.length);
        parserStateRef.current = 'metrics';
      } else {
        serialBufferRef.current = buffer.slice(nextIndex + FRAME_PREAMBLE.length);
        parserStateRef.current = 'frame';
      }
    }
  }

  async function disconnectSerial() {
    readLoopActiveRef.current = false;
    if (startStreamTimerRef.current) {
      clearInterval(startStreamTimerRef.current);
      startStreamTimerRef.current = null;
    }
    try {
      if (reader) {
        await reader.cancel();
        setReader(null);
      }
      if (port) {
        await port.close();
        setPort(null);
      }
    } catch (error) {
      setErrorMessage(error.message || 'Failed to disconnect serial port.');
    } finally {
      setSerialStatus('Disconnected');
    }
  }

  async function connectSerial() {
    if (!serialSupported) {
      setErrorMessage('This browser does not support Web Serial. Please use Chrome or Edge on desktop.');
      return;
    }

    setConnecting(true);
    setErrorMessage('');
    setLogs([]);
    setMetrics(initialMetrics);
    setFrameCounter(0);
    setFrameReceived(false);
    frameReceivedRef.current = false;
    serialBufferRef.current = new Uint8Array(0);
    parserStateRef.current = 'seek';

    try {
      const nextPort = await navigator.serial.requestPort();
      await nextPort.open({ baudRate: 921600 });
      setPort(nextPort);
      setSerialStatus('Connected, waiting for ESP32 ready signal');

      await sendStartStream(nextPort);
      startStreamTimerRef.current = setInterval(() => {
        if (!readLoopActiveRef.current || frameReceivedRef.current) {
          return;
        }
        sendStartStream(nextPort);
      }, 1200);

      readLoopActiveRef.current = true;
      while (nextPort.readable && readLoopActiveRef.current) {
        const nextReader = nextPort.readable.getReader();
        setReader(nextReader);

        try {
          setSerialStatus('Streaming from ESP32 hardware');
          while (readLoopActiveRef.current) {
            const { value, done } = await nextReader.read();
            if (done) break;
            if (!value || value.length === 0) continue;
            serialBufferRef.current = concatUint8(serialBufferRef.current, value);
            parseSerialBuffer();
          }
        } finally {
          nextReader.releaseLock();
        }
      }
    } catch (error) {
      setErrorMessage(error.message || 'Failed to connect to serial device.');
      setSerialStatus('Disconnected');
    } finally {
      if (startStreamTimerRef.current) {
        clearInterval(startStreamTimerRef.current);
        startStreamTimerRef.current = null;
      }
      setConnecting(false);
    }
  }

  const scoreCards = [
    { label: 'Person A', value: metrics.scores?.A || 0, color: 'var(--accent-a)' },
    { label: 'Person B', value: metrics.scores?.B || 0, color: 'var(--accent-b)' },
    { label: 'Person C', value: metrics.scores?.C || 0, color: 'var(--accent-c)' },
  ];

  return (
    <div className="app-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <header className="topbar">
        <div>
          <p className="eyebrow">ESP32 Hardware Stream</p>
          <h1>Face Access Dashboard</h1>
        </div>
        <div className="status-cluster">
          <span className={`pill ${accessGranted ? 'pill-success' : 'pill-danger'}`}>
            {accessGranted ? `ACCESS: ${prettyName(metrics.vote)}` : 'ACCESS: LOCKED'}
          </span>
          <span className="pill pill-neutral">{serialStatus}</span>
          <span className="pill pill-neutral">Frames: {frameCounter}</span>
        </div>
      </header>

      <main className="layout-grid">
        <section className="camera-panel panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Hardware video</p>
              <h2>ESP32 camera feed</h2>
            </div>
            <button
              className={port ? 'ghost-button' : 'primary-button'}
              onClick={port ? disconnectSerial : connectSerial}
              disabled={connecting}
            >
              {connecting ? 'Connecting...' : port ? 'Disconnect serial' : 'Connect ESP32'}
            </button>
          </div>

          <div className={`camera-stage ${accessGranted ? 'camera-stage-allow' : 'camera-stage-deny'}`}>
            <canvas ref={canvasRef} width={FRAME_WIDTH} height={FRAME_HEIGHT} className="hardware-canvas" />
            {!frameReceived && (
              <div className="camera-placeholder">
                <p>The large screen will show frames coming from the ESP32 camera.</p>
                <span>Click Connect ESP32 after flashing the updated firmware and closing `idf.py monitor`.</span>
              </div>
            )}
            <div className="camera-overlay">
              <div>
                <span className="overlay-label">Current frame</span>
                <strong>{prettyName(metrics.frame)}</strong>
              </div>
              <div>
                <span className="overlay-label">Confidence</span>
                <strong>{formatPercent(metrics.frameConfidence)}</strong>
              </div>
            </div>
          </div>
        </section>

        <section className="side-panel">
          <div className="panel metrics-panel">
            <div className="panel-header compact-header">
              <div>
                <p className="section-kicker">Recognition</p>
                <h2>Realtime hardware result</h2>
              </div>
            </div>

            <div className="hero-metric-row">
              <article className="hero-card">
                <span>Vote result</span>
                <strong>{prettyName(metrics.vote)}</strong>
                <small>{metrics.voteCount}/{metrics.voteWindow} frames passed</small>
              </article>
              <article className="hero-card muted-card">
                <span>Frame result</span>
                <strong>{prettyName(metrics.frame)}</strong>
                <small>{formatPercent(metrics.frameConfidence)} confidence</small>
              </article>
            </div>

            <div className="score-list">
              {scoreCards.map((score) => (
                <div key={score.label} className="score-item">
                  <div className="score-label-row">
                    <span>{score.label}</span>
                    <span>{formatPercent(score.value)}</span>
                  </div>
                  <div className="score-bar-track">
                    <div className="score-bar-fill" style={{ width: `${score.value * 100}%`, background: score.color }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel raw-panel">
            <div className="panel-header compact-header">
              <div>
                <p className="section-kicker">Serial feed</p>
                <h2>Hardware stream notes</h2>
              </div>
            </div>

            <div className="info-grid">
              <div>
                <span className="mini-label">Protocol</span>
                <p>ESP32 now sends one JSON metrics block plus one 320x240 RGB565 frame for each inference loop.</p>
              </div>
              <div>
                <span className="mini-label">Port usage</span>
                <p>Close `idf.py monitor` before opening the webpage, otherwise the USB serial port stays busy.</p>
              </div>
            </div>

            {errorMessage && <div className="error-box">{errorMessage}</div>}

            <div className="log-console">
              {logs.length === 0 ? (
                <p className="log-empty">Boot logs and parser messages from the ESP32 stream will appear here.</p>
              ) : (
                logs.map((line, index) => (
                  <div key={`${line}-${index}`} className="log-line">{line}</div>
                ))
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
