const { app, BrowserWindow, dialog, shell } = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const https = require("node:https");
const net = require("node:net");
const path = require("node:path");

const HOST = "127.0.0.1";
const PORT = Number(process.env.DK_PORT || "8000");
const LLAMA_CPP_PORT = Number(process.env.LLAMA_CPP_PORT || "8081");
const APP_URL = `http://${HOST}:${PORT}`;
const LLAMA_CPP_BASE_URL = `http://${HOST}:${LLAMA_CPP_PORT}/v1`;

let mainWindow;
let setupWindow;
let backendProcess;
let llamaServerProcess;
let selectedLlamaModel;

const GRANITE_MODEL_SPECS = [
  {
    file: "granite-4.1-3b-Q4_K_M.gguf",
    alias: "granite4.1:3b-q4_K_M",
    label: "Granite 4.1 3B Q4_K_M",
    dirs: ["granite4.1-3b", "granite4-3b"],
    url: "https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF/resolve/main/granite-4.1-3b-Q4_K_M.gguf"
  }
];

function getRepoRoot() {
  return app.isPackaged ? process.resourcesPath : path.resolve(__dirname, "..");
}

function getBackendExecutable() {
  const executable = process.platform === "win32"
    ? "DeployableKnowledgeBackend.exe"
    : "DeployableKnowledgeBackend";

  return path.join(process.resourcesPath, "backend", executable);
}

function getLlamaServerExecutable() {
  const executable = process.platform === "win32" ? "llama-server.exe" : "llama-server";
  const candidates = [
    path.join(process.resourcesPath, "llama.cpp", "win-x64", executable),
    path.join(process.resourcesPath, "llama.cpp", executable),
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return findBundledFile(path.join(process.resourcesPath, "llama.cpp"), executable) || candidates[0];
}

function getGraniteModelInfo() {
  const userModelsDir = path.join(app.getPath("userData"), "models");

  for (const spec of GRANITE_MODEL_SPECS) {
    const candidates = [
      ...spec.dirs.map((dir) => path.join(userModelsDir, dir, spec.file)),
      path.join(userModelsDir, spec.file),
      ...spec.dirs.map((dir) => path.join(process.resourcesPath, "models", dir, spec.file)),
      path.join(process.resourcesPath, "models", spec.file),
    ];

    for (const candidate of candidates) {
      if (fs.existsSync(candidate)) {
        return { ...spec, path: candidate };
      }
    }

    const nested = findBundledFile(path.join(process.resourcesPath, "models"), spec.file);
    if (nested) {
      return { ...spec, path: nested };
    }
  }

  const fallback = GRANITE_MODEL_SPECS[GRANITE_MODEL_SPECS.length - 1];
  return {
    ...fallback,
    path: path.join(userModelsDir, fallback.dirs[0], fallback.file)
  };
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 MB";
  }

  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function setupWindowHtml() {
  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, "Segoe UI", system-ui, sans-serif;
      background: #111827;
      color: #f9fafb;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #111827;
    }
    main {
      width: min(440px, calc(100vw - 48px));
    }
    h1 {
      margin: 0 0 10px;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }
    p {
      margin: 0 0 18px;
      color: #cbd5e1;
      font-size: 13px;
      line-height: 1.5;
    }
    progress {
      width: 100%;
      height: 14px;
      border: 0;
      border-radius: 7px;
      overflow: hidden;
      background: #1f2937;
    }
    progress::-webkit-progress-bar {
      background: #1f2937;
    }
    progress::-webkit-progress-value {
      background: #38bdf8;
    }
    progress::-moz-progress-bar {
      background: #38bdf8;
    }
    #status {
      margin-top: 12px;
      min-height: 18px;
      color: #e5e7eb;
      font-size: 12px;
    }
  </style>
</head>
<body>
  <main>
    <h1>Setting Up Local Model</h1>
    <p>Downloading Granite 4.1 3B for the bundled llama.cpp provider. This only happens once.</p>
    <progress id="progress"></progress>
    <div id="status">Preparing download...</div>
  </main>
</body>
</html>`;
}

async function showSetupWindow() {
  if (setupWindow) {
    return setupWindow;
  }

  setupWindow = new BrowserWindow({
    width: 520,
    height: 260,
    resizable: false,
    minimizable: false,
    maximizable: false,
    show: false,
    backgroundColor: "#111827",
    title: "Deployable Knowledge Setup",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  setupWindow.on("closed", () => {
    setupWindow = null;
  });

  await setupWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(setupWindowHtml())}`);
  setupWindow.show();
  return setupWindow;
}

function updateSetupProgress({ downloaded = 0, total = 0, status = "" }) {
  if (!setupWindow || setupWindow.isDestroyed()) {
    return;
  }

  const percent = total > 0 ? Math.min(100, Math.round((downloaded / total) * 100)) : null;
  const statusText =
    status ||
    (total > 0
      ? `${percent}% - ${formatBytes(downloaded)} of ${formatBytes(total)}`
      : `${formatBytes(downloaded)} downloaded`);

  setupWindow.webContents
    .executeJavaScript(
      `
      (() => {
        const progress = document.getElementById("progress");
        const status = document.getElementById("status");
        const percent = ${percent === null ? "null" : JSON.stringify(percent)};
        if (percent === null) {
          progress.removeAttribute("value");
        } else {
          progress.max = 100;
          progress.value = percent;
        }
        status.textContent = ${JSON.stringify(statusText)};
      })();
      `,
      true
    )
    .catch(() => {});
}

function closeSetupWindow() {
  if (setupWindow && !setupWindow.isDestroyed()) {
    setupWindow.close();
  }
  setupWindow = null;
}

function download(url, destination, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    fs.mkdirSync(path.dirname(destination), { recursive: true });

    const partial = `${destination}.part`;
    const request = https.get(url, { headers: { "User-Agent": "deployable-knowledge-builder" } }, (response) => {
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        response.resume();
        download(response.headers.location, destination).then(resolve, reject);
        return;
      }

      if (response.statusCode !== 200) {
        response.resume();
        reject(new Error(`GET ${url} returned ${response.statusCode}`));
        return;
      }

      const total = Number(response.headers["content-length"] || 0);
      let downloaded = 0;
      let lastProgressUpdate = 0;
      const file = fs.createWriteStream(partial);
      response.on("data", (chunk) => {
        downloaded += chunk.length;
        const now = Date.now();
        if (now - lastProgressUpdate > 250 || downloaded === total) {
          lastProgressUpdate = now;
          onProgress({ downloaded, total });
        }
      });
      response.pipe(file);
      file.on("finish", () => {
        file.close(() => {
          fs.rename(partial, destination, (error) => {
            if (error) {
              reject(error);
              return;
            }
            resolve();
          });
        });
      });
      file.on("error", reject);
    });

    request.on("error", reject);
  }).finally(() => {
    const partial = `${destination}.part`;
    if (fs.existsSync(partial)) {
      fs.rmSync(partial, { force: true });
    }
  });
}

async function ensureGraniteModel() {
  const modelInfo = getGraniteModelInfo();
  if (fs.existsSync(modelInfo.path)) {
    return modelInfo;
  }

  await showSetupWindow();
  updateSetupProgress({ status: "Starting download..." });

  try {
    await download(modelInfo.url, modelInfo.path, updateSetupProgress);
    updateSetupProgress({ downloaded: 1, total: 1, status: "Download complete. Starting app..." });
    await new Promise((resolve) => setTimeout(resolve, 500));
    return modelInfo;
  } finally {
    closeSetupWindow();
  }
}

function findBundledFile(startDir, fileName) {
  if (!fs.existsSync(startDir)) {
    return null;
  }

  for (const entry of fs.readdirSync(startDir, { withFileTypes: true })) {
    const fullPath = path.join(startDir, entry.name);
    if (entry.isFile() && entry.name === fileName) {
      return fullPath;
    }
    if (entry.isDirectory()) {
      const nested = findBundledFile(fullPath, fileName);
      if (nested) {
        return nested;
      }
    }
  }

  return null;
}

function describeResourceDirectory(startDir, depth = 2) {
  if (!fs.existsSync(startDir)) {
    return `${startDir} does not exist`;
  }

  const lines = [];

  function walk(dir, level) {
    if (level > depth || lines.length > 80) {
      return;
    }

    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      const relativePath = path.relative(startDir, fullPath) || entry.name;
      lines.push(`${entry.isDirectory() ? "[d]" : "[f]"} ${relativePath}`);
      if (entry.isDirectory()) {
        walk(fullPath, level + 1);
      }
      if (lines.length > 80) {
        return;
      }
    }
  }

  walk(startDir, 0);
  return lines.join("\n") || `${startDir} is empty`;
}

function getPythonCommand() {
  if (process.env.DK_PYTHON) {
    return { command: process.env.DK_PYTHON, args: [] };
  }

  const root = getRepoRoot();
  const venvPython = process.platform === "win32"
    ? path.join(root, "venv", "Scripts", "python.exe")
    : path.join(root, "venv", "bin", "python");

  if (fs.existsSync(venvPython)) {
    return { command: venvPython, args: [] };
  }

  return {
    command: process.platform === "win32" ? "python" : "python3",
    args: []
  };
}

function waitForTcpServer(port, label, timeoutMs = 45000) {
  const startedAt = Date.now();

  return new Promise((resolve, reject) => {
    const tryConnect = () => {
      const socket = net.createConnection(port, HOST);

      socket.once("connect", () => {
        socket.end();
        resolve();
      });

      socket.once("error", () => {
        socket.destroy();
        if (Date.now() - startedAt > timeoutMs) {
          reject(new Error(`${label} did not start on ${HOST}:${port}`));
          return;
        }
        setTimeout(tryConnect, 350);
      });
    };

    tryConnect();
  });
}

function waitForServer(timeoutMs = 45000) {
  return waitForTcpServer(PORT, "Backend", timeoutMs);
}

function waitForLlamaServer(timeoutMs = 60000) {
  return waitForTcpServer(LLAMA_CPP_PORT, "llama.cpp", timeoutMs);
}

async function startLlamaCppServer() {
  if (!app.isPackaged) {
    return;
  }

  const executable = getLlamaServerExecutable();
  const modelInfo = await ensureGraniteModel();
  const modelAlias = process.env.LLAMA_CPP_MODEL || modelInfo.alias;

  if (!fs.existsSync(executable)) {
    throw new Error(
      [
        `Missing bundled llama.cpp server: ${executable}`,
        `resourcesPath: ${process.resourcesPath}`,
        "llama.cpp resource contents:",
        describeResourceDirectory(path.join(process.resourcesPath, "llama.cpp")),
      ].join("\n")
    );
  }
  if (!fs.existsSync(modelInfo.path)) {
    throw new Error(
      [
        `Missing downloaded Granite model: ${modelInfo.path}`,
        `resourcesPath: ${process.resourcesPath}`,
        "downloaded models contents:",
        describeResourceDirectory(path.join(app.getPath("userData"), "models")),
      ].join("\n")
    );
  }

  llamaServerProcess = spawn(
    executable,
    [
      "--host",
      HOST,
      "--port",
      String(LLAMA_CPP_PORT),
      "--model",
      modelInfo.path,
      "--ctx-size",
      process.env.LLAMA_CPP_CONTEXT_SIZE || "32768",
      "--alias",
      modelAlias
    ],
    {
      cwd: path.dirname(executable),
      env: process.env,
      stdio: "ignore",
      windowsHide: true
    }
  );
  selectedLlamaModel = { ...modelInfo, alias: modelAlias };
}

function startBackend() {
  const root = getRepoRoot();
  const dataRoot = app.getPath("userData");
  const env = {
    ...process.env,
    PYTHONPATH: root,
    DK_RESOURCE_ROOT: root,
    DK_DATA_DIR: dataRoot,
    DK_BUNDLED_LLAMA: app.isPackaged ? "1" : process.env.DK_BUNDLED_LLAMA || "0",
    LLAMA_CPP_BASE_URL,
    LLAMA_CPP_MODEL: process.env.LLAMA_CPP_MODEL || selectedLlamaModel?.alias || undefined,
    DEFAULT_LLM_PROVIDER: process.env.DEFAULT_LLM_PROVIDER || undefined,
    DEFAULT_LLM_MODEL: process.env.DEFAULT_LLM_MODEL || undefined,
    CHROMA_TELEMETRY_ENABLED: "false"
  };
  for (const [key, value] of Object.entries(env)) {
    if (value === undefined) {
      delete env[key];
    }
  }

  if (app.isPackaged) {
    const executable = getBackendExecutable();
    backendProcess = spawn(executable, [], {
      cwd: process.resourcesPath,
      env,
      stdio: "ignore",
      windowsHide: true
    });
    return;
  }

  const python = getPythonCommand();
  backendProcess = spawn(
    python.command,
    [
      ...python.args,
      "-m",
      "uvicorn",
      "app.main:app",
      "--host",
      HOST,
      "--port",
      String(PORT)
    ],
    {
      cwd: root,
      env,
      stdio: "inherit",
      windowsHide: true
    }
  );
}

function isSameOriginAppUrl(rawUrl) {
  try {
    return new URL(rawUrl).origin === new URL(APP_URL).origin;
  } catch (error) {
    return false;
  }
}

function wireWindowOpenHandler(window) {
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (isSameOriginAppUrl(url)) {
      const child = new BrowserWindow({
        width: 1100,
        height: 820,
        minWidth: 760,
        minHeight: 520,
        parent: mainWindow || window,
        backgroundColor: "#111827",
        webPreferences: {
          preload: path.join(__dirname, "preload.js"),
          contextIsolation: true,
          nodeIntegration: false,
          sandbox: true,
          session: window.webContents.session
        }
      });
      wireWindowOpenHandler(child);
      child.loadURL(url);
      return { action: "deny" };
    }

    shell.openExternal(url);
    return { action: "deny" };
  });
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 980,
    minHeight: 640,
    show: false,
    backgroundColor: "#111827",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
  });

  wireWindowOpenHandler(mainWindow);

  await mainWindow.loadURL(APP_URL);
}

app.whenReady().then(async () => {
  try {
    await startLlamaCppServer();
    if (app.isPackaged) {
      await waitForLlamaServer();
    }
    startBackend();
    await waitForServer();
    await createWindow();
  } catch (error) {
    await dialog.showMessageBox({
      type: "error",
      title: "Deployable Knowledge failed to start",
      message: "The local backend did not become available.",
      detail: error.message
    });
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill();
  }
  if (llamaServerProcess && !llamaServerProcess.killed) {
    llamaServerProcess.kill();
  }
});
