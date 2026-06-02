const { app, BrowserWindow, dialog, shell } = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");

const HOST = "127.0.0.1";
const PORT = Number(process.env.DK_PORT || "8000");
const LLAMA_CPP_PORT = Number(process.env.LLAMA_CPP_PORT || "8081");
const APP_URL = `http://${HOST}:${PORT}`;
const LLAMA_CPP_BASE_URL = `http://${HOST}:${LLAMA_CPP_PORT}/v1`;

let mainWindow;
let backendProcess;
let llamaServerProcess;
let selectedLlamaModel;

const GRANITE_MODEL_SPECS = [
  {
    file: "granite-4.1-3b-Q4_K_M.gguf",
    alias: "granite4.1:3b-q4_K_M",
    label: "Granite 4.1 3B Q4_K_M",
    dirs: ["granite4.1-3b", "granite4-3b"]
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
  for (const spec of GRANITE_MODEL_SPECS) {
    const candidates = [
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
    path: path.join(process.resourcesPath, "models", fallback.dirs[0], fallback.file)
  };
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

function startLlamaCppServer() {
  if (!app.isPackaged) {
    return;
  }

  const executable = getLlamaServerExecutable();
  const modelInfo = getGraniteModelInfo();
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
        `Missing bundled Granite model: ${modelInfo.path}`,
        `resourcesPath: ${process.resourcesPath}`,
        "models resource contents:",
        describeResourceDirectory(path.join(process.resourcesPath, "models")),
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
    startLlamaCppServer();
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
