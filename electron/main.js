const { app, BrowserWindow, dialog, shell } = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");

const HOST = "127.0.0.1";
const PORT = Number(process.env.DK_PORT || "8000");
const APP_URL = `http://${HOST}:${PORT}`;

let mainWindow;
let backendProcess;

function getRepoRoot() {
  return app.isPackaged ? process.resourcesPath : path.resolve(__dirname, "..");
}

function getBackendExecutable() {
  const executable = process.platform === "win32"
    ? "DeployableKnowledgeBackend.exe"
    : "DeployableKnowledgeBackend";

  return path.join(process.resourcesPath, "backend", executable);
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

function waitForServer(timeoutMs = 45000) {
  const startedAt = Date.now();

  return new Promise((resolve, reject) => {
    const tryConnect = () => {
      const socket = net.createConnection(PORT, HOST);

      socket.once("connect", () => {
        socket.end();
        resolve();
      });

      socket.once("error", () => {
        socket.destroy();
        if (Date.now() - startedAt > timeoutMs) {
          reject(new Error(`Backend did not start at ${APP_URL}`));
          return;
        }
        setTimeout(tryConnect, 350);
      });
    };

    tryConnect();
  });
}

function startBackend() {
  const root = getRepoRoot();
  const dataRoot = app.getPath("userData");
  const env = {
    ...process.env,
    PYTHONPATH: root,
    DK_RESOURCE_ROOT: root,
    DK_DATA_DIR: dataRoot,
    CHROMA_TELEMETRY_ENABLED: "false"
  };

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

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  await mainWindow.loadURL(APP_URL);
}

app.whenReady().then(async () => {
  startBackend();

  try {
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
});
