const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");

function getPythonCommand() {
  if (process.env.DK_PYTHON) {
    return process.env.DK_PYTHON;
  }

  const venvPython = process.platform === "win32"
    ? path.join(root, "venv", "Scripts", "python.exe")
    : path.join(root, "venv", "bin", "python");

  if (fs.existsSync(venvPython)) {
    return venvPython;
  }

  return process.platform === "win32" ? "python" : "python3";
}

const python = getPythonCommand();
const result = spawnSync(
  python,
  ["-m", "PyInstaller", "--clean", "--noconfirm", "electron/backend.spec"],
  {
    cwd: root,
    env: {
      ...process.env,
      PYTHONPATH: root
    },
    stdio: "inherit"
  }
);

if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}

if (result.status !== 0) {
  console.error(
    "Backend build failed. Install build dependencies with: python -m pip install -r requirements-build.txt"
  );
  process.exit(result.status || 1);
}
