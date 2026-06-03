const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("deployableKnowledge", {
  platform: process.platform,
  cancelSetup: () => ipcRenderer.send("setup:cancel")
});
