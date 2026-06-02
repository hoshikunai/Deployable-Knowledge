const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("deployableKnowledge", {
  platform: process.platform
});
