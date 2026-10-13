const { contextBridge, ipcRenderer } = require("electron");
contextBridge.exposeInMainWorld("codeclaw", {
  invoke: (method, params) =>
    ipcRenderer.invoke("workbench:invoke", method, params || {}),
  onEvent: (callback) => {
    const handler = (_event, payload) => callback(payload);
    ipcRenderer.on("workbench:event", handler);
    return () => ipcRenderer.removeListener("workbench:event", handler);
  },
});
