# Desktop update delivery

- Keep one stable user launch path: `desktop/release/Xuloom-win32-x64/Xuloom.exe` on Windows x64.
- Update that installation and its resources. Do not create additional feature-named executables or portable application directories unless the user explicitly requests another build.
- If Xuloom is running, stage only the changed application/Python package resources under `desktop/build`, then use `desktop/scripts/update-existing.ps1` to apply them after that installation exits. Preserve rollback resources. Never terminate the user's app to replace it.
- Run UI checks with isolated test configuration and hidden windows. Source Electron tests are suitable while the installed application is still open; after the update, verify the same installed executable.
