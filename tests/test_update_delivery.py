"""Exercise the actual updater's partial-install rollback on disposable resource files."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == "nt", "Windows resource updater")
class UpdateDeliveryTests(unittest.TestCase):
    def test_partial_install_failure_restores_resources_then_the_same_payload_can_retry(self):
        with tempfile.TemporaryDirectory(prefix="xuloom-update-fixture-") as temporary:
            root = Path(temporary).resolve()
            package, stage = root / "installation", root / "stage"
            package.mkdir(); stage.mkdir()
            # Deliberately plain text, never executed; the updater only checks its presence.
            executable = package / "Xuloom.exe"
            executable.write_bytes(b"NON_EXECUTABLE_TEST_MARKER")
            metadata = "xuloom-0.1.0.dist-info"
            entries = (("app", "resources/app"), ("xuloom", "resources/python/Lib/site-packages/xuloom"),
                ("metadata", "resources/python/Lib/site-packages/" + metadata))
            for source, target in entries:
                (stage / source).mkdir(parents=True)
                (package / target).mkdir(parents=True)
                (stage / source / "proof.txt").write_text("new " + source)
                (package / target / "proof.txt").write_text("old " + source)
            (stage / "manifest.json").write_text(json.dumps({"metadataDirectory": metadata}))
            updater = Path(__file__).resolve().parents[1] / "desktop/scripts/update-existing.ps1"
            wrapper = root / "fault.ps1"
            wrapper.write_text('''param($Updater, $Package, $Stage)
$script:Injected = $false
function Move-Item {
    param([string]$LiteralPath, [string]$Destination)
    if (-not $script:Injected -and $LiteralPath -eq (Join-Path $Stage 'metadata')) {
        $script:Injected = $true
        throw 'fixture failure after application and engine replacement'
    }
    Microsoft.PowerShell.Management\\Move-Item -LiteralPath $LiteralPath -Destination $Destination
}
& $Updater -PackageDirectory $Package -StagedResources $Stage -WaitSeconds 0
exit $LASTEXITCODE
''', encoding="utf-8")
            command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"]
            options = {"capture_output": True, "text": True, "timeout": 30, "creationflags": subprocess.CREATE_NO_WINDOW}
            failed = subprocess.run([*command, str(wrapper), str(updater), str(package), str(stage)], **options)
            self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
            self.assertEqual(json.loads((stage / "status.json").read_text())["state"], "failed")
            for source, target in entries:
                self.assertEqual((package / target / "proof.txt").read_text(), "old " + source)
                self.assertEqual((stage / source / "proof.txt").read_text(), "new " + source)
            applied = subprocess.run([*command, str(updater), "-PackageDirectory", str(package), "-StagedResources", str(stage), "-WaitSeconds", "0"], **options)
            self.assertEqual(applied.returncode, 0, applied.stdout + applied.stderr)
            self.assertEqual(json.loads((stage / "status.json").read_text())["state"], "applied")
            for source, target in entries:
                self.assertEqual((package / target / "proof.txt").read_text(), "new " + source)
                self.assertEqual((stage / "backup" / source / "proof.txt").read_text(), "old " + source)
            self.assertEqual(executable.read_bytes(), b"NON_EXECUTABLE_TEST_MARKER")
