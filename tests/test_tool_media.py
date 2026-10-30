import base64
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from xuloom.context.tool_media import archive_images, image_preview, receipt_text


class ToolMediaTests(unittest.TestCase):
    def test_workspace_images_and_document_relative_paths_keep_file_preview_scope(self):
        from xuloom.features.file_changes import preview_image_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            (root / "docs/figures").mkdir(parents=True)
            (root / "docs/report.md").write_text("![image](figures/plot.png)")
            picture = root / "docs/figures/plot.png"
            picture.write_bytes(base64.b64decode(self.image()["source"]["data"]))
            self.assertEqual(preview_image_file(root, "docs/figures/plot.png", [])["width"], 720)
            self.assertTrue(Path(preview_image_file(root, "figures/plot.png", [], str(root / "docs/report.md"))["path"]).samefile(picture))
            outside = Path(directory) / "private.png"
            outside.write_bytes(picture.read_bytes())
            with self.assertRaises(ValueError): preview_image_file(root, "../private.png", [])
            with self.assertRaises(ValueError): preview_image_file(root, str(outside), [])
            with self.assertRaises(ValueError): preview_image_file(root, "../../private.png", [], str(root / "docs/report.md"))
            self.assertEqual(preview_image_file(root, str(outside), [{"path": str(outside)}])["width"], 720)
            with self.assertRaises(ValueError): preview_image_file(root, "zzc", [])
            with self.assertRaises(ValueError): preview_image_file(root, "docs/report.md", [])
            picture.write_bytes(b"not an image")
            with self.assertRaises((ValueError, OSError)): preview_image_file(root, str(picture), [])

    def image(self):
        stream = io.BytesIO()
        Image.new("RGB", (720, 360), "#479cc9").save(stream, "PNG")
        return {"type": "image", "source": {"type": "base64", "data": base64.b64encode(stream.getvalue()).decode()}}

    def test_snapshot_bounds_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "session"
            blocks = [{"type": "text", "text": "Read image"}, self.image()]
            media = archive_images(blocks, root, "proof.png")[0]
            self.assertEqual(archive_images(blocks)[0]["id"], media["id"])
            self.assertEqual(receipt_text(blocks), "Read image")
            self.assertNotIn("base64", str(media))
            self.assertEqual(image_preview(root, media["id"], thumbnail=True)["previewWidth"], 256)
            self.assertEqual(image_preview(root, media["id"])["previewWidth"], 720)
            for invalid in ("../escape", "https://example.invalid/p.png", "A" * 64, None):
                with self.assertRaises(ValueError): image_preview(root, invalid)
            with self.assertRaises(ValueError): image_preview(Path(directory) / "other", media["id"])
            (root / "images" / (media["id"] + ".image")).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "存档已变化"): image_preview(root, media["id"])

    def test_invalid_unsupported_and_failed_storage_are_explicit_nonfatal(self):
        self.assertIn("error", archive_images([{"type": "image", "source": {"type": "url", "url": "https://example.invalid"}}])[0])
        self.assertIn("error", archive_images([{"type": "image", "source": {"type": "base64", "data": "!bad!"}}])[0])
        with patch("xuloom.context.tool_media.MAX_PIXELS", 1):
            self.assertIn("error", archive_images([self.image()])[0])
        with tempfile.TemporaryDirectory() as directory, patch("xuloom.context.tool_media.atomic_bytes", side_effect=PermissionError("fixture")):
            self.assertIn("fixture", archive_images([self.image()], directory)[0]["error"])

    def test_linked_image_directory_is_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "session"
            with patch.object(Path, "is_symlink", return_value=True):
                self.assertIn("error", archive_images([self.image()], root)[0])
                with self.assertRaises(ValueError): image_preview(root, "a" * 64)
            self.assertFalse(root.exists())
