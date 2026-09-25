import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "doc_scripts" / "sync_lark_doc.py"
spec = importlib.util.spec_from_file_location("sync_lark_doc", SCRIPT_PATH)
sync_lark_doc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sync_lark_doc)

PNG_BYTES = b"\x89PNG\r\n\x1a\npng data"


class FakeHttpResponse:
    def __init__(self, data: bytes, content_type: str):
        self._data = data
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._data


class SyncLarkDocImageTest(unittest.TestCase):
    def test_localize_images_does_not_save_non_image_http_response(self):
        with TemporaryDirectory() as tmp:
            with patch.object(sync_lark_doc, "IMAGES_DIR", Path(tmp)):
                with patch.object(
                    sync_lark_doc.urllib.request,
                    "urlopen",
                    return_value=FakeHttpResponse(b"<!doctype html><title>login</title>", "text/html"),
                ):
                    content = "![diagram](https://example.com/protected-image)"

                    self.assertEqual(sync_lark_doc.localize_images(content, "chapter"), content)
                    self.assertEqual(list(Path(tmp).rglob("*")), [])

    def test_localize_images_uses_lark_cli_for_feishu_file_urls(self):
        with TemporaryDirectory() as tmp:
            with patch.object(sync_lark_doc, "IMAGES_DIR", Path(tmp)):
                with patch.object(sync_lark_doc.urllib.request, "urlopen", side_effect=AssertionError):
                    with patch.object(sync_lark_doc.shutil, "which", return_value="lark-cli"):
                        with patch.object(sync_lark_doc.subprocess, "run", side_effect=self._fake_media_run):
                            localized = sync_lark_doc.localize_images(
                                "![bf16](https://feishu.cn/file/XUbWbt0NPowmqyxci38lestsgAb)",
                                "02-llm",
                            )

            self.assertEqual(localized, "![bf16](../assets/images/02-llm/image-01.png)")
            self.assertEqual((Path(tmp) / "02-llm" / "image-01.png").read_bytes(), PNG_BYTES)

    def test_non_lark_file_path_uses_regular_http_download(self):
        with TemporaryDirectory() as tmp:
            with patch.object(sync_lark_doc, "IMAGES_DIR", Path(tmp)):
                with patch.object(
                    sync_lark_doc.urllib.request,
                    "urlopen",
                    return_value=FakeHttpResponse(PNG_BYTES, "image/png"),
                ):
                    with patch.object(sync_lark_doc.subprocess, "run", side_effect=AssertionError):
                        localized = sync_lark_doc.localize_images(
                            "![asset](https://example.com/file/not-a-lark-token)",
                            "chapter",
                        )

            self.assertEqual(localized, "![asset](../assets/images/chapter/image-01.png)")
            self.assertEqual((Path(tmp) / "chapter" / "image-01.png").read_bytes(), PNG_BYTES)

    @staticmethod
    def _fake_media_run(cmd, capture_output, text, env, cwd=None):
        assert cmd[:3] == ["lark-cli", "docs", "+media-preview"]
        assert cmd[cmd.index("--token") + 1] == "XUbWbt0NPowmqyxci38lestsgAb"
        output_base = cmd[cmd.index("--output") + 1]
        saved_path = Path(cwd) / f"{output_base}.png"
        saved_path.write_bytes(PNG_BYTES)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "data": {
                        "content_type": "image/png",
                        "saved_path": str(saved_path),
                    },
                }
            ),
            stderr="",
        )


if __name__ == "__main__":
    unittest.main()
