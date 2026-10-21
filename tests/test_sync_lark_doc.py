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
JPEG_BYTES = b"\xff\xd8\xffjpeg data"


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
    def test_localize_whiteboards_downloads_board_thumbnail(self):
        with TemporaryDirectory() as tmp:
            with patch.object(sync_lark_doc, "IMAGES_DIR", Path(tmp)):
                with patch.object(sync_lark_doc.shutil, "which", return_value="lark-cli"):
                    with patch.object(
                        sync_lark_doc.subprocess,
                        "run",
                        side_effect=self._fake_whiteboard_run,
                    ):
                        localized = sync_lark_doc.localize_whiteboards(
                            '<whiteboard token="board-token"></whiteboard>',
                            "chapter",
                        )

            self.assertEqual(
                localized,
                "![飞书画板](../assets/images/chapter/whiteboard-01.jpg)",
            )
            self.assertEqual(
                (Path(tmp) / "chapter" / "whiteboard-01.jpg").read_bytes(),
                JPEG_BYTES,
            )

    def test_localize_whiteboards_keeps_visible_placeholder_when_download_fails(self):
        with TemporaryDirectory() as tmp:
            with patch.object(sync_lark_doc, "IMAGES_DIR", Path(tmp)):
                with patch.object(sync_lark_doc.shutil, "which", return_value="lark-cli"):
                    with patch.object(
                        sync_lark_doc.subprocess,
                        "run",
                        return_value=subprocess.CompletedProcess(
                            ["lark-cli", "docs", "+media-download"],
                            1,
                            stdout="",
                            stderr="download failed",
                        ),
                    ):
                        localized = sync_lark_doc.localize_whiteboards(
                            '<whiteboard token="board-token"></whiteboard>',
                            "chapter",
                        )

            self.assertEqual(
                localized,
                "> [飞书画板导出失败，请到原文档查看对应内容]",
            )
            self.assertEqual(list(Path(tmp).rglob("*")), [])

    def test_localize_whiteboards_ignores_code_and_comment_examples(self):
        content = """````markdown
```xml
<whiteboard token="board-token"></whiteboard>
```
````

~~~markdown
```xml
<whiteboard token="board-token"></whiteboard>
```
~~~

`<whiteboard token="board-token"></whiteboard>`

<!-- <whiteboard token="board-token"></whiteboard> -->

<!--
<whiteboard token="board-token"></whiteboard>
-->

<whiteboard token="board-token"></whiteboard>
"""
        expected = """````markdown
```xml
<whiteboard token="board-token"></whiteboard>
```
````

~~~markdown
```xml
<whiteboard token="board-token"></whiteboard>
```
~~~

`<whiteboard token="board-token"></whiteboard>`

<!-- <whiteboard token="board-token"></whiteboard> -->

<!--
<whiteboard token="board-token"></whiteboard>
-->

![飞书画板](../assets/images/chapter/whiteboard-01.jpg)
"""
        with TemporaryDirectory() as tmp:
            with patch.object(sync_lark_doc, "IMAGES_DIR", Path(tmp)):
                with patch.object(sync_lark_doc.shutil, "which", return_value="lark-cli"):
                    with patch.object(
                        sync_lark_doc.subprocess,
                        "run",
                        side_effect=self._fake_whiteboard_run,
                    ) as run:
                        localized = sync_lark_doc.localize_whiteboards(content, "chapter")

            self.assertEqual(localized, expected)
            self.assertEqual(run.call_count, 1)

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

    def test_lark_media_download_removes_wrong_extension_intermediate(self):
        with TemporaryDirectory() as tmp:
            with patch.object(sync_lark_doc, "IMAGES_DIR", Path(tmp)):
                with patch.object(sync_lark_doc.shutil, "which", return_value="lark-cli"):
                    with patch.object(
                        sync_lark_doc.subprocess,
                        "run",
                        side_effect=self._fake_mislabeled_media_run,
                    ):
                        localized = sync_lark_doc.localize_images(
                            "![asset](https://feishu.cn/file/mislabeled-token)",
                            "chapter",
                        )

            self.assertEqual(localized, "![asset](../assets/images/chapter/image-01.jpg)")
            self.assertEqual(
                sorted(path.name for path in (Path(tmp) / "chapter").iterdir()),
                ["image-01.jpg"],
            )
            self.assertEqual(
                (Path(tmp) / "chapter" / "image-01.jpg").read_bytes(),
                JPEG_BYTES,
            )

    def test_lark_media_download_rejects_saved_path_outside_destination(self):
        def fake_run(cmd, capture_output, text, env, cwd=None):
            saved_path = Path(cwd).parent / "outside.png"
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

        source = "![asset](https://feishu.cn/file/outside-token)"
        with TemporaryDirectory() as tmp:
            with patch.object(sync_lark_doc, "IMAGES_DIR", Path(tmp)):
                with patch.object(sync_lark_doc.shutil, "which", return_value="lark-cli"):
                    with patch.object(sync_lark_doc.subprocess, "run", side_effect=fake_run):
                        localized = sync_lark_doc.localize_images(source, "chapter")

            self.assertEqual(localized, source)
            self.assertEqual((Path(tmp) / "outside.png").read_bytes(), PNG_BYTES)
            self.assertFalse((Path(tmp) / "chapter").exists())

    @staticmethod
    def _fake_media_run(cmd, capture_output, text, env, cwd=None):
        assert cmd[:3] == ["lark-cli", "docs", "+media-preview"]
        assert "--as" not in cmd
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

    @staticmethod
    def _fake_whiteboard_run(cmd, capture_output, text, env, cwd=None):
        assert cmd[:3] == ["lark-cli", "docs", "+media-download"]
        assert "--as" not in cmd
        assert cmd[cmd.index("--type") + 1] == "whiteboard"
        assert cmd[cmd.index("--token") + 1] == "board-token"
        output_base = cmd[cmd.index("--output") + 1]
        saved_path = Path(cwd) / f"{output_base}.jpg"
        saved_path.write_bytes(JPEG_BYTES)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "data": {
                        "content_type": "image/jpeg",
                        "saved_path": str(saved_path),
                    },
                }
            ),
            stderr="",
        )

    @staticmethod
    def _fake_mislabeled_media_run(cmd, capture_output, text, env, cwd=None):
        output_base = cmd[cmd.index("--output") + 1]
        saved_path = Path(cwd) / f"{output_base}.png"
        saved_path.write_bytes(JPEG_BYTES)
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
