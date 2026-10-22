import errno
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
SVG_BYTES = b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="16" height="9"/></svg>'


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
    def test_transient_lark_errors_classifies_common_network_failures(self):
        self.assertTrue(hasattr(sync_lark_doc, "is_transient_lark_error"))
        transient_messages = (
            "TLS handshake timeout",
            "context deadline exceeded",
            "network is unreachable",
            "no route to host",
            "broken pipe",
            "connection reset by peer",
        )
        for message in transient_messages:
            with self.subTest(message=message):
                self.assertTrue(
                    sync_lark_doc.is_transient_lark_error(SystemExit(message))
                )

        for error_number in (
            errno.ETIMEDOUT,
            errno.ECONNRESET,
            errno.ECONNREFUSED,
            errno.ENETUNREACH,
            errno.EHOSTUNREACH,
            errno.EPIPE,
        ):
            with self.subTest(errno=error_number):
                self.assertTrue(
                    sync_lark_doc.is_transient_lark_error(
                        OSError(error_number, "network failure")
                    )
                )

        for message in (
            "permission denied",
            "invalid argument",
            "whiteboard not found",
            "unsafe or invalid SVG",
        ):
            with self.subTest(non_transient=message):
                self.assertFalse(
                    sync_lark_doc.is_transient_lark_error(SystemExit(message))
                )

    def test_localize_whiteboards_exports_board_as_svg(self):
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
                "![飞书画板](../assets/images/chapter/whiteboard-01.svg)",
            )
            self.assertEqual(
                (Path(tmp) / "chapter" / "whiteboard-01.svg").read_bytes(),
                SVG_BYTES,
            )

    def test_localize_whiteboards_retries_transient_svg_export(self):
        attempts = 0

        def fake_run(cmd, capture_output, text, env, cwd=None):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return subprocess.CompletedProcess(
                    cmd,
                    1,
                    stdout="",
                    stderr="TLS handshake timeout",
                )
            return self._fake_whiteboard_run(cmd, capture_output, text, env, cwd)

        with TemporaryDirectory() as tmp:
            with patch.object(sync_lark_doc, "IMAGES_DIR", Path(tmp)):
                with patch.object(sync_lark_doc.shutil, "which", return_value="lark-cli"):
                    with patch.object(sync_lark_doc.subprocess, "run", side_effect=fake_run):
                        localized = sync_lark_doc.localize_whiteboards(
                            '<whiteboard token="board-token"></whiteboard>',
                            "chapter",
                        )

            self.assertEqual(attempts, 2)
            self.assertEqual(
                localized,
                "![飞书画板](../assets/images/chapter/whiteboard-01.svg)",
            )
            self.assertEqual(
                (Path(tmp) / "chapter" / "whiteboard-01.svg").read_bytes(),
                SVG_BYTES,
            )

    def test_localize_whiteboards_falls_back_after_second_transient_failure(self):
        commands = []

        def fake_run(cmd, capture_output, text, env, cwd=None):
            commands.append(cmd[:3])
            if cmd[:3] == ["lark-cli", "whiteboard", "+export"]:
                return subprocess.CompletedProcess(
                    cmd,
                    1,
                    stdout="",
                    stderr="context deadline exceeded",
                )
            return self._fake_whiteboard_jpeg_run(cmd, capture_output, text, env, cwd)

        with TemporaryDirectory() as tmp:
            with patch.object(sync_lark_doc, "IMAGES_DIR", Path(tmp)):
                with patch.object(sync_lark_doc.shutil, "which", return_value="lark-cli"):
                    with patch.object(sync_lark_doc.subprocess, "run", side_effect=fake_run):
                        localized = sync_lark_doc.localize_whiteboards(
                            '<whiteboard token="board-token"></whiteboard>',
                            "chapter",
                        )

            self.assertEqual(
                commands,
                [
                    ["lark-cli", "whiteboard", "+export"],
                    ["lark-cli", "whiteboard", "+export"],
                    ["lark-cli", "docs", "+media-download"],
                ],
            )
            self.assertEqual(
                localized,
                "![飞书画板](../assets/images/chapter/whiteboard-01.jpg)",
            )

    def test_localize_whiteboards_falls_back_to_jpeg_when_svg_export_fails(self):
        commands = []

        def fake_run(cmd, capture_output, text, env, cwd=None):
            commands.append(cmd[:3])
            if cmd[:3] == ["lark-cli", "whiteboard", "+export"]:
                return subprocess.CompletedProcess(
                    cmd,
                    1,
                    stdout="",
                    stderr="SVG export unavailable",
                )
            return self._fake_whiteboard_jpeg_run(cmd, capture_output, text, env, cwd)

        with TemporaryDirectory() as tmp:
            with patch.object(sync_lark_doc, "IMAGES_DIR", Path(tmp)):
                with patch.object(sync_lark_doc.shutil, "which", return_value="lark-cli"):
                    with patch.object(sync_lark_doc.subprocess, "run", side_effect=fake_run):
                        localized = sync_lark_doc.localize_whiteboards(
                            '<whiteboard token="board-token"></whiteboard>',
                            "chapter",
                        )

            self.assertEqual(
                commands,
                [
                    ["lark-cli", "whiteboard", "+export"],
                    ["lark-cli", "docs", "+media-download"],
                ],
            )
            self.assertEqual(
                localized,
                "![飞书画板](../assets/images/chapter/whiteboard-01.jpg)",
            )
            self.assertEqual(
                (Path(tmp) / "chapter" / "whiteboard-01.jpg").read_bytes(),
                JPEG_BYTES,
            )

    def test_localize_whiteboards_rejects_unsafe_svg_and_falls_back_to_jpeg(self):
        unsafe_svgs = {
            "script": b'<svg xmlns="http://www.w3.org/2000/svg"><script/></svg>',
            "foreign object": (
                b'<svg xmlns="http://www.w3.org/2000/svg"><foreignObject/></svg>'
            ),
            "event handler": (
                b'<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"/>'
            ),
            "XML stylesheet processing instruction": (
                b'<?xml-stylesheet type="text/css" href="https://example.com/track.css"?>'
                b'<svg xmlns="http://www.w3.org/2000/svg"/>'
            ),
            "UTF-16 doctype": (
                '<!DOCTYPE svg><svg xmlns="http://www.w3.org/2000/svg"/>'.encode("utf-16")
            ),
            "remote href": (
                b'<svg xmlns="http://www.w3.org/2000/svg"><image href="https://example.com/a.png"/></svg>'
            ),
            "relative href": (
                b'<svg xmlns="http://www.w3.org/2000/svg"><image href="tracker.png"/></svg>'
            ),
            "root relative href": (
                b'<svg xmlns="http://www.w3.org/2000/svg"><image href="/tracker.png"/></svg>'
            ),
            "absolute XML base": (
                b'<svg xmlns="http://www.w3.org/2000/svg" xml:base="/tracker.svg">'
                b'<use href="#shape"/></svg>'
            ),
            "relative XML base": (
                b'<svg xmlns="http://www.w3.org/2000/svg" xml:base="assets/tracker.svg">'
                b'<image href="#shape"/></svg>'
            ),
            "javascript href": (
                b'<svg xmlns="http://www.w3.org/2000/svg"><a href="javascript:alert(1)"/></svg>'
            ),
            "remote stylesheet": (
                b'<svg xmlns="http://www.w3.org/2000/svg"><style>@import url(https://example.com/a.css)</style></svg>'
            ),
            "relative stylesheet URL": (
                b'<svg xmlns="http://www.w3.org/2000/svg"><style>rect { fill: url(/paint) }</style></svg>'
            ),
            "escaped CSS URL": (
                br'<svg xmlns="http://www.w3.org/2000/svg"><style>rect { fill: u\72l(/paint) }</style></svg>'.replace(b'\\"', b'"')
            ),
            "escaped CSS import": (
                br'<svg xmlns="http://www.w3.org/2000/svg"><style>@\69mport "/track.css"</style></svg>'.replace(b'\\"', b'"')
            ),
            "style attribute": (
                b'<svg xmlns="http://www.w3.org/2000/svg"><rect style="fill: red"/></svg>'
            ),
            "dynamic href": (
                b'<svg xmlns="http://www.w3.org/2000/svg"><set attributeName="href" to="javascript:alert(1)"/></svg>'
            ),
            "animated href": (
                b'<svg xmlns="http://www.w3.org/2000/svg"><animate attributeName="href" values="#safe;tracker.png"/></svg>'
            ),
        }

        for name, unsafe_svg in unsafe_svgs.items():
            with self.subTest(name=name), TemporaryDirectory() as tmp:
                def fake_run(cmd, capture_output, text, env, cwd=None):
                    if cmd[:3] == ["lark-cli", "whiteboard", "+export"]:
                        output_name = cmd[cmd.index("--output") + 1]
                        (Path(cwd) / output_name).write_bytes(unsafe_svg)
                        return subprocess.CompletedProcess(
                            cmd,
                            0,
                            stdout=json.dumps(
                                {"ok": True, "data": {"svg_path": output_name}}
                            ),
                            stderr="",
                        )
                    return self._fake_whiteboard_jpeg_run(
                        cmd, capture_output, text, env, cwd
                    )

                with patch.object(sync_lark_doc, "IMAGES_DIR", Path(tmp)):
                    with patch.object(sync_lark_doc.shutil, "which", return_value="lark-cli"):
                        with patch.object(sync_lark_doc.subprocess, "run", side_effect=fake_run):
                            localized = sync_lark_doc.localize_whiteboards(
                                '<whiteboard token="board-token"></whiteboard>',
                                "chapter",
                            )

                self.assertEqual(
                    localized,
                    "![飞书画板](../assets/images/chapter/whiteboard-01.jpg)",
                )
                self.assertFalse((Path(tmp) / "chapter" / "whiteboard-01.svg").exists())
                self.assertEqual(
                    (Path(tmp) / "chapter" / "whiteboard-01.jpg").read_bytes(),
                    JPEG_BYTES,
                )

    def test_localize_whiteboards_removes_stale_image_extension(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "chapter"
            dest.mkdir()
            (dest / "whiteboard-01.jpg").write_bytes(JPEG_BYTES)
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
                "![飞书画板](../assets/images/chapter/whiteboard-01.svg)",
            )
            self.assertEqual(
                sorted(path.name for path in dest.iterdir()),
                ["whiteboard-01.svg"],
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

![飞书画板](../assets/images/chapter/whiteboard-01.svg)
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
        assert cmd[:3] == ["lark-cli", "whiteboard", "+export"]
        assert cmd[cmd.index("--as") + 1] == "user"
        assert cmd[cmd.index("--output-type") + 1] == "svg"
        assert cmd[cmd.index("--whiteboard-token") + 1] == "board-token"
        output_name = cmd[cmd.index("--output") + 1]
        saved_path = Path(cwd) / output_name
        saved_path.write_bytes(SVG_BYTES)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "data": {
                        "svg_path": output_name,
                    },
                }
            ),
            stderr="",
        )

    @staticmethod
    def _fake_whiteboard_jpeg_run(cmd, capture_output, text, env, cwd=None):
        assert cmd[:3] == ["lark-cli", "docs", "+media-download"]
        assert "--as" in cmd
        assert cmd[cmd.index("--as") + 1] == "user"
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
