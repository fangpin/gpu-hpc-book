import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from doc_scripts.export_platform_posts import chapter_source_url, platform_footer, rewrite_chapter


class ExportPlatformPostsTest(unittest.TestCase):
    def test_rewrite_chapter_appends_author_and_source_link(self):
        with TemporaryDirectory() as tmp:
            chapter = Path(tmp) / "01-intro.md"
            chapter.write_text("# Intro\n\nBody.\n", encoding="utf-8")

            text, total, rewritten = rewrite_chapter(
                chapter,
                "https://example.com/book/",
                None,
                footer=platform_footer(
                    author="Pin Fang",
                    author_url="https://github.com/fangpin",
                    source_url="https://example.com/book/chapters/01-intro.html",
                ),
            )

        self.assertEqual(total, 0)
        self.assertEqual(rewritten, 0)
        self.assertTrue(
            text.endswith(
                "\n\n---\n\n作者：[Pin Fang](https://github.com/fangpin)\n\n"
                "原文链接：[https://example.com/book/chapters/01-intro.html]"
                "(https://example.com/book/chapters/01-intro.html)\n"
            )
        )

    def test_chapter_source_url_uses_pages_url_for_any_image_mode(self):
        chapter = Path("02-wide-softmax.md")

        self.assertEqual(
            chapter_source_url("https://example.com/book/", chapter),
            "https://example.com/book/chapters/02-wide-softmax.html",
        )


if __name__ == "__main__":
    unittest.main()
