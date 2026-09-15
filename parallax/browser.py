"""Playwright launch with the Parallax extension loaded (headless new mode).

Verified against Playwright 1.62 docs (chrome-extensions page): extensions
need a persistent context and channel="chromium" for headless.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright

EXTENSION_DIR = Path(__file__).resolve().parent / "extension"
VIEWPORT = {"width": 896, "height": 672}


def launch_context(pw: Playwright, headless: bool = True, user_data_dir: str | None = None) -> BrowserContext:
    if user_data_dir is None:
        user_data_dir = tempfile.mkdtemp(prefix="parallax-profile-")
    ext = str(EXTENSION_DIR)
    return pw.chromium.launch_persistent_context(
        user_data_dir,
        channel="chromium",
        headless=headless,
        viewport=VIEWPORT,
        device_scale_factor=1,
        args=[f"--disable-extensions-except={ext}", f"--load-extension={ext}"],
    )


class Browser:
    """Context manager owning a Playwright instance and one persistent context."""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._pw = None
        self.context: BrowserContext | None = None

    def __enter__(self) -> "Browser":
        self._pw = sync_playwright().start()
        self.context = launch_context(self._pw, headless=self.headless)
        return self

    def __exit__(self, *exc):
        if self.context is not None:
            self.context.close()
        if self._pw is not None:
            self._pw.stop()

    def new_page(self) -> Page:
        page = self.context.new_page()
        page.set_viewport_size(VIEWPORT)
        return page


def goto(page: Page, path_or_url: str) -> None:
    url = path_or_url if "://" in path_or_url else Path(path_or_url).resolve().as_uri()
    page.goto(url, wait_until="load")
    page.wait_for_function("() => window.__parallax !== undefined")
