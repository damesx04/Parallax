"""Fetch and render the extension's annotated accessibility tree."""
from __future__ import annotations

from playwright.sync_api import Page


def get_tree(page: Page) -> list[dict]:
    return page.evaluate("() => window.__parallax.tree()")


def draw_overlay(page: Page) -> int:
    return page.evaluate("() => window.__parallax.overlay(true)")


def clear_overlay(page: Page) -> None:
    page.evaluate("() => window.__parallax.overlay(false)")


def render_entry(e: dict) -> str:
    states = f" [{', '.join(e['states'])}]" if e.get("states") else ""
    x, y, w, h = e["box"]
    name = e["name"].replace('"', "'")
    return f"{'  ' * e['depth']}[{e['id']}] {e['role']} \"{name}\"{states} ({x},{y},{w},{h})"


def render_tree(entries: list[dict]) -> str:
    return "\n".join(render_entry(e) for e in entries)
