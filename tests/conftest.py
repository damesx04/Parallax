import pytest

from parallax.browser import Browser


@pytest.fixture(scope="session")
def browser():
    with Browser(headless=True) as b:
        yield b


@pytest.fixture
def page(browser):
    p = browser.new_page()
    yield p
    p.close()
