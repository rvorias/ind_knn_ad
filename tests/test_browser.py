"""Optional real-browser workflow smoke test (install Playwright + Chromium)."""

import io
import socket
import threading
import time
from importlib import import_module

import pytest
from PIL import Image

playwright_api = pytest.importorskip("playwright.sync_api")
uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")
create_app = import_module("indad.server").create_app


def payload(name, value):
    output = io.BytesIO()
    Image.new("RGB", (40, 40), (value, value, value)).save(output, "PNG")
    return {"name": name, "mimeType": "image/png", "buffer": output.getvalue()}


def test_operator_browser_workflow(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        config = uvicorn.Config(create_app(tmp_path), log_level="error")
        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.02)
            assert server.started
            with playwright_api.sync_playwright() as playwright:
                try:
                    browser = playwright.chromium.launch()
                except playwright_api.Error as exc:
                    pytest.skip(f"Chromium is unavailable: {exc}")
                try:
                    page = browser.new_page(viewport={"width": 1440, "height": 1000})
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(f"http://127.0.0.1:{listener.getsockname()[1]}")
                    page.get_by_role("button", name="Create your first dataset").click()
                    page.get_by_label("Dataset name", exact=True).fill("operator_parts")
                    page.locator("#create-form").get_by_role(
                        "button", name="Create dataset", exact=True
                    ).click()
                    page.get_by_role(
                        "button", name="Import images", exact=False
                    ).click()
                    page.locator("#import-files").set_input_files(
                        [payload("a.png", 10), payload("b.png", 20)]
                    )
                    page.get_by_role("button", name="Save images", exact=True).click()
                    playwright_api.expect(page.locator("#train-count")).to_have_text(
                        "2"
                    )
                    page.get_by_role(
                        "button", name="Import images", exact=False
                    ).click()
                    page.locator("#import-purpose").select_option("test-good")
                    page.locator("#import-files").set_input_files(payload("c.png", 30))
                    page.get_by_role("button", name="Save images", exact=True).click()
                    playwright_api.expect(page.locator("#test-count")).to_have_text("1")
                    playwright_api.expect(page.locator("#ready-label")).to_have_text(
                        "Ready to inspect"
                    )
                    page.get_by_label("Search filenames").fill("a.png")
                    playwright_api.expect(page.locator(".sample-card")).to_have_count(1)
                    page.get_by_label("Search filenames").fill("")
                    page.get_by_role("tab", name="Quality checks").click()
                    playwright_api.expect(page.locator("#findings")).to_contain_text(
                        "single test class"
                    )
                    with page.expect_download() as download:
                        page.get_by_role(
                            "button", name="Export manifest", exact=False
                        ).click()
                    assert (
                        download.value.suggested_filename
                        == "operator_parts-manifest.json"
                    )
                    page.get_by_role("tab", name="Inspection baseline").click()
                    page.locator("#method").select_option("spade")
                    page.get_by_role(
                        "button", name="Train baseline", exact=True
                    ).click()
                    playwright_api.expect(
                        page.locator("#run-status")
                    ).to_have_attribute("data-status", "ready", timeout=15000)
                    page.get_by_role(
                        "button", name="Inspect image", exact=False
                    ).click()
                    playwright_api.expect(
                        page.locator(".prediction-images img")
                    ).to_have_count(3, timeout=10000)
                    page.get_by_role("tab", name="Samples").click()
                    page.get_by_role(
                        "button", name="Review test/good/c.png", exact=True
                    ).click()
                    page.get_by_role(
                        "button", name="Correct label or collection"
                    ).click()
                    page.locator("#edit-purpose").select_option("test-defect")
                    page.locator("#edit-label").fill("scratch")
                    page.locator("#edit-reason").fill("Reviewed defect")
                    page.get_by_role("button", name="Save change", exact=True).click()
                    playwright_api.expect(page.locator("#test-summary")).to_have_text(
                        "0 healthy · 1 defective"
                    )
                    page.locator("#change-history summary").click()
                    page.get_by_role("button", name="Undo change", exact=True).click()
                    playwright_api.expect(page.locator("#test-summary")).to_have_text(
                        "1 healthy · 0 defective"
                    )
                    page.get_by_role(
                        "button", name="Review train/good/a.png", exact=True
                    ).click()
                    page.get_by_role(
                        "button", name="Exclude sample", exact=True
                    ).click()
                    page.locator("#edit-reason").fill("Blurred capture")
                    page.get_by_role("button", name="Save change", exact=True).click()
                    playwright_api.expect(page.locator("#train-count")).to_have_text(
                        "1"
                    )
                    page.get_by_role(
                        "button", name="Restore sample", exact=True
                    ).click()
                    playwright_api.expect(page.locator("#train-count")).to_have_text(
                        "2"
                    )
                    assert not errors
                    page.set_viewport_size({"width": 390, "height": 844})
                    playwright_api.expect(page.locator(".new-dataset")).to_be_visible()
                    assert page.evaluate(
                        "document.documentElement.scrollWidth <= window.innerWidth"
                    )
                    page.reload()
                    page.get_by_role("tab", name="Inspection baseline").click()
                    playwright_api.expect(page.locator("#method")).to_have_value(
                        "spade"
                    )
                    playwright_api.expect(page.locator("#predict")).to_be_enabled()
                finally:
                    browser.close()
        finally:
            server.should_exit = True
            thread.join(timeout=10)
