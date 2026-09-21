"""Capture deterministic screenshots of the running Streamlit walkthrough."""

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


def _screenshot(page: Page, path: Path) -> None:
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(500)
    page.screenshot(path=path, full_page=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    parser.add_argument("--browser-channel", default="chrome")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel=args.browser_channel,
            headless=True,
            args=["--hide-scrollbars", "--force-device-scale-factor=1"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.goto(args.url, wait_until="domcontentloaded")
        page.get_by_text("From data to a monitored prediction", exact=False).wait_for(
            timeout=30_000
        )
        _screenshot(page, args.output / "ui-overview.png")

        page.get_by_text("2 · Train model", exact=True).click()
        page.get_by_text("Train, track, and apply the quality gate", exact=False).wait_for()
        _screenshot(page, args.output / "ui-training.png")

        page.get_by_text("3 · Review evaluation", exact=True).click()
        page.get_by_text("Held-out confusion matrix", exact=True).wait_for()
        _screenshot(page, args.output / "ui-evaluation.png")

        page.get_by_text("4 · Try a prediction", exact=True).click()
        page.get_by_role("button", name="Predict species").click()
        page.get_by_text("Predicted species", exact=False).wait_for()
        _screenshot(page, args.output / "ui-prediction.png")

        page.get_by_text("5 · Check drift", exact=True).click()
        page.get_by_role("button", name="Run simulated drift check").click()
        page.get_by_text("Drift detected", exact=False).wait_for()
        _screenshot(page, args.output / "ui-drift.png")
        browser.close()


if __name__ == "__main__":
    main()
