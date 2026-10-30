"""Screenshot every page at desktop and phone width, logged in as the demo account.

    python tools/screenshots.py before        # writes screenshots/before/*.png
    python tools/screenshots.py after

Needs: pip install playwright, then: python -m playwright install chromium
Starts its own Streamlit server on port 8502 and stops it afterwards. The AI advisor shot asks
"How can I save $200 this month?" through the real local model, so Ollama must be running.
"""
import pathlib
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
PORT = 8502
URL = f"http://localhost:{PORT}"
PAGES = ["Overview", "Plan ahead", "Transactions", "AI advisor"]
SIZES = {"desktop": (1440, 2200), "phone": (390, 4200)}   # tall, because Streamlit scrolls inside the page


def wait_for_server():
    for _ in range(120):
        try:
            urllib.request.urlopen(URL, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise SystemExit("Streamlit didn't start.")


def settle(page, seconds=3.0):
    """Wait until Streamlit has finished rerunning."""
    time.sleep(0.5)
    page.wait_for_function("!document.querySelector('[data-testid=\"stStatusWidget\"] [data-testid=\"stSpinner\"]')",
                           timeout=240_000)
    page.wait_for_function("document.querySelectorAll('[data-stale=\"true\"]').length === 0", timeout=240_000)
    time.sleep(seconds)


def go(page, name):
    page.get_by_text(name, exact=True).first.click()
    settle(page)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "after"
    out = ROOT / "screenshots" / label
    out.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "src/dashboard.py", "--server.headless",
                               "true", "--server.port", str(PORT)], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_server()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(URL)
            page.get_by_label("Username", exact=True).wait_for(timeout=180_000)   # first load imports torch
            page.get_by_label("Username", exact=True).fill("demo")
            page.get_by_label("Password", exact=True).fill("demo1234")      # local test account from make_demo_account.py
            page.get_by_role("button", name="Log in").click()
            settle(page, 6)
            for i, name in enumerate(PAGES, start=1):
                page.set_viewport_size({"width": 1440, "height": 900})
                go(page, name)
                if name == "AI advisor":
                    page.get_by_role("button", name="How can I save $200 this month?").first.click()
                    page.get_by_text("Answered by").first.wait_for(timeout=300_000)
                    settle(page)
                slug = name.lower().replace(" ", "_")
                for size, (w, h) in SIZES.items():
                    page.set_viewport_size({"width": w, "height": h})
                    settle(page, 2)
                    path = out / f"{i}_{slug}_{size}.png"
                    page.screenshot(path=str(path))
                    print(path.relative_to(ROOT))
            browser.close()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
