"""
docs/report_sources/capture_ui_screenshots.py

Drives the REAL web UI (served by the running API at /web/) in headless Microsoft Edge through the Chrome DevTools
Protocol and saves screenshots for the report: connection, trip selection, and the result view for chosen trips.
Also screenshots two raw API responses (/health, /trips) as rendered by the browser.

    1) start the API:  uvicorn api.main:app --host 0.0.0.0 --port 8000      (set DATA_DIR / EVAL_PARTITION for the evaluation trips)
    2) python docs/report_sources/capture_ui_screenshots.py http://<laptop-1-ip>:8000     -> docs/report_sources/screenshots/*.png
Nothing is mocked: every screenshot is taken after the UI has really called the API.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

EDGE = next(p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe") if Path(p).exists())
OUT = Path(__file__).resolve().parent / "screenshots"
PORT = 9333


class Page:
    def __init__(self):
        self.proc = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", f"--remote-debugging-port={PORT}", "--window-size=1500,950",
                                      "--user-data-dir=" + str(OUT / "_edge_profile"), "--no-first-run", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                ws = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        self.ws = connect(ws, max_size=None)
        self.i = 0
        self.call("Page.enable"); self.call("Runtime.enable")
        self.call("Emulation.setDeviceMetricsOverride", width=1500, height=950, deviceScaleFactor=1, mobile=False)

    def call(self, method, **params):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.i:
                return m.get("result", {})

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    def goto(self, url, wait=4):
        self.call("Page.navigate", url=url); time.sleep(wait)

    def shot(self, name, height=None):
        kw = {"clip": {"x": 0, "y": 0, "width": 1500, "height": height, "scale": 1}} if height else {}
        data = self.call("Page.captureScreenshot", format="png", **kw)["data"]
        (OUT / name).write_bytes(base64.b64decode(data)); print("saved", name)

    def close(self):
        try:
            self.ws.close()
        finally:
            self.proc.terminate()


def main(base):
    OUT.mkdir(exist_ok=True)
    p = Page()
    try:
        # raw API responses as rendered by a browser
        p.goto(base + "/health", 2); p.shot("api_health.png", height=70)
        p.goto(base + "/trips", 2); p.shot("api_trips.png", height=120)

        # the evaluator UI
        p.goto(base + "/web/", 5)
        p.js("document.getElementById('btnTestConn').click()"); time.sleep(4)
        p.shot("ui_1_connected.png")
        trips = p.js("Array.from(document.querySelectorAll('#tripSelect option')).map(o=>o.value).filter(v=>v)")
        print("trips in dropdown:", trips)
        for tid in ("00000015", "00000049"):
            if tid not in trips:
                continue
            p.js(f"(()=>{{const s=document.getElementById('tripSelect'); s.value='{tid}'; s.dispatchEvent(new Event('change'));}})()")
            time.sleep(6)
            if tid == "00000015":
                p.shot("ui_2_trip_selected.png")
            p.js("document.getElementById('btnRunPredict').click()")
            for _ in range(40):                       # wait for the result drawer to be populated
                time.sleep(1.5)
                if (p.js("document.getElementById('hmmAcc') && document.getElementById('hmmAcc').textContent") or "").strip() not in ("", "--", "-"):
                    break
            time.sleep(4)
            p.shot(f"ui_3_result_{tid}.png")
            print(tid, "HMM acc shown:", p.js("document.getElementById('hmmAcc').textContent"), "| classifier:", p.js("document.getElementById('clfAcc').textContent"))
    finally:
        p.close()


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000")
