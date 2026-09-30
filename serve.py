"""Result pages at http://localhost:8000: the paper-trading log and the backtests.

Only the start page, backtest.html and the result files in runs/ are served (no directory listings, no other files),
with caching off so a refresh shows new data.
Usage: python serve.py [port]
"""
import http.server
import os
import sys

VIEWS = [("Paper trading (live, since 28 Sep 2026)", "rank_paper"),
         ("Backtest: fills 1 hour after the decision (main result)", "rank_pit20_d1"),
         ("Replay of September 2026 (never used to build the system)", "rank_replay_sep"),
         ("Backtest: fills at the 00:00 close", "rank_pit20"),
         ("Backtest: fills 3 hours after the decision", "rank_pit20_d3")]


class Handler(http.server.SimpleHTTPRequestHandler):
    def allowed(self):
        path = self.path.split("?")[0]
        name = path[len("/runs/"):]
        return path == "/backtest.html" or (path.startswith("/runs/") and name.endswith(".json") and
                                            "/" not in name and ".." not in name)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            links = "".join(f'<li><a href="backtest.html?run={run}">{label}</a></li>' for label, run in VIEWS
                            if os.path.exists(f"runs/{run}.json"))
            body = (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Crypto ranking</title>'
                    f'<meta name="viewport" content="width=device-width, initial-scale=1"></head>'
                    f'<body style="font-family: system-ui, sans-serif; margin: 24px; line-height: 1.8">'
                    f'<h2>Crypto ranking</h2><ul>{links}</ul></body></html>').encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.allowed():
            super().do_GET()
        else:
            self.send_error(404)

    def do_HEAD(self):
        if self.allowed():
            super().do_HEAD()
        else:
            self.send_error(404)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
print(f"Serving on http://localhost:{port}")
http.server.ThreadingHTTPServer(("", port), Handler).serve_forever()
