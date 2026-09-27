"""Static file server for the monitor, with caching disabled so page edits show up on a normal refresh."""
import http.server


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):  # the page polls every second; keep the console quiet
        pass


print("Serving on http://localhost:8000")
http.server.ThreadingHTTPServer(("", 8000), NoCacheHandler).serve_forever()
