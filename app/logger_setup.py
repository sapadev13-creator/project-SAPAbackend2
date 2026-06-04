import logging
import os
import sys

LOG_DIR = "sapadev/tmp"
os.makedirs(LOG_DIR, exist_ok=True)


class _SafeConsoleHandler(logging.StreamHandler):
    """Console handler yang aman di Windows (cp1252) saat ada emoji/unicode."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                enc = getattr(stream, "encoding", None) or "utf-8"
                safe = msg.encode(enc, errors="replace").decode(enc, errors="replace")
                stream.write(safe + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def _configure_stdout_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_stdout_utf8()

logger = logging.getLogger("sapadev")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    _file_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    _console_fmt = logging.Formatter("INFO:     %(message)s")

    file_handler = logging.FileHandler(
        f"{LOG_DIR}/app.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(_file_fmt)
    logger.addHandler(file_handler)

    console_handler = _SafeConsoleHandler(sys.stdout)
    console_handler.setFormatter(_console_fmt)
    logger.addHandler(console_handler)
