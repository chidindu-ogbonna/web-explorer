import logging

base_logger = logging.getLogger("web_explorer")
base_logger.setLevel(logging.INFO)
base_logger.propagate = False  # prevent double logging

if not base_logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s: %(name)s: %(message)s"))
    base_logger.addHandler(ch)
