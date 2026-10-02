import logging

from rich.console import Console
from rich.logging import RichHandler


console = Console()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[RichHandler()]
)

logger = logging.getLogger("CombustionAgent")