# Наш пакет mcp/ является namespace-пакетом: расширяем __path__ так,
# чтобы импорты вида `mcp.types`, `mcp.server.Server` и `mcp.server.stdio`
# находили установленный SDK, а не только наши файлы.
#
# Без этого наш mcp/ затеняет pip-пакет `mcp` (у них одинаковое имя),
# и `import mcp.types` падает с ModuleNotFoundError.
from pathlib import Path
import sys as _sys

_site_mcp = [
    str(p)
    for p in Path(_sys.prefix).glob("lib/*/site-packages/mcp")
    if (p / "types.py").exists()
]
__path__ = list(__path__) + _site_mcp  # type: ignore[name-defined]
