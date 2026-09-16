import sys
from pathlib import Path

# Make `import inferno` work without packaging the project.
sys.path.insert(0, str(Path(__file__).resolve().parent))
