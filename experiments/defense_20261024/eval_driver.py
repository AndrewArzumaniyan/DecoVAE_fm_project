"""Run an evaluator in this process and write a completion marker only after success."""
import json
from pathlib import Path
import runpy
import sys
from datetime import datetime

marker = Path(sys.argv[1]).resolve()
script = Path(sys.argv[2]).resolve()
root = Path.cwd().resolve()
if not marker.is_relative_to(root) or not script.is_relative_to(root):
    raise SystemExit('Evaluator and completion marker must be inside the owned root')
arguments = sys.argv[3:]
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), *arguments]
runpy.run_path(str(script), run_name='__main__')
marker.write_text(json.dumps({'passed': True, 'script': str(script),
                              'finished': datetime.now().astimezone().isoformat()}, indent=2) + '\n')
