import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import json
from dualstack.data.episodes import inspect_episode

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('path'); a=p.parse_args()
    result=inspect_episode(a.path)
    Path(a.path).with_suffix('.check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    sys.exit(0 if result['valid'] else 1)
