"""Fetch small official reference/license files; no runtime import of upstream code."""
import json
from pathlib import Path
import urllib.request

def main():
    out=Path(__file__).resolve().parents[1]/'docs/upstream'
    out.mkdir(parents=True,exist_ok=True)
    urls={
        'robosuite_LICENSE':'https://raw.githubusercontent.com/ARISE-Initiative/robosuite/master/LICENSE',
        'act_LICENSE':'https://raw.githubusercontent.com/tonyzhaozh/act/main/LICENSE',
        'mink_LICENSE':'https://raw.githubusercontent.com/kevinzakka/mink/v0.0.13/LICENSE',
        'mink_dual_panda.py':'https://raw.githubusercontent.com/kevinzakka/mink/v0.0.13/examples/dual_panda.py',
        'mujoco_python.rst':'https://raw.githubusercontent.com/google-deepmind/mujoco/main/doc/python.rst',
    }
    result={}
    for name,url in urls.items():
        try:
            with urllib.request.urlopen(url,timeout=30) as response: data=response.read()
            (out/name).write_bytes(data)
            result[name]={'url':url,'bytes':len(data),'success':True}
        except Exception as exc: result[name]={'url':url,'error':repr(exc),'success':False}
        print(name,result[name],flush=True)
    (out/'research_downloads.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__': main()
