"""Download only the Panda model and referenced assets; preserve upstream license."""
import json
import pathlib
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]

def fetch(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()

def main():
    out = ROOT / 'assets' / 'franka_emika_panda'
    out.mkdir(parents=True, exist_ok=True)
    try:
        sha = json.loads(fetch('https://api.github.com/repos/google-deepmind/mujoco_menagerie/commits/main'))['sha']
    except Exception:
        sha = 'main'  # GitHub API can be rate limited; bundled bytes are hashed below.
    base = f'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{sha}/franka_emika_panda/'
    xml = fetch(base + 'panda.xml')
    (out / 'panda.xml').write_bytes(xml)
    tree = ET.fromstring(xml)
    files = ['LICENSE', 'README.md'] + ['assets/' + e.attrib['file'] for e in tree.findall('.//mesh') if 'file' in e.attrib]
    def download(name):
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(base + name))
        return name
    with ThreadPoolExecutor(max_workers=6) as pool:
        for name in pool.map(download, sorted(set(files))):
            print(name, flush=True)
    import hashlib
    hashes = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out / 'SOURCE.json').write_text(json.dumps({'repository': 'https://github.com/google-deepmind/mujoco_menagerie', 'revision': sha, 'sha256': hashes}, indent=2), encoding='utf-8')
    refs = ROOT / 'docs' / 'upstream'
    refs.mkdir(parents=True, exist_ok=True)
    urls = {
        'robosuite_two_arm_lift.py': 'https://raw.githubusercontent.com/ARISE-Initiative/robosuite/master/robosuite/environments/manipulation/two_arm_lift.py',
        'robosuite_stack.py': 'https://raw.githubusercontent.com/ARISE-Initiative/robosuite/master/robosuite/environments/manipulation/stack.py',
        'robosuite_osc.py': 'https://raw.githubusercontent.com/ARISE-Initiative/robosuite/master/robosuite/controllers/parts/arm/osc.py',
        'act_scripted_policy.py': 'https://raw.githubusercontent.com/tonyzhaozh/act/main/scripted_policy.py',
        'act_sim_env.py': 'https://raw.githubusercontent.com/tonyzhaozh/act/main/sim_env.py',
        'mink_readme.md': 'https://raw.githubusercontent.com/kevinzakka/mink/main/README.md',
        'mujoco_python.html': 'https://mujoco.readthedocs.io/en/stable/python.html',
    }
    # These files are reference material, not imported or copied into the runtime.
    for name, url in urls.items():
        try:
            data = fetch(url)
            (refs / name).write_bytes(data)
            print('reference', name, len(data), flush=True)
        except Exception as exc:
            (refs / (name+'.error.txt')).write_text(f'URL: {url}\nError: {exc!r}\n',encoding='utf-8')
            print('reference unavailable', name, repr(exc), flush=True)
    (refs / 'urls.json').write_text(json.dumps(urls, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
