"""Choose the downloaded official full catalogue; does not replace search/scoring."""
import argparse
import re
from pathlib import Path


def configure(root):
    root = Path(root)
    for name in ('items_shuffle.json','items_ins_v2.json'):
        if not (root/'data'/name).is_file():
            raise FileNotFoundError(f'Missing {root / "data" / name}. Run official setup.sh -d all first.')
    path = root/'web_agent_site/utils.py'
    original = path.read_text(encoding='utf-8')
    text = original
    for name, replacement in [('DEFAULT_ATTR_PATH','items_ins_v2.json'),('DEFAULT_FILE_PATH','items_shuffle.json')]:
        pattern = rf'^({name}\s*=\s*[^\n]*)([\r]?)$'
        match = re.search(pattern, text, re.M)
        if not match:
            raise RuntimeError(f'No active {name} assignment. Inspect utils.py for upstream changes.')
        # Only modify its known data filename, preserving join(BASE_DIR, ...) and comments.
        line, count = re.subn(r'items_(?:ins_v2|shuffle)(?:_\d+)?\.json', replacement, match.group(1), count=1)
        if count != 1:
            raise RuntimeError(f'Unexpected {name} assignment. Refusing to guess a path.')
        text = text[:match.start(1)] + line + text[match.end(1):]
    # DEBUG_PROD_SIZE=None means do not cap the selected full data file.
    text, count = re.subn(r'^DEBUG_PROD_SIZE\s*=.*$', 'DEBUG_PROD_SIZE = None', text, count=1, flags=re.M)
    # Some upstream revisions have no separate size cap; their selected file is sufficient.
    compile(text, str(path), 'exec')
    backup = path.with_suffix('.py.expel-backup')
    if not backup.exists():
        backup.write_text(original, encoding='utf-8')
    path.write_text(text, encoding='utf-8')
    print('Selected full downloaded catalogue in:', path)
    print('Backup:', backup)
    print('Restart the WebShop server before testing.')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('webshop_root')
    configure(p.parse_args().webshop_root)
