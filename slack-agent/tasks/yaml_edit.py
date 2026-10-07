import io

from ruamel.yaml import YAML
from ruamel.yaml.util import load_yaml_guess_indent

from tasks.models import FileEdit


def apply_yaml_edits(text: str, edits: list[FileEdit]) -> str:
    """Set dotted keys in a YAML document, preserving comments and indentation.

    Every key in the path must already exist: a typo in the request should fail
    loudly instead of silently adding a new key.
    """
    data, indent, block_seq_indent = load_yaml_guess_indent(text)
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=block_seq_indent, sequence=indent, offset=block_seq_indent)

    for edit in edits:
        node = data
        parts = edit.key.split(".")
        for part in parts[:-1]:
            node = _child(node, part, edit)
        last = parts[-1]
        if isinstance(node, list):
            node[_index(node, last, edit)] = edit.value
        elif isinstance(node, dict) and last in node:
            node[last] = edit.value
        else:
            raise KeyError(f"Key `{edit.key}` not found in `{edit.path}`")

    out = io.StringIO()
    yaml.dump(data, out)
    return out.getvalue()


def _child(node, part: str, edit: FileEdit):
    if isinstance(node, list):
        return node[_index(node, part, edit)]
    if isinstance(node, dict) and part in node:
        return node[part]
    raise KeyError(f"Key `{edit.key}` not found in `{edit.path}`")


def _index(node: list, part: str, edit: FileEdit) -> int:
    if not part.isdigit() or int(part) >= len(node):
        raise KeyError(f"Key `{edit.key}` not found in `{edit.path}`")
    return int(part)
