#!/usr/bin/env python3
"""Validate installable skill metadata and published Markdown links/heading anchors.

Requires .github/requirements.txt. Only tracked Markdown is checked by default;
external links are not fetched. Link/image references use Markdown parsing.
"""
import argparse
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
import yaml


def validate_metadata(path):
    text = path.read_text(encoding='utf-8')
    parts = text.split('---', 2) if text.startswith('---\n') else []
    if len(parts) != 3:
        raise ValueError(f'{path}: missing YAML frontmatter')
    data = yaml.safe_load(parts[1])
    if not isinstance(data, dict):
        raise ValueError(f'{path}: frontmatter must be a mapping')
    allowed = {'name', 'description', 'license', 'compatibility', 'metadata', 'allowed-tools'}
    if set(data) - allowed:
        raise ValueError(f'{path}: unknown metadata fields {set(data) - allowed}')
    name = data.get('name', '')
    if (not isinstance(name, str) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name)
            or len(name) > 64 or name != path.parent.name):
        raise ValueError(f'{path}: invalid name or directory mismatch')
    for key, limit in [('description', 1024), ('compatibility', 500)]:
        value = data.get(key)
        if (key == 'description' or key in data) and (
                not isinstance(value, str) or not value.strip() or len(value) > limit):
            raise ValueError(f'{path}: invalid {key}')
    for key in ('license', 'allowed-tools'):
        if key in data and not isinstance(data[key], str):
            raise ValueError(f'{path}: {key} must be a string')
    if 'metadata' in data and (not isinstance(data['metadata'], dict) or
            any(not isinstance(k, str) or not isinstance(v, str) for k, v in data['metadata'].items())):
        raise ValueError(f'{path}: metadata must map strings to strings')


def markdown(path):
    text = path.read_text(encoding='utf-8')
    if text.startswith('---\n'):
        text = text.split('---', 2)[-1]
    return MarkdownIt('commonmark').parse(text)


def anchors(path):
    tokens, used, found = markdown(path), set(), set()
    for index, token in enumerate(tokens):
        if token.type != 'heading_open':
            continue
        children = tokens[index + 1].children or []
        title = ''.join(t.content for t in children if t.type in ('text', 'code_inline', 'image'))
        slug = re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-')
        base, suffix = slug, 0
        while slug in used:
            suffix += 1
            slug = f'{base}-{suffix}'
        used.add(slug)
        found.add(slug)
    return found


def validate_links(root, paths):
    errors, count, cache = [], 0, {}
    for source in paths:
        for token in markdown(source):
            for child in token.children or []:
                href = child.attrGet('href') if child.type == 'link_open' else (
                    child.attrGet('src') if child.type == 'image' else None)
                if href is None:
                    continue
                url = urlsplit(href)
                if url.scheme or url.netloc:
                    continue
                count += 1
                target = (root / unquote(url.path).lstrip('/') if url.path.startswith('/') else
                          source.parent / unquote(url.path)) if url.path else source
                target = target.resolve()
                if not target.is_relative_to(root.resolve()) or not target.exists():
                    errors.append(f'{source.relative_to(root)}: missing local target {href}')
                elif url.fragment and target.suffix.lower() == '.md':
                    if target not in cache:
                        cache[target] = anchors(target)
                    if unquote(url.fragment) not in cache[target]:
                        errors.append(f'{source.relative_to(root)}: missing heading {href}')
    return count, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    validate_metadata(root / 'skills/precision-skill/SKILL.md')
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    paths = [root / p for p in tracked if p.endswith('.md')]
    count, errors = validate_links(root, paths)
    if errors:
        raise ValueError('\n'.join(errors))
    print(f'Skill metadata valid; checked {count} local links in {len(paths)} Markdown files.')


if __name__ == '__main__':
    main()
