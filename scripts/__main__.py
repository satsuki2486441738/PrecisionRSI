"""Run repository maintenance without embedding business logic in CI."""
import argparse
import importlib
import sys

COMMANDS = {
    'validate': ('scripts', 'validate'),
    'validate-history': ('community.records', 'validation'),
    'validate-differences': ('community.records', 'differences'),
    'validate-immutability': ('scripts', 'validate_immutability'),
    'collect-differences': ('scripts', 'collect_differences'),
    'plan-history': ('scripts', 'plan_history_integration'),
    'export-statistics': ('scripts', 'export_statistics'),
    'export-paper': ('scripts', 'export_paper_results'),
    'prepare-public': ('scripts', 'prepare_public_results'),
    'rsi-snapshot': ('scripts', 'rsi_snapshot'),
    'install': ('scripts', 'install'),
    'smoke-install': ('scripts', 'smoke_install'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS)
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    # The selected CLI owns all remaining options, including --help.
    package, module = COMMANDS[args.command]
    previous = sys.argv
    try:
        sys.argv = [package + '.' + module, *remaining]
        return importlib.import_module(package + '.' + module).main()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    raise SystemExit(main())
