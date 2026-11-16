"""Run repository maintenance without embedding business logic in CI."""
import argparse
import importlib
import sys

COMMANDS = {
    'validate': ('precision_maintenance', 'validate'),
    'validate-history': ('precision_community', 'validation'),
    'validate-differences': ('precision_community', 'differences'),
    'validate-immutability': ('precision_maintenance', 'validate_immutability'),
    'collect-differences': ('precision_maintenance', 'collect_differences'),
    'plan-history': ('precision_maintenance', 'plan_history_integration'),
    'export-statistics': ('precision_maintenance', 'export_statistics'),
    'export-paper': ('precision_maintenance', 'export_paper_results'),
    'prepare-public': ('precision_maintenance', 'prepare_public_results'),
    'rsi-snapshot': ('precision_maintenance', 'rsi_snapshot'),
    'install': ('precision_maintenance', 'install'),
    'smoke-install': ('precision_maintenance', 'smoke_install'),
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
