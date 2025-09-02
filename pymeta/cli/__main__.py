#!/usr/bin/env python3
"""
Main entry point for PyMeta CLI.

Usage:
    pymeta --help
    pymeta meta --help
    pymeta plots --help
    pymeta bias --help
    pymeta live --help
"""

import sys
import argparse
from typing import List, Optional

from .commands import meta, plots, bias, live


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog='pymeta',
        description='PyMeta: Comprehensive meta-analysis toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    pymeta meta --data studies.csv --model random
    pymeta plots forest --data studies.csv --output forest.png
    pymeta bias --data studies.csv --tests egger,begg
    pymeta live init --config config.yaml
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='PyMeta 0.1.0'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # Add subcommands
    meta.add_parser(subparsers)
    plots.add_parser(subparsers)
    bias.add_parser(subparsers)
    live.add_parser(subparsers)
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    
    if args is None:
        args = sys.argv[1:]
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Handle no command
    if not parsed_args.command:
        parser.print_help()
        return 1
    
    # Set verbosity
    if parsed_args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # Execute command
    try:
        if parsed_args.command == 'meta':
            return meta.run(parsed_args)
        elif parsed_args.command == 'plots':
            return plots.run(parsed_args)
        elif parsed_args.command == 'bias':
            return bias.run(parsed_args)
        elif parsed_args.command == 'live':
            return live.run(parsed_args)
        else:
            print(f"Unknown command: {parsed_args.command}", file=sys.stderr)
            return 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())