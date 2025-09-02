"""
Living review command for PyMeta CLI.

Handles living systematic review operations.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
import pandas as pd
import pymeta as pm


def add_parser(subparsers: Any) -> None:
    """Add living review parser to subparsers."""
    parser = subparsers.add_parser(
        'live',
        help='Manage living reviews',
        description='Tools for living systematic reviews'
    )
    
    # Subcommands for living review operations
    live_subparsers = parser.add_subparsers(
        dest='live_command',
        help='Living review operation',
        metavar='OPERATION'
    )
    
    # Initialize new living review
    init_parser = live_subparsers.add_parser('init', help='Initialize new living review')
    init_parser.add_argument('--config', '-c', required=True, type=Path,
                            help='Configuration file (YAML)')
    init_parser.add_argument('--review-id', required=True,
                            help='Unique identifier for the review')
    init_parser.add_argument('--output', '-o', type=Path, default=Path('.'),
                            help='Output directory')
    
    # Update existing living review
    update_parser = live_subparsers.add_parser('update', help='Update living review')
    update_parser.add_argument('--review-id', required=True,
                              help='Review identifier')
    update_parser.add_argument('--data', type=Path,
                              help='New data file (if not using auto-search)')
    update_parser.add_argument('--force', action='store_true',
                              help='Force update even if no new studies')
    
    # Search for new studies
    search_parser = live_subparsers.add_parser('search', help='Search for new studies')
    search_parser.add_argument('--review-id', required=True,
                              help='Review identifier')
    search_parser.add_argument('--source', choices=['pubmed', 'arxiv', 'all'],
                              default='all', help='Search source')
    search_parser.add_argument('--since', help='Search since date (YYYY-MM-DD)')
    
    # Generate living review report
    report_parser = live_subparsers.add_parser('report', help='Generate living review report')
    report_parser.add_argument('--review-id', required=True,
                              help='Review identifier')
    report_parser.add_argument('--output', '-o', type=Path,
                              help='Output file for report')
    report_parser.add_argument('--format', choices=['html', 'pdf', 'markdown'],
                              default='html', help='Report format')
    
    # List living reviews
    list_parser = live_subparsers.add_parser('list', help='List all living reviews')
    list_parser.add_argument('--status', choices=['active', 'paused', 'completed'],
                            help='Filter by status')
    
    # Setup monitoring
    monitor_parser = live_subparsers.add_parser('monitor', help='Setup monitoring')
    monitor_parser.add_argument('--review-id', required=True,
                               help='Review identifier')
    monitor_parser.add_argument('--frequency', choices=['daily', 'weekly', 'monthly'],
                               default='weekly', help='Monitoring frequency')
    monitor_parser.add_argument('--notifications', nargs='+',
                               choices=['email', 'slack', 'webhook'],
                               help='Notification methods')


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_living_review(args: argparse.Namespace) -> int:
    """Initialize a new living review."""
    try:
        print(f"Initializing living review: {args.review_id}")
        
        # Load configuration
        config = load_config(args.config)
        
        # Validate configuration
        required_fields = ['search_strategy', 'inclusion_criteria', 'data_sources']
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
        
        # Create living review instance
        living_review = pm.LivingReview(
            review_id=args.review_id,
            config=config,
            output_dir=args.output
        )
        
        # Initialize directory structure
        living_review.setup_directory_structure()
        
        # Setup initial search monitors
        living_review.setup_search_monitors()
        
        # Create initial database/tracking files
        living_review.initialize_tracking()
        
        print(f"Living review '{args.review_id}' initialized successfully!")
        print(f"Configuration saved to: {args.output / args.review_id / 'config.yaml'}")
        print(f"Next steps:")
        print(f"  1. Run initial search: pymeta live search --review-id {args.review_id}")
        print(f"  2. Setup monitoring: pymeta live monitor --review-id {args.review_id}")
        
        return 0
        
    except Exception as e:
        print(f"Error initializing living review: {e}", file=sys.stderr)
        return 1


def update_living_review(args: argparse.Namespace) -> int:
    """Update an existing living review."""
    try:
        print(f"Updating living review: {args.review_id}")
        
        # Load living review
        living_review = pm.LivingReview.load(args.review_id)
        
        # Check for new studies
        if args.data:
            # Manual data update
            new_data = pd.read_csv(args.data)
            print(f"Loaded {len(new_data)} studies from {args.data}")
        else:
            # Automatic search for new studies
            print("Searching for new studies...")
            new_studies = living_review.search_new_studies()
            if not new_studies and not args.force:
                print("No new studies found. Use --force to update anyway.")
                return 0
            new_data = pd.DataFrame(new_studies)
        
        # Update analysis
        if len(new_data) > 0 or args.force:
            print("Updating meta-analysis...")
            
            # Combine with existing data
            current_data = living_review.get_current_data()
            updated_data = living_review.merge_data(current_data, new_data)
            
            # Perform updated meta-analysis
            new_result = pm.meta_analysis(updated_data, model='random')
            
            # Check for significant changes
            last_result = living_review.get_last_result()
            significant_change = living_review.check_significant_change(last_result, new_result)
            
            # Save updated results
            living_review.save_update(new_result, new_data, significant_change)
            
            # Generate alerts if needed
            if significant_change:
                print("🚨 Significant change detected!")
                living_review.send_alerts(new_result)
            
            print(f"Update completed. New overall effect: {new_result.overall_effect:.4f}")
            print(f"Studies: {len(updated_data)} (added: {len(new_data)})")
        
        return 0
        
    except Exception as e:
        print(f"Error updating living review: {e}", file=sys.stderr)
        return 1


def search_new_studies(args: argparse.Namespace) -> int:
    """Search for new studies."""
    try:
        print(f"Searching for new studies for review: {args.review_id}")
        
        # Load living review
        living_review = pm.LivingReview.load(args.review_id)
        
        # Configure search
        search_config = {
            'sources': [args.source] if args.source != 'all' else ['pubmed', 'arxiv'],
            'since_date': args.since
        }
        
        # Perform search
        new_studies = living_review.search_studies(search_config)
        
        print(f"Search completed. Found {len(new_studies)} potentially relevant studies.")
        
        # Screen studies
        if new_studies:
            print("Screening studies...")
            screened_studies = living_review.screen_studies(new_studies)
            
            included_count = sum(1 for study in screened_studies if study['include'])
            print(f"Screening completed. {included_count} studies included.")
            
            # Save search results
            living_review.save_search_results(screened_studies)
            
            if included_count > 0:
                print("New studies available for inclusion. Run 'pymeta live update' to update analysis.")
        
        return 0
        
    except Exception as e:
        print(f"Error searching for studies: {e}", file=sys.stderr)
        return 1


def generate_report(args: argparse.Namespace) -> int:
    """Generate living review report."""
    try:
        print(f"Generating report for living review: {args.review_id}")
        
        # Load living review
        living_review = pm.LivingReview.load(args.review_id)
        
        # Generate report
        report_content = living_review.generate_report()
        
        # Determine output file
        if args.output:
            output_file = args.output
        else:
            output_file = Path(f"{args.review_id}_report.{args.format}")
        
        # Save report
        if args.format == 'html':
            living_review.save_html_report(report_content, output_file)
        elif args.format == 'pdf':
            living_review.save_pdf_report(report_content, output_file)
        elif args.format == 'markdown':
            living_review.save_markdown_report(report_content, output_file)
        
        print(f"Report saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"Error generating report: {e}", file=sys.stderr)
        return 1


def list_living_reviews(args: argparse.Namespace) -> int:
    """List all living reviews."""
    try:
        print("Living Reviews:")
        print("=" * 50)
        
        # Get all living reviews
        reviews = pm.LivingReview.list_all()
        
        if not reviews:
            print("No living reviews found.")
            return 0
        
        # Filter by status if specified
        if args.status:
            reviews = [r for r in reviews if r['status'] == args.status]
        
        # Display reviews
        for review in reviews:
            print(f"ID: {review['id']}")
            print(f"  Status: {review['status']}")
            print(f"  Created: {review['created_date']}")
            print(f"  Last updated: {review['last_updated']}")
            print(f"  Studies: {review['n_studies']}")
            print(f"  Current effect: {review['current_effect']:.4f}")
            print()
        
        return 0
        
    except Exception as e:
        print(f"Error listing living reviews: {e}", file=sys.stderr)
        return 1


def setup_monitoring(args: argparse.Namespace) -> int:
    """Setup monitoring for living review."""
    try:
        print(f"Setting up monitoring for review: {args.review_id}")
        
        # Load living review
        living_review = pm.LivingReview.load(args.review_id)
        
        # Configure monitoring
        monitor_config = {
            'frequency': args.frequency,
            'notifications': args.notifications or []
        }
        
        # Setup automated monitoring
        living_review.setup_monitoring(monitor_config)
        
        print(f"Monitoring configured:")
        print(f"  Frequency: {args.frequency}")
        print(f"  Notifications: {', '.join(args.notifications or ['none'])}")
        
        # Setup scheduler (platform-specific)
        living_review.setup_scheduler()
        
        print("Automated monitoring is now active.")
        
        return 0
        
    except Exception as e:
        print(f"Error setting up monitoring: {e}", file=sys.stderr)
        return 1


def run(args: argparse.Namespace) -> int:
    """Run living review command."""
    if not args.live_command:
        print("Error: No living review operation specified", file=sys.stderr)
        return 1
    
    if args.live_command == 'init':
        return initialize_living_review(args)
    elif args.live_command == 'update':
        return update_living_review(args)
    elif args.live_command == 'search':
        return search_new_studies(args)
    elif args.live_command == 'report':
        return generate_report(args)
    elif args.live_command == 'list':
        return list_living_reviews(args)
    elif args.live_command == 'monitor':
        return setup_monitoring(args)
    else:
        print(f"Error: Unknown living review operation: {args.live_command}", file=sys.stderr)
        return 1