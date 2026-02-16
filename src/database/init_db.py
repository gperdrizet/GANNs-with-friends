"""
Database initialization script.
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager


def load_config(config_path: str = 'config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_db_url(db_config: dict) -> str:
    """Build PostgreSQL connection URL from config."""
    return (
        f"postgresql://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )


def main():
    """Initialize database tables."""
    print('Loading configuration...')
    config = load_config()
    
    print('Connecting to database...')
    db_url = build_db_url(config['database'])
    db_manager = DatabaseManager(db_url)
    
    print('Creating tables...')
    db_manager.init_database()
    
    print('Initializing training state...')
    db_manager.init_training_state()
    
    print('Database initialized successfully!')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Initialize distributed GAN database')
    parser.add_argument('--reset', action='store_true', 
                       help='Reset database (WARNING: deletes all data)')
    args = parser.parse_args()
    
    if args.reset:
        confirm = input("WARNING: This will delete all data. Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            config = load_config()
            db_url = build_db_url(config['database'])
            db_manager = DatabaseManager(db_url)
            print('Resetting database...')
            db_manager.reset_database()
            db_manager.init_training_state()
            print('Database reset complete!')
        else:
            print('Reset cancelled.')
    else:
        main()
