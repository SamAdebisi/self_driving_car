#!/usr/bin/env python3
"""
Main Entry Point for Self-Driving Car Simulation
Industry-standard implementation with comprehensive error handling
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from self_driving_car.simulation.app import CarApp
from self_driving_car.utils.logger_setup import setup_logging


def main():
    """Main entry point"""
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    try:
        logger.info("Starting Self-Driving Car Simulation")
        logger.info("=" * 50)
        
        # Create and run application
        app = CarApp()
        app.run()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

