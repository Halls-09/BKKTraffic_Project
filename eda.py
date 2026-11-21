"""
Shortcut entry point for Bangkok Traffic Seaborn EDA.

Usage:
  python eda.py          # Generate all 8 Seaborn charts + HTML visual gallery + JSON summary
  python eda.py 1 2      # Regenerate only Chart 1 and Chart 2
  python eda.py --list   # Show all available charts
"""
from src.eda.run_eda import main

if __name__ == "__main__":
    main()
