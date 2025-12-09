# config.py
import os

# Βρίσκουμε το directory που βρίσκεται το config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Διαδρομές (Relative Paths)
ROOT_DIR = os.path.join(BASE_DIR, 'sample_data')
NWP_SOURCE_DIR = os.path.join(BASE_DIR, 'nwp')  # Αν έχεις NWP files

# Παράμετροι Αλγορίθμου (Τα ίδια)
CUT_IN_SPEED = 3.5
ISO_CONTAMINATION = 0.04

# Ρυθμίσεις Micro-Analysis
WINDOW_SIZE = 60      
SIGMA = 5             
BUFFER = 1.5          
FROZEN_WINDOW = 60    

# Επιλογή Μεθόδου για Isolated Mode
ISOLATED_METHOD_TYPE = 'ISO_FOREST'
