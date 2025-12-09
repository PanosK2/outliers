# config.py

# Διαδρομές (Paths)
ROOT_DIR = r'C:\Users\Takis\Documents\Διπλωματική\DATA'
NWP_SOURCE_DIR = r'C:\Users\Takis\PycharmProjects\PythonProject\nwp'

# Παράμετροι Αλγορίθμου
CUT_IN_SPEED = 3.5
ISO_CONTAMINATION = 0.04

# Ρυθμίσεις Micro-Analysis
WINDOW_SIZE = 60      # 1 ώρα
SIGMA = 5             # 5 τυπικές αποκλίσεις
BUFFER = 1.5          # 1.5 MW buffer
FROZEN_WINDOW = 60    # 1 ώρα

# Επιλογή Μεθόδου για Isolated Mode (Χωρίς Καιρό)
# Επιλογές: 'STATISTICAL' (Μόνο κανόνες/rolling) ή 'ISO_FOREST' (Machine Learning)
ISOLATED_METHOD_TYPE = 'ISO_FOREST'