import os
from pathlib import Path

# Adjust the root data directory according to config.py
ROOT_DATA_DIR = Path(__file__).resolve().parent / "datasets" / "glaucoma"

directories = [
    # ACRIMA
    "ACRIMA/database",
    
    # DRISHTI-GS1
    "DRISHTI-GS1/Drishti_GS_trainingData/Images",
    "DRISHTI-GS1/Drishti_GS_trainingData/GT",
    "DRISHTI-GS1/Drishti_GS_testingData/Images",
    "DRISHTI-GS1/Drishti_GS_testingData/GT",
    
    # RIM-ONE DL
    "RIM-ONE_DL/Train/Glaucoma",
    "RIM-ONE_DL/Train/Normal",
    "RIM-ONE_DL/Test/Glaucoma",
    "RIM-ONE_DL/Test/Normal",
    
    # EyePACS-AIROGS
    "EyePACS-AIROGS/train/images",
    "EyePACS-AIROGS/test/images",
]

def create_structure():
    print(f"Creating dataset structure at: {ROOT_DATA_DIR}")
    for d in directories:
        dir_path = ROOT_DATA_DIR / d
        dir_path.mkdir(parents=True, exist_ok=True)
        # Create a .gitkeep file so git tracks these empty directories
        (dir_path / ".gitkeep").touch()
        print(f"Created/Verified: {d}")
        
    print("\nDataset directories created successfully.")

if __name__ == "__main__":
    create_structure()
