import os

# Define the paths directly in the script
subset_dir = "data/raw/2025省市文件_无表格"
fullset_dir = "data/raw/2025省市文件_全"

def find_missing_files(subset_dir, fullset_dir):
    subset_files = set(os.listdir(subset_dir))
    fullset_files = set(os.listdir(fullset_dir))

    missing_files = fullset_files - subset_files

    if missing_files:
        print("Missing files:")
        for file in sorted(missing_files):
            print(file)
    else:
        print("No missing files. The subset contains all files from the full set.")

if __name__ == "__main__":
    find_missing_files(subset_dir, fullset_dir)