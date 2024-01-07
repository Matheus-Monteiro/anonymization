from os import walk
from os.path import join


def find_csv_files(root_folder: str) -> list[str]:
    csv_files = []
    for root, _, files in walk(root_folder):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(join(root, file))

    csv_files = sorted(csv_files, key=lambda x: "original_data" not in x)

    return csv_files
