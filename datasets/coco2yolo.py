from pylabel import importer
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
dataset_name = input("Enter dataset path: ")
DATA_PATH = os.path.join(ROOT, dataset_name)
json_path = os.path.join(DATA_PATH, 'annotations/instances_default.json')
importer.ImportCoco(json_path).export.ExportToYoloV5(
    output_path=os.path.join(DATA_PATH, 'labels'), yaml_file=dataset_name + '.yaml', cat_id_index=int(0))
