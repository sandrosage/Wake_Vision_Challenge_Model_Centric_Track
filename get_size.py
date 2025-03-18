import tempfile
import zipfile
from argparse import ArgumentParser
import os

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

parser = ArgumentParser()

parser.add_argument(
    "--path",
    required=True,
    type=str,
    help="Name of the model",
)

args = parser.parse_args()
print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(args.path)))
