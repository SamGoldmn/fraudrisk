import logging
import os, sys

ROOT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(ROOT_DIR, "forensics"))

from operations.store import EllipticDataStore

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

force_rebuild = True

store = EllipticDataStore(force_rebuild=force_rebuild)
elliptic_data = store.get_data()
logging.info("Head of features")
elliptic_data.features.head()
logging.info("Head of labels")
elliptic_data.labels.head()

