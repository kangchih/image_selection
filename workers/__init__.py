from .image_selection import ImageSelectionWorker

def use_worker(key):
    key = key.lower()
    available_workers = {
        "is": ImageSelectionWorker
        # "image_selection_batch": ImageSelectionBatchWorker,
    }
    if key not in available_workers.keys():
        raise KeyError(f"[use_worker] worker key {key} is invalid, available keys: {available_workers.keys()}")
    return available_workers[key]