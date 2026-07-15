from benchmark.pipelines import AlzheimerLoader


class DataLoader:
    """
    Component for loading resting-state EEG BIDS datasets (ds004504).
    """
    def __init__(self, dataset_path):
        self.loader = AlzheimerLoader(dataset_path)

    @property
    def participants(self):
        """Returns the participant metadata DataFrame."""
        return self.loader.participants

    def get_group(self, subject_id):
        """Returns the group label (A, C, F, or Unknown) for a subject ID."""
        meta = self.loader.get_subject_metadata(subject_id)
        return meta.get('Group', 'Unknown')

    def load(self, subject_id):
        """Loads and returns the raw MNE object for a subject."""
        return self.loader.load_subject(subject_id)

    def load_subject(self, subject_id):
        """Forward to load for backwards compatibility."""
        return self.load(subject_id)
