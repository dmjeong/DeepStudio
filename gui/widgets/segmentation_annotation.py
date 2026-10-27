"""Public semantic annotation entry point, shared by every segmentation model."""

from widgets.annotation_workspace import AnnotationWorkspace


class SegmentationAnnotationDialog(AnnotationWorkspace):
    def __init__(self, image_path, class_names, document, save, parent=None, **kwargs):
        super().__init__(
            image_path, class_names, document, save, parent, task="segment", **kwargs
        )
