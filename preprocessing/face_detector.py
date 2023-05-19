from typing import List
from facenet_pytorch.models.mtcnn import MTCNN
from torch.utils.data import Dataset


class ImageFaceDetector():
    def __init__(self, device, thresholds = [0.60, 0.60, 0.60]) -> None:
        super().__init__()
        self.detector =  MTCNN(
            thresholds=thresholds,
            margin=0,
            device=device,
        )

    def _detect_faces(self, images) -> List:
        boxes, *_ = self.detector.detect(images, landmarks=False)
        if boxes is None:
            return []
        return [b.tolist() if b is not None else None for b in boxes]

    @property
    def _batch_size(self):
        return 1



class ImageDataset(Dataset):

    def __init__(self, images) -> None:
        super().__init__()
        self.images = images

    def __getitem__(self, index: int):
        image = self.images[index]
        return image

    def __len__(self) -> int:
        return len(self.images)