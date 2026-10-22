from dataclasses import dataclass
from enum import Enum


class ProcessType(Enum):
    """Pipeline steps supported by this release (body pipeline)."""
    ma_cap = "ma_cap"
    ma_masks = "ma_masks"
    ma_2d = "ma_2d"
    ma_3d = "ma_3d"
    ma_vis = "ma_vis"

    @classmethod
    def is_valid(cls, value):
        return value in {m.value for m in cls}


@dataclass
class Process:
    """Pipeline (step, sequence) execution row.

    `pid` was historically called `cluster_job_id` (HTCondor-era) — kept
    that name on the DB column for schema stability, but Python and API
    use the generic `pid` instead.
    """
    def __init__(
        self,
        process_id: int,
        sequence_id: int,
        capture_id: int,
        process: ProcessType,
        process_mapping: str,
        validation_mapping: str,
        pid: str,
        image_path: str,
        out_file: str,
        err_file: str,
        status: str,
        created_at: str,
    ):
        self.process_id = process_id
        self.sequence_id = sequence_id
        self.capture_id = capture_id
        self.process = process
        self.process_mapping = process_mapping
        self.validation_mapping = validation_mapping
        self.pid = pid
        self.image_path = image_path
        self.out_file = out_file
        self.err_file = err_file
        self.status = status
        self.created_at = created_at

    def serialize(self):
        return {
            "processId": self.process_id,
            "sequenceId": self.sequence_id,
            "captureId": self.capture_id,
            "processType": self.process,
            "processMapping": self.process_mapping,
            "validationMapping": self.validation_mapping,
            "pid": self.pid,
            "imagePath": self.image_path,
            "outFile": self.out_file,
            "errFile": self.err_file,
            "status": self.status,
            "createdAt": self.created_at,
        }
