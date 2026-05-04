"""Level C: Medical heart model from CT/MRI segmentation data."""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

from src.heart_simulation.heartbeat_engine import HeartbeatEngine
from src.config import load_heart_config


class MedicalHeartModel:
    """Heart model reconstructed from CT/MRI medical imaging data.

    Pipeline:
    1. Load DICOM/NIfTI medical scan
    2. Segment heart structures (TotalSegmentator / nnU-Net)
    3. Convert segmentation mask → 3D mesh (marching cubes)
    4. Export to .glb/.obj for web viewer

    Supported datasets:
    - MMWHS (Multi-Modality Whole Heart Segmentation)
    - ACDC (Automated Cardiac Diagnosis Challenge)
    - Custom CT/MRI DICOM files
    """

    HEART_STRUCTURES = {
        1: "left_ventricle_blood",
        2: "right_ventricle_blood",
        3: "left_atrium_blood",
        4: "right_atrium_blood",
        5: "myocardium",
        6: "aorta",
        7: "pulmonary_artery",
    }

    def __init__(self, data_dir: Optional[str | Path] = None):
        self.config = load_heart_config()
        self.data_dir = Path(data_dir) if data_dir else None
        self.mesh = None
        self.segmentation = None
        logger.info("MedicalHeartModel initialized (Level C)")

    def load_nifti(self, nifti_path: str | Path) -> np.ndarray:
        """Load NIfTI medical image file."""
        try:
            import nibabel as nib
            img = nib.load(str(nifti_path))
            data = img.get_fdata()
            logger.info(f"Loaded NIfTI: shape={data.shape}, dtype={data.dtype}")
            return data
        except ImportError:
            logger.error("nibabel not installed. Run: pip install nibabel")
            raise

    def segment_heart(self, volume: np.ndarray, method: str = "threshold") -> np.ndarray:
        """Segment heart from CT/MRI volume.

        For production: use TotalSegmentator or nnU-Net.
        This provides a basic threshold method as fallback.
        """
        if method == "totalsegmentator":
            return self._segment_totalsegmentator(volume)
        elif method == "threshold":
            return self._segment_threshold(volume)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _segment_threshold(self, volume: np.ndarray) -> np.ndarray:
        """Basic threshold segmentation (demo only)."""
        # Normalize to 0-255
        normalized = ((volume - volume.min()) / (volume.max() - volume.min()) * 255).astype(np.uint8)
        # Simple threshold for heart region (CT Hounsfield units: 30-300 for soft tissue)
        mask = np.logical_and(normalized > 80, normalized < 200).astype(np.uint8)
        logger.info(f"Threshold segmentation: {mask.sum()} voxels")
        return mask

    def _segment_totalsegmentator(self, volume: np.ndarray) -> np.ndarray:
        """Segment using TotalSegmentator (requires separate installation)."""
        try:
            from totalsegmentator.python_api import totalsegmentator
            logger.info("Running TotalSegmentator...")
            # This requires the input as a NIfTI file path
            raise NotImplementedError(
                "TotalSegmentator integration requires file-based input. "
                "Use CLI: totalsegmentator -i input.nii.gz -o output_dir --task cardiac"
            )
        except ImportError:
            logger.error("TotalSegmentator not installed. Run: pip install TotalSegmentator")
            raise

    def mesh_from_segmentation(
        self,
        segmentation: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> "trimesh.Trimesh":
        """Convert segmentation mask to 3D mesh using marching cubes."""
        try:
            import trimesh
            from skimage.measure import marching_cubes
        except ImportError:
            logger.error("Install: pip install trimesh scikit-image")
            raise

        verts, faces, normals, _ = marching_cubes(
            segmentation,
            level=0.5,
            spacing=spacing,
            step_size=2,  # Reduce for higher quality
        )

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

        # Clean up mesh
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.fill_holes()

        # Simplify if too many faces
        if len(mesh.faces) > 100000:
            mesh = mesh.simplify_quadric_decimation(50000)
            logger.info(f"Simplified mesh to {len(mesh.faces)} faces")

        logger.info(f"Generated mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        self.mesh = mesh
        return mesh

    def export_mesh(self, output_path: str | Path, file_format: str = "glb"):
        """Export mesh to file (.glb, .obj, .ply, .stl)."""
        if self.mesh is None:
            raise ValueError("No mesh generated. Run mesh_from_segmentation first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if file_format == "glb":
            self.mesh.export(str(output_path), file_type="glb")
        elif file_format == "obj":
            self.mesh.export(str(output_path), file_type="obj")
        elif file_format == "ply":
            self.mesh.export(str(output_path), file_type="ply")
        elif file_format == "stl":
            self.mesh.export(str(output_path), file_type="stl")

        logger.info(f"Mesh exported: {output_path}")

    def get_animation_config(self, engine: HeartbeatEngine, condition: str = "normal") -> dict:
        """Generate animation config for medical heart model."""
        base_params = engine.get_animation_params()
        sim_config = self.config.get("simulation", {})
        colors = sim_config.get("colors", {})

        # Medical-specific visualization
        return {
            "level": "medical",
            "modelPath": None,  # Set after export
            "heartbeat": base_params,
            "structures": self.HEART_STRUCTURES,
            "color": colors.get(condition, "#e74c3c"),
            "renderMode": "surface",  # surface, wireframe, xray
            "opacity": 0.85,
            "crossSection": {
                "enabled": False,
                "plane": "sagittal",  # sagittal, coronal, axial
            },
            "damageVisualization": condition == "infarction",
            "viewer": self.config.get("viewer", {}),
        }

    def process_pipeline(
        self,
        nifti_path: str | Path,
        output_glb: str | Path,
        method: str = "threshold",
    ) -> dict:
        """Full pipeline: NIfTI → segmentation → mesh → export."""
        logger.info(f"Processing medical heart: {nifti_path}")

        volume = self.load_nifti(nifti_path)
        segmentation = self.segment_heart(volume, method=method)
        mesh = self.mesh_from_segmentation(segmentation)
        self.export_mesh(output_glb, file_format="glb")

        return {
            "volume_shape": volume.shape,
            "segmentation_voxels": int(segmentation.sum()),
            "mesh_vertices": len(mesh.vertices),
            "mesh_faces": len(mesh.faces),
            "output_path": str(output_glb),
        }
