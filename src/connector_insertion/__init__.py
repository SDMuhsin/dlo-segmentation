"""3D connector insertion for the PointWire synthetic renderer (see README.md).

Importing the package pulls in numpy/scipy/sklearn only. The mesh-processing
modules (``assets``, ``build_library``) need trimesh and are imported on demand.
"""
from .config import ConnectorInsertionConfig
from .constants import INSERTED_CONNECTOR_LABEL
from .insert import ConnectorInsertion, plan_connector_insertion
from .library import ConnectorAsset, ConnectorLibrary

__all__ = ["ConnectorInsertionConfig", "ConnectorInsertion", "ConnectorAsset",
           "ConnectorLibrary", "INSERTED_CONNECTOR_LABEL", "plan_connector_insertion"]
