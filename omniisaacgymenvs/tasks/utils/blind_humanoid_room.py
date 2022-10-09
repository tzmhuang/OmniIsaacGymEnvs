from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage

class Room(XFormPrim):
    def __init__(self, prim_path, name, usd_path=None, translation=None, orientation=None):
        self._usd_path = usd_path
        self._name = name
        self._usd_path = usd_path

        if self._usd_path is None:
            self._usd_path = "/data/zanming/Omniverse/USD_exports/room_simple_25b25.usd"

        add_reference_to_stage(self._usd_path, prim_path)
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
        )