from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
import cv2

class Room(XFormPrim):
    def __init__(self, prim_path, name, usd_path=None, translation=None, orientation=None):
        self._usd_path = usd_path
        self._name = name
        self.occ_map = None
        
        # if no room path is given
        if self._usd_path is None:
            return

        self._occ_path = self._usd_path.replace(".usd", ".png") 
        
        # try:
        #     self.occ_map = cv2.imread(self._occ_path, cv2.IMREAD_GRAYSCALE)
        # except:

        add_reference_to_stage(self._usd_path, prim_path)
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
        )