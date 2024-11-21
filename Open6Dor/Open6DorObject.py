import os
import numpy as np
from robosuite.models.objects import MujocoXMLObject
from libero.libero.envs.base_object import register_object

import json

import re
class CustomObjects(MujocoXMLObject):
    def __init__(self, custom_path, name, obj_name, joints=[dict(type="free", damping="0.0005")], goal_position = None, goal_quat = None):
        # make sure custom path is an absolute path
        assert(os.path.isabs(custom_path)), "Custom path must be an absolute path"
        # make sure the custom path is also an xml file
        assert(custom_path.endswith(".xml")), "Custom path must be an xml file"
        super().__init__(
            custom_path,
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.goal_quat = goal_quat
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.object_properties = {"vis_site_names": {}}
    # TODO: Wenyao add the realtionship between this_position and other_position
    def under(self, this_position, this_mat, other_position, other_height=0.10):
        """
        Checks whether an object is on this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """

        delta_position = this_mat @ (other_position - this_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return total_size[2] - 0.005 < delta_position[2] < total_size[
            2
        ] + other_height and np.all(np.abs(delta_position[:2]) < total_size[:2])
        
    def left(self, this_position, this_mat, other_position, other_height=0.10):
        """
        Checks whether the object is placed left within this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """

        # (TODO) Yifeng: The transformation for size is a little bit
        # hacky at the moment. Will dig deeper into it.

        delta_position = this_mat @ (this_position - other_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return delta_position[0]<0 
    
    
    def right(self, this_position, this_mat, other_position):
        """
        Checks whether the object is contained within this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """


        delta_position = this_mat @ (this_position - other_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return delta_position[0]>0 
    
    
    
    def front(self, this_position, this_mat, other_position):
        """
        Checks whether the object is contained within this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """

        # (TODO) Yifeng: The transformation for size is a little bit
        # hacky at the moment. Will dig deeper into it.

        delta_position = this_mat @ (this_position - other_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return delta_position[1]>0 
    
    def behind(self, this_position, this_mat, other_position):
        """
        Checks whether the object is contained within this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """
        # (TODO) Yifeng: The transformation for size is a little bit
        # hacky at the moment. Will dig deeper into it.
        delta_position = this_mat @ (this_position - other_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return delta_position[1]<0 
    
    
    def top(self, this_position, this_mat, other_position):
        """
        Checks whether the object is contained within this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """
        # (TODO) Yifeng: The transformation for size is a little bit
        # hacky at the moment. Will dig deeper into it.
        delta_position = this_mat @ (other_position - this_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return delta_position[1]<0 
    
    
    def quat(self, this_position, this_mat, goal_quat):
        """
        Checks whether the object is contained within this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """
        # (TODO) Yifeng: The transformation for size is a little bit
        # hacky at the moment. Will dig deeper into it.
        delta_position = this_mat @ (other_position - this_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return delta_position[1]<0 
    


# Auto-generated classes

@register_object
class Toy(CustomObjects):
    def __init__(self, name='toy', obj_name='toy', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '9adc77036b434348ae049776c50df624', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Mug(CustomObjects):
    def __init__(self, name='mug', obj_name='mug', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '9321c45cb9cf459f9f803507d3a11fb3', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Mouse(CustomObjects):
    def __init__(self, name='mouse', obj_name='mouse', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'f017c73a85484b6da3a66ec1cda70c71', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Eraser(CustomObjects):
    def __init__(self, name='eraser', obj_name='eraser', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'f0916a69ba514ecc973b44528b9dcc43', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Wrench(CustomObjects):
    def __init__(self, name='wrench', obj_name='wrench', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                # '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '5bf781c9fd8d4121b735503b13ba2eaf', 'material', 'material.xml'
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/ycb_16k_backup', '042_adjustable_wrench_google_16k', 'textured', 'textured.xml',
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class SdCard(CustomObjects):
    def __init__(self, name='sd_card', obj_name='sd_card', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '02f7f045679d402da9b9f280030821d4', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bottle(CustomObjects):
    def __init__(self, name='bottle', obj_name='bottle', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '8c0f8088a5d546c886d1b07e3f4eafae', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Knife(CustomObjects):
    def __init__(self, name='knife', obj_name='knife', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '85db1d392a89459dbde5cdd951c4f6fb', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Lighter(CustomObjects):
    def __init__(self, name='lighter', obj_name='lighter', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'fd058916fb8843c09b4a2e05e3ee894e', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cup(CustomObjects):
    def __init__(self, name='cup', obj_name='cup', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/ycb_16k_backup', '065-a_cups_google_16k', 'textured', 'textured.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Pot(CustomObjects):
    def __init__(self, name='pot', obj_name='pot', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'cba3e4df7c354d22b7913d45a28a8159', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Box(CustomObjects):
    def __init__(self, name='box', obj_name='box', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '9660e0c0326b4f7386014e27717231ae', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Spoon(CustomObjects):
    def __init__(self, name='spoon', obj_name='spoon', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/ycb_16k_backup', '031_spoon_google_16k', 'textured', 'textured.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Can(CustomObjects):
    def __init__(self, name='can', obj_name='can', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/ycb_16k_backup', '002_master_chef_can_google_16k', 'textured', 'textured.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Pear(CustomObjects):
    def __init__(self, name='pear', obj_name='pear', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'f3dd3064db4f4e8880344425970cecad', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Keyboard(CustomObjects):
    def __init__(self, name='keyboard', obj_name='keyboard', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '0161b6232216457aa8f4f3393960ec85', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Skillet(CustomObjects):
    def __init__(self, name='skillet', obj_name='skillet', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/ycb_16k_backup', '027_skillet_google_16k', 'textured', 'textured.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Calculator(CustomObjects):
    def __init__(self, name='calculator', obj_name='calculator', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'b4bb8bd7791648fdb36c458fd9a877bf', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class PitcherBase(CustomObjects):
    def __init__(self, name='pitcher_base', obj_name='pitcher_base', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/ycb_16k_backup', '019_pitcher_base_google_16k', 'textured', 'textured.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Headphone(CustomObjects):
    def __init__(self, name='headphone', obj_name='headphone', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'efb1727dee214d49b88c58792e4bdffc', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class TissueBox(CustomObjects):
    def __init__(self, name='tissue_box', obj_name='tissue_box', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'dc4c91abf45342b4bb8822f50fa162b2', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Fork(CustomObjects):
    def __init__(self, name='fork', obj_name='fork', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'e98e7fb33743442383812b68608f7006', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class EnvelopeBox(CustomObjects):
    def __init__(self, name='envelope_box', obj_name='envelope_box', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '40b28d1cbfbd4e9f9acf653b748324ee', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Camera(CustomObjects):
    def __init__(self, name='camera', obj_name='camera', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '6a63fc4af48b453c91ca2b335a4d464d', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class MobilePhone(CustomObjects):
    def __init__(self, name='mobile_phone', obj_name='mobile_phone', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'bad5e04d290841779dad272897f384e1', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl(CustomObjects):
    def __init__(self, name='bowl', obj_name='bowl', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'bf83213efac34b3cb2ad6ac5ddaf05d9', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Watch(CustomObjects):
    def __init__(self, name='watch', obj_name='watch', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'd6e38c73f8f2402aac890e8ebb115302', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Wallet(CustomObjects):
    def __init__(self, name='wallet', obj_name='wallet', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'ccfcc4ae6efa44a2ba34c4c479be7daf', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class GlueGun(CustomObjects):
    def __init__(self, name='glue_gun', obj_name='glue_gun', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'a1da12b81f034db894496c1ffc4be1f5', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Binder(CustomObjects):
    def __init__(self, name='binder', obj_name='binder', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'a0026e7d1b244b9a9223daf4223c9372', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Pen(CustomObjects):
    def __init__(self, name='pen', obj_name='pen', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'fbeb7c776372466daffa619106e0b2e0', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Hammer(CustomObjects):
    def __init__(self, name='hammer', obj_name='hammer', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/ycb_16k_backup', '048_hammer_google_16k', 'textured', 'textured.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Book(CustomObjects):
    def __init__(self, name='book', obj_name='book', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'c61227cac7224b86b43c53ac2a2b6ec7', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class RemoteControl(CustomObjects):
    def __init__(self, name='remote_control', obj_name='remote_control', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'f1bbc61a42b94ee9a2976ca744f8962e', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Microphone(CustomObjects):
    def __init__(self, name='microphone', obj_name='microphone', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '5172dbe9281a45f48cee8c15bdfa1831', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Paperweight(CustomObjects):
    def __init__(self, name='paperweight', obj_name='paperweight', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '21c10181c122474a8f72a4ad0331f185', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ToiletPaperRoll(CustomObjects):
    def __init__(self, name='toilet_paper_roll', obj_name='toilet_paper_roll', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '7ca7ebcfad964498b49af73be442acf9', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Shoe(CustomObjects):
    def __init__(self, name='shoe', obj_name='shoe', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'd6951661c1e445d8a1d00b7d38d86030', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Clipboard(CustomObjects):
    def __init__(self, name='clipboard', obj_name='clipboard', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'a37158f20ccf436483029e8295629738', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Organizer(CustomObjects):
    def __init__(self, name='organizer', obj_name='organizer', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '4f52ebe6c8cf432986e47d8de83386ce', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Wineglass(CustomObjects):
    def __init__(self, name='wineglass', obj_name='wineglass', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'afca0b1933b84b1dbebd1244f25e72fc', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apple(CustomObjects):
    def __init__(self, name='apple', obj_name='apple', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/ycb_16k_backup', '013_apple_google_16k', 'textured', 'textured.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class HotGlueGun(CustomObjects):
    def __init__(self, name='hot_glue_gun', obj_name='hot_glue_gun', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '5ceec13e87774982afe825e7e74a0ce1', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Spatula(CustomObjects):
    def __init__(self, name='spatula', obj_name='spatula', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/ycb_16k_backup', '033_spatula_google_16k', 'textured', 'textured.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Usb(CustomObjects):
    def __init__(self, name='usb', obj_name='usb', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'b1098389cc3b460bb1f1ce558d4b0764', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Stapler(CustomObjects):
    def __init__(self, name='stapler', obj_name='stapler', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'd2c345be5c084b4e8c92b48f7be69315', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class TapeMeasure(CustomObjects):
    def __init__(self, name='tape_measure', obj_name='tape_measure', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '7138f5e13b2e4e17975c23ba5584164b', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Multimeter(CustomObjects):
    def __init__(self, name='multimeter', obj_name='multimeter', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '79ac684e66b34554ba869bd2fc3c2653', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Glasses(CustomObjects):
    def __init__(self, name='glasses', obj_name='glasses', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'e4f348e98ceb45e3abc77da5b738f1b2', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Highlighter(CustomObjects):
    def __init__(self, name='highlighter', obj_name='highlighter', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'fd058916fb8843c09b4a2e05e3ee894e', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Mixer(CustomObjects):
    def __init__(self, name='mixer', obj_name='mixer', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '990d0cb9499540fda49b1ff36be9ba26', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Clock(CustomObjects):
    def __init__(self, name='clock', obj_name='clock', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'fdf932b04e6c4e0fbd6e274563b94536', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Marker(CustomObjects):
    def __init__(self, name='marker', obj_name='marker', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/ycb_16k_backup', '040_large_marker_google_16k', 'textured', 'textured.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Screwdriver(CustomObjects):
    def __init__(self, name='screwdriver', obj_name='screwdriver', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/ycb_16k_backup', '044_flat_screwdriver_google_16k', 'textured', 'textured.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Hat(CustomObjects):
    def __init__(self, name='hat', obj_name='hat', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'dc02f9c1fd01432fadd6e7d15851dc37', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Flashlight(CustomObjects):
    def __init__(self, name='flashlight', obj_name='flashlight', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '123a79642c2646d8b315576828fea84a', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Orange(CustomObjects):
    def __init__(self, name='orange', obj_name='orange', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'a4c9a503f63e440e8d6d924d4e5c36b1', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class HardDrive(CustomObjects):
    def __init__(self, name='hard_drive', obj_name='hard_drive', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'e32b159011544b8aa7cacd812636353e', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BinderClips(CustomObjects):
    def __init__(self, name='binder_clips', obj_name='binder_clips', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '5e55c38b24dc4c239448efa1c6b7f49f', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Scissors(CustomObjects):
    def __init__(self, name='scissors', obj_name='scissors', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '2b1120c4b517409184413ed709c57a66', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Calipers(CustomObjects):
    def __init__(self, name='calipers', obj_name='calipers', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '95c967114b624291bc26bcd705aaa334', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CreditCard(CustomObjects):
    def __init__(self, name='credit_card', obj_name='credit_card', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '5de830b2cccf4fe7a2e6b400abf26ca7', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap(CustomObjects):
    def __init__(self, name='cap', obj_name='cap', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '87e82aa304ce4571b69bdd5182549c72', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class PowerDrill(CustomObjects):
    def __init__(self, name='power_drill', obj_name='power_drill', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/ycb_16k_backup', '035_power_drill_google_16k', 'textured', 'textured.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Speaker(CustomObjects):
    def __init__(self, name='speaker', obj_name='speaker', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '4445f4442a884999960e7e6660459095', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Ladle(CustomObjects):
    def __init__(self, name='ladle', obj_name='ladle', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', 'ce27e3369587476c85a436905f1e5c00', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None



@register_object
class Plate(CustomObjects):
    def __init__(self, name='plate', obj_name='plate', goal_quat=[0, 0, 0, 1]):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                '/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale', '6d5a4e1174dd4f5fa0a9a9076e476b91', 'material', 'material.xml'
            )),
            name=name,
            obj_name=obj_name,
            goal_quat=goal_quat
        )

        self.rotation = {
            'x': (-np.pi / 2, -np.pi / 2),
            'y': (-np.pi, -np.pi),
            'z': (np.pi, np.pi),
        }
        self.rotation_axis = None



