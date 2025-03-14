from .base_predicates import *


VALIDATE_PREDICATE_FN_DICT = {
    "true": TruePredicateFn(),
    "false": FalsePredicateFn(),
    "in": In(),
    # "incontact": InContactPredicateFn(),
    "on": On(),
    "up": Up(),
    # "stack":     Stack(),
    # "temporal":  TemporalPredicate(),
    "printjointstate": PrintJointState(),
    "open": Open(),
    "close": Close(),
    "turnon": TurnOn(),
    "turnoff": TurnOff(),
    
    # TODO: wenyao add the relationship predicate function
    # "position" : Position(),
    "left": Left(),
    "right": Right(),
    "front": Front(),
    "behind": Behind(),
    "between": Between(),
    "center": Center(),
    "quat": Quat(),
    "top": Behind(),
    "betweenquat": BetweenQuat,
    "leftquat": LeftQuat,
    "rightquat": RightQuat,
    "frontquat": FrontQuat,
    "behindquat": BehindQuat,
    "topquat": TopQuat,
    "centerquat": CenterQuat,
    
}



def update_predicate_fn_dict(fn_key, fn_name):
    VALIDATE_PREDICATE_FN_DICT.update({fn_key: eval(fn_name)()})


def eval_predicate_fn(predicate_fn_name, *args):
    assert predicate_fn_name in VALIDATE_PREDICATE_FN_DICT
    return VALIDATE_PREDICATE_FN_DICT[predicate_fn_name](*args)


def get_predicate_fn_dict():
    return VALIDATE_PREDICATE_FN_DICT


def get_predicate_fn(predicate_fn_name):
    return VALIDATE_PREDICATE_FN_DICT[predicate_fn_name.lower()]
