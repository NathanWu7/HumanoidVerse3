from .body_model_smpl import BodyModelSMPLH, BodyModelSMPLX
def make_smplx(type="neu_fullpose", **kwargs):
    if type == "smpl":
        bm_kwargs = {
            "model_path": "../smpl_retarget/smpl_model",
            "model_type": "smpl",
            "gender": "neutral",
            "num_betas": 10,
            "create_body_pose": False,
            "create_betas": False,
            "create_global_orient": False,
            "create_transl": False,
        }
        bm_kwargs.update(kwargs)
        # model = SMPL(**bm_kwargs)
        model = BodyModelSMPLH(**bm_kwargs)
    elif type == "smplh":
        bm_kwargs = {
            "model_type": "smplh",
            "gender": kwargs.get("gender", "male"),
            "use_pca": False,
            "flat_hand_mean": False,
        }
        model = BodyModelSMPLH(model_path="../body_model", **bm_kwargs)
    elif type == "supermotion":
        # SuperMotion is trained on BEDLAM dataset, the smplx config is the same except only 10 betas are used
        bm_kwargs = {
            "model_type": "smplx",
            "gender": "neutral",
            "num_pca_comps": 12,
            "flat_hand_mean": False,
        }
        bm_kwargs.update(kwargs)
        model = BodyModelSMPLX(model_path="../smpl_retarget/smpl_model", **bm_kwargs)
    else:
        raise NotImplementedError

    return model