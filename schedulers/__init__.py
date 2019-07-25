import logging

from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR
from schedulers.schedulers import WarmUpLR, ConstantLR, PolynomialLR

key2scheduler = {
    "constant_lr": ConstantLR,
    "poly_lr": PolynomialLR,
    "step": StepLR,
    "multi_step": MultiStepLR,
    "cosine_annealing": CosineAnnealingLR,
    "exp_lr": ExponentialLR,
}

def get_scheduler(optimizer, params):
    if params is None:
        print("Using constant LR Scheduling")
        return ConstantLR(optimizer)

    scheduler_dict = params.copy()
    s_type = scheduler_dict["name"]
    scheduler_dict.pop("name")

    print("Using {} scheduler with {} params".format(s_type, scheduler_dict))

    warmup_dict = {}
    if "warmup_iters" in scheduler_dict:
        # This can be done in a more pythonic way...
        warmup_dict["warmup_iters"] = scheduler_dict.get("warmup_iters", 100)
        warmup_dict["mode"] = scheduler_dict.get("warmup_mode", "linear")
        warmup_dict["gamma"] = scheduler_dict.get("warmup_factor", 0.2)

        print(
            "Using Warmup with {} iters {} gamma and {} mode".format(
                warmup_dict["warmup_iters"], warmup_dict["gamma"], warmup_dict["mode"]
            )
        )

        scheduler_dict.pop("warmup_iters", None)
        scheduler_dict.pop("warmup_mode", None)
        scheduler_dict.pop("warmup_factor", None)

        base_scheduler = key2scheduler[s_type](optimizer, **scheduler_dict)
        return WarmUpLR(optimizer, base_scheduler, **warmup_dict)

    return key2scheduler[s_type](optimizer, **scheduler_dict)
