import optuna
from typing import Any, Dict, List, Tuple, Sequence, Optional, Union

def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    # epochs = trial.suggest_int(name="epochs", low=10, high=30, step=5)
    batch_size = trial.suggest_categorical("batch_size", [16,32,64])
    lr = trial.suggest_float(name="lr", low=1e-5, high = 1e-3)
    return {
        # "EPOCHS": epochs,
        "BATCH_SIZE": batch_size,
        "LEARNING_RATE": lr,
    }

def search_model(trial: optuna.trial.Trial) -> List[Any]:
    """Search model structure from user-specified search space."""
    model = []
    n_stride = 0
    MAX_NUM_STRIDE = 5
    UPPER_STRIDE = 2  # 5(224 example): 224, 112, 56, 28, 14, 7

    # Module 1
    m1 = trial.suggest_categorical("m1", ["Conv", "DWConv"])
    m1_args = []
    m1_repeat = trial.suggest_int("m1/repeat", 1, 3)
    m1_out_channel = trial.suggest_int("m1/out_channels", low=16, high=64, step=16)
    m1_stride = trial.suggest_int("m1/stride", low=1, high=UPPER_STRIDE)
    if m1_stride == 2:
        n_stride += 1
    m1_activation = trial.suggest_categorical("m1/activation", ["ReLU", "Hardswish"])
    if m1 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m1_args = [m1_out_channel, 3, m1_stride, None, 1, m1_activation]
    elif m1 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m1_args = [m1_out_channel, 3, m1_stride, None, m1_activation]
    model.append([m1_repeat, m1, m1_args])

    # Module 2
    m2 = trial.suggest_categorical(
        "m2", ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "Pass"]
    )
    m2_args = []
    m2_repeat = trial.suggest_int("m2/repeat", 1, 5)
    m2_out_channel = trial.suggest_int("m2/out_channels", low=16, high=128, step=16)
    m2_stride = trial.suggest_int("m2/stride", low=1, high=UPPER_STRIDE)
    # force stride m2
    if n_stride == 0:
        m2_stride = 2
    if m2 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m2_kernel = trial.suggest_int("m2/kernel_size", low=1, high=5, step=2)
        m2_activation = trial.suggest_categorical(
            "m2/activation", ["ReLU", "Hardswish"]
        )
        m2_args = [m2_out_channel, m2_kernel, m2_stride, None, 1, m2_activation]
    elif m2 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m2_kernel = trial.suggest_int("m2/kernel_size", low=1, high=5, step=2)
        m2_activation = trial.suggest_categorical(
            "m2/activation", ["ReLU", "Hardswish"]
        )
        m2_args = [m2_out_channel, m2_kernel, m2_stride, None, m2_activation]
    elif m2 == "InvertedResidualv2":
        m2_c = trial.suggest_int("m2/v2_c", low=16, high=32, step=16)
        m2_t = trial.suggest_int("m2/v2_t", low=1, high=4)
        m2_args = [m2_c, m2_t, m2_stride]
    elif m2 == "InvertedResidualv3":
        m2_kernel = trial.suggest_int("m2/kernel_size", low=3, high=5, step=2)
        m2_t = round(trial.suggest_float("m2/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m2_c = trial.suggest_int("m2/v3_c", low=16, high=40, step=8)
        m2_se = trial.suggest_categorical("m2/v3_se", [0, 1])
        m2_hs = trial.suggest_categorical("m2/v3_hs", [0, 1])
        # k t c SE HS s
        m2_args = [m2_kernel, m2_t, m2_c, m2_se, m2_hs, m2_stride]
    if not m2 == "Pass":
        if m2_stride == 2:
            n_stride += 1
            if n_stride >= MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m2_repeat, m2, m2_args])

    # Module 3
    m3 = trial.suggest_categorical(
        "m3", ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "Pass"]
    )
    m3_args = []
    m3_repeat = trial.suggest_int("m3/repeat", 1, 5)
    m3_stride = trial.suggest_int("m3/stride", low=1, high=UPPER_STRIDE)
    if m3 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m3_out_channel = trial.suggest_int("m3/out_channels", low=16, high=128, step=16)
        m3_kernel = trial.suggest_int("m3/kernel_size", low=1, high=5, step=2)
        m3_activation = trial.suggest_categorical(
            "m3/activation", ["ReLU", "Hardswish"]
        )
        m3_args = [m3_out_channel, m3_kernel, m3_stride, None, 1, m3_activation]
    elif m3 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m3_out_channel = trial.suggest_int("m3/out_channels", low=16, high=128, step=16)
        m3_kernel = trial.suggest_int("m3/kernel_size", low=1, high=5, step=2)
        m3_activation = trial.suggest_categorical(
            "m3/activation", ["ReLU", "Hardswish"]
        )
        m3_args = [m3_out_channel, m3_kernel, m3_stride, None, m3_activation]
    elif m3 == "InvertedResidualv2":
        m3_c = trial.suggest_int("m3/v2_c", low=8, high=32, step=8)
        m3_t = trial.suggest_int("m3/v2_t", low=1, high=8)
        m3_args = [m3_c, m3_t, m3_stride]
    elif m3 == "InvertedResidualv3":
        m3_kernel = trial.suggest_int("m3/kernel_size", low=3, high=5, step=2)
        m3_t = round(trial.suggest_float("m3/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m3_c = trial.suggest_int("m3/v3_c", low=8, high=40, step=8)
        m3_se = trial.suggest_categorical("m3/v3_se", [0, 1])
        m3_hs = trial.suggest_categorical("m3/v3_hs", [0, 1])
        m3_args = [m3_kernel, m3_t, m3_c, m3_se, m3_hs, m3_stride]
    if not m3 == "Pass":
        if m3_stride == 2:
            n_stride += 1
            if n_stride >= MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m3_repeat, m3, m3_args])

    # Module 4
    m4 = trial.suggest_categorical(
        "m4", ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "Pass"]
    )
    m4_args = []
    m4_repeat = trial.suggest_int("m4/repeat", 1, 5)
    m4_stride = trial.suggest_int("m4/stride", low=1, high=UPPER_STRIDE)
    # force stride m4
    if n_stride == 1:
        m4_stride = 2
    if m4 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m4_out_channel = trial.suggest_int("m4/out_channels", low=16, high=256, step=16)
        m4_kernel = trial.suggest_int("m4/kernel_size", low=1, high=5, step=2)
        m4_activation = trial.suggest_categorical(
            "m4/activation", ["ReLU", "Hardswish"]
        )
        m4_args = [m4_out_channel, m4_kernel, m4_stride, None, 1, m4_activation]
    elif m4 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m4_out_channel = trial.suggest_int("m4/out_channels", low=16, high=256, step=16)
        m4_kernel = trial.suggest_int("m4/kernel_size", low=1, high=5, step=2)
        m4_activation = trial.suggest_categorical(
            "m4/activation", ["ReLU", "Hardswish"]
        )
        m4_args = [m4_out_channel, m4_kernel, m4_stride, None, m4_activation]
    elif m4 == "InvertedResidualv2":
        m4_c = trial.suggest_int("m4/v2_c", low=8, high=64, step=8)
        m4_t = trial.suggest_int("m4/v2_t", low=1, high=8)
        m4_args = [m4_c, m4_t, m4_stride]
    elif m4 == "InvertedResidualv3":
        m4_kernel = trial.suggest_int("m4/kernel_size", low=3, high=5, step=2)
        m4_t = round(trial.suggest_float("m4/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m4_c = trial.suggest_int("m4/v3_c", low=8, high=80, step=8)
        m4_se = trial.suggest_categorical("m4/v3_se", [0, 1])
        m4_hs = trial.suggest_categorical("m4/v3_hs", [0, 1])
        m4_args = [m4_kernel, m4_t, m4_c, m4_se, m4_hs, m4_stride]
    if not m4 == "Pass":
        if m4_stride == 2:
            n_stride += 1
            if n_stride >= MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m4_repeat, m4, m4_args])

    # Module 5
    m5 = trial.suggest_categorical(
        "m5", ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "Pass"]
    )
    m5_args = []
    m5_repeat = trial.suggest_int("m5/repeat", 1, 5)
    m5_stride = 1
    if m5 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m5_out_channel = trial.suggest_int("m5/out_channels", low=16, high=256, step=16)
        m5_kernel = trial.suggest_int("m5/kernel_size", low=1, high=5, step=2)
        m5_activation = trial.suggest_categorical(
            "m5/activation", ["ReLU", "Hardswish"]
        )
        m5_stride = trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
        m5_args = [m5_out_channel, m5_kernel, m5_stride, None, 1, m5_activation]
    elif m5 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m5_out_channel = trial.suggest_int("m5/out_channels", low=16, high=256, step=16)
        m5_kernel = trial.suggest_int("m5/kernel_size", low=1, high=5, step=2)
        m5_activation = trial.suggest_categorical(
            "m5/activation", ["ReLU", "Hardswish"]
        )
        m5_stride = trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
        m5_args = [m5_out_channel, m5_kernel, m5_stride, None, m5_activation]
    elif m5 == "InvertedResidualv2":
        m5_c = trial.suggest_int("m5/v2_c", low=16, high=128, step=16)
        m5_t = trial.suggest_int("m5/v2_t", low=1, high=8)
        m5_stride = trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
        m5_args = [m5_c, m5_t, m5_stride]
    elif m5 == "InvertedResidualv3":
        m5_kernel = trial.suggest_int("m5/kernel_size", low=3, high=5, step=2)
        m5_t = round(trial.suggest_float("m5/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m5_c = trial.suggest_int("m5/v3_c", low=16, high=80, step=16)
        m5_se = trial.suggest_categorical("m5/v3_se", [0, 1])
        m5_hs = trial.suggest_categorical("m5/v3_hs", [0, 1])
        m5_stride = trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
        m5_args = [m5_kernel, m5_t, m5_c, m5_se, m5_hs, m5_stride]
    if not m5 == "Pass":
        if m5_stride == 2:
            n_stride += 1
            if n_stride >= MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m5_repeat, m5, m5_args])

    # Module 6
    m6 = trial.suggest_categorical(
        "m6", ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "Pass"]
    )
    m6_args = []
    m6_repeat = trial.suggest_int("m6/repeat", 1, 5)
    m6_stride = trial.suggest_int("m6/stride", low=1, high=UPPER_STRIDE)
    # force stride m6
    if n_stride == 2:
        m4_stride = 2
    if m6 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m6_out_channel = trial.suggest_int("m6/out_channels", low=16, high=512, step=16)
        m6_kernel = trial.suggest_int("m6/kernel_size", low=1, high=5, step=2)
        m6_activation = trial.suggest_categorical(
            "m6/activation", ["ReLU", "Hardswish"]
        )
        m6_args = [m6_out_channel, m6_kernel, m6_stride, None, 1, m6_activation]
    elif m6 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m6_out_channel = trial.suggest_int("m6/out_channels", low=16, high=512, step=16)
        m6_kernel = trial.suggest_int("m6/kernel_size", low=1, high=5, step=2)
        m6_activation = trial.suggest_categorical(
            "m6/activation", ["ReLU", "Hardswish"]
        )
        m6_args = [m6_out_channel, m6_kernel, m6_stride, None, m6_activation]
    elif m6 == "InvertedResidualv2":
        m6_c = trial.suggest_int("m6/v2_c", low=16, high=128, step=16)
        m6_t = trial.suggest_int("m6/v2_t", low=1, high=8)
        m6_args = [m6_c, m6_t, m6_stride]
    elif m6 == "InvertedResidualv3":
        m6_kernel = trial.suggest_int("m6/kernel_size", low=3, high=5, step=2)
        m6_t = round(trial.suggest_float("m6/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m6_c = trial.suggest_int("m6/v3_c", low=16, high=160, step=16)
        m6_se = trial.suggest_categorical("m6/v3_se", [0, 1])
        m6_hs = trial.suggest_categorical("m6/v3_hs", [0, 1])
        m6_args = [m6_kernel, m6_t, m6_c, m6_se, m6_hs, m6_stride]
    if not m6 == "Pass":
        if m6_stride == 2:
            n_stride += 1
            if n_stride >= MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m6_repeat, m6, m6_args])

    # Module 7
    m7 = trial.suggest_categorical(
        "m7", ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "Pass"]
    )
    m7_args = []
    m7_repeat = trial.suggest_int("m7/repeat", 1, 5)
    m7_stride = trial.suggest_int("m7/stride", low=1, high=UPPER_STRIDE)
    if m7 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m7_out_channel = trial.suggest_int(
            "m7/out_channels", low=128, high=1024, step=128
        )
        m7_kernel = trial.suggest_int("m7/kernel_size", low=1, high=5, step=2)
        m7_activation = trial.suggest_categorical(
            "m7/activation", ["ReLU", "Hardswish"]
        )
        m7_args = [m7_out_channel, m7_kernel, m7_stride, None, 1, m7_activation]
    elif m7 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m7_out_channel = trial.suggest_int(
            "m7/out_channels", low=128, high=1024, step=128
        )
        m7_kernel = trial.suggest_int("m7/kernel_size", low=1, high=5, step=2)
        m7_activation = trial.suggest_categorical(
            "m7/activation", ["ReLU", "Hardswish"]
        )
        m7_args = [m7_out_channel, m7_kernel, m7_stride, None, m7_activation]
    elif m7 == "InvertedResidualv2":
        m7_c = trial.suggest_int("m7/v2_c", low=16, high=160, step=16)
        m7_t = trial.suggest_int("m7/v2_t", low=1, high=8)
        m7_args = [m7_c, m7_t, m7_stride]
    elif m7 == "InvertedResidualv3":
        m7_kernel = trial.suggest_int("m7/kernel_size", low=3, high=5, step=2)
        m7_t = round(trial.suggest_float("m7/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m7_c = trial.suggest_int("m7/v3_c", low=8, high=160, step=8)
        m7_se = trial.suggest_categorical("m7/v3_se", [0, 1])
        m7_hs = trial.suggest_categorical("m7/v3_hs", [0, 1])
        m7_args = [m7_kernel, m7_t, m7_c, m7_se, m7_hs, m7_stride]
    if not m7 == "Pass":
        if m7_stride == 2:
            n_stride += 1
            if n_stride >= MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m7_repeat, m7, m7_args])

    # last layer
    last_dim = trial.suggest_int("last_dim", low=128, high=1024, step=128)
    # We can setup fixed structure as well
    model.append([1, "Conv", [last_dim, 1, 1]])
    model.append([1, "GlobalAvgPool", []])
    model.append([1, "FixedConv", [6, 1, 1, None, 1, None]])

    module_info = {}
    module_info["m1"] = {"type": m1, "repeat": m1_repeat, "stride": m1_stride}
    module_info["m2"] = {"type": m2, "repeat": m2_repeat, "stride": m2_stride}
    module_info["m3"] = {"type": m3, "repeat": m3_repeat, "stride": m3_stride}
    module_info["m4"] = {"type": m4, "repeat": m4_repeat, "stride": m4_stride}
    module_info["m5"] = {"type": m5, "repeat": m5_repeat, "stride": m5_stride}
    module_info["m6"] = {"type": m6, "repeat": m6_repeat, "stride": m6_stride}
    module_info["m7"] = {"type": m7, "repeat": m7_repeat, "stride": m7_stride}

    return model, module_info