def get_parameter_groups(model, cfg):
    """
    Assign different weight decays and learning rates to different parameters.
    Returns a parameter group which can be passed to the optimizer.
    """
    weight_decay = cfg.weight_decay
    base_lr = cfg.learning_rate

    backbone_params = []
    embed_params = []
    other_params = []

    # inspired by detectron2
    memo = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Avoid duplicating parameters
        if param in memo:
            continue
        memo.add(param)

        if name.startswith('module'):
            name = name[7:]

        inserted = False
        other_params.append(param)

    parameter_groups = [
        {
            'params': other_params,
            'lr': base_lr,
            'weight_decay': weight_decay
        },
    ]

    return parameter_groups
