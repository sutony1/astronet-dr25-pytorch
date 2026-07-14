import torch

from astronet_dr25.model import AstroNet, OFFICIAL_PARAMETER_COUNT


def test_official_shapes_and_parameter_count() -> None:
    model = AstroNet()
    local = torch.zeros(2, 201)
    global_view = torch.zeros(2, 2001)
    assert model.local_column(local.unsqueeze(1)).shape == (2, 32, 46)
    assert model.global_column(global_view.unsqueeze(1)).shape == (2, 256, 59)
    assert model(local, global_view).shape == (2,)
    assert model.parameter_count == OFFICIAL_PARAMETER_COUNT
