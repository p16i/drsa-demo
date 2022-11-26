import torch


class LayerInspectionContext:
    # the attribute is set to the explainer where we are inspecting.
    LAYER_INSPECTED = "__inspection_context__layer_with_inspection"
    # the attribute is set to the module (or layer) we are inspecting.
    IS_INSPECTED = "__inspection_context__with_inspection"
    # the attribute is set to the module (or layer) we are inspecting and
    # it is used to store intermediate activation
    IS_INSPECTED_ACTIVATION = "__inspection_context__with_inspection_activation"

    def __init__(self, explainer, module: torch.nn.Module) -> None:
        # todo: add type to explainer
        # remark: when adding this type; there is a circular import issue.
        self.explainer = explainer
        self.module = module

    def __enter__(self):
        setattr(
            self.explainer,
            LayerInspectionContext.LAYER_INSPECTED,
            self.module,
        )
        setattr(self.module, LayerInspectionContext.IS_INSPECTED, True)

    def __exit__(self, *args):
        delattr(self.explainer, LayerInspectionContext.LAYER_INSPECTED)
        delattr(self.module, LayerInspectionContext.IS_INSPECTED)
        delattr(self.module, LayerInspectionContext.IS_INSPECTED_ACTIVATION)

    def get_activation(self) -> torch.Tensor:
        if not hasattr(self.explainer, LayerInspectionContext.LAYER_INSPECTED):
            raise RuntimeError("No intermediate heatmap layer specified")
        act = getattr(self.module, LayerInspectionContext.IS_INSPECTED_ACTIVATION)
        return act
