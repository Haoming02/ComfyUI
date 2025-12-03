from typing_extensions import override

from comfy.model_patcher import ModelPatcher
from comfy_api.latest import ComfyExtension, io


def apply_dype_flux(model: ModelPatcher, method: str) -> ModelPatcher:
    if getattr(model.model, "_dype", None) == method:
        return model

    m = model.clone()
    m.model._dype = method

    return m


class DyPEPatchModelFlux(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DyPEPatchModelFlux",
            display_name="DyPE Patch Model (Flux)",
            category="_for_testing",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "method",
                    options=["yarn", "ntk", "base"],
                    default="yarn",
                ),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model: ModelPatcher, method: str) -> io.NodeOutput:
        m = apply_dype_flux(model, method)
        return io.NodeOutput(m)


class DyPEExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DyPEPatchModelFlux,
        ]


async def comfy_entrypoint():
    return DyPEExtension()
