from typing_extensions import override

from comfy.model_patcher import ModelPatcher
from comfy_api.latest import ComfyExtension, io


class DyPEPatchModelFlux(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DyPEPatchModelFlux",
            display_name="DyPE Patch Model (Flux)",
            category="_for_testing",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("latent"),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model: ModelPatcher, latent: dict) -> io.NodeOutput:
        m = model.clone()
        return io.NodeOutput(m)


class DyPEExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DyPEPatchModelFlux,
        ]


async def comfy_entrypoint():
    return DyPEExtension()
