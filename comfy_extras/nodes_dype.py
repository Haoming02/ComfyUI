from typing_extensions import override

from comfy.model_patcher import ModelPatcher
from comfy_api.latest import ComfyExtension, io


def apply_dype_flux(
    model: ModelPatcher,
    latent_width: int,
    latent_height: int,
    method: str,
    dype_start_sigma: float,
    dype_scale: float,
    dype_exponent: float,
    base_shift: float,
    max_shift: float,
) -> ModelPatcher:

    cache_params = (
        latent_width,
        latent_height,
        method,
        dype_start_sigma,
        dype_scale,
        dype_exponent,
        base_shift,
        max_shift,
    )

    if getattr(model.model, "_dype_params", None) == cache_params:
        return model

    m = model.clone()
    m.model._dype_params = cache_params

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
                io.Latent.Input("latent"),
                io.Combo.Input(
                    "method",
                    options=["vision_yarn", "yarn", "ntk", "base"],
                    default="vision_yarn",
                ),
                io.Float.Input(
                    "dype_start_sigma",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                ),
                io.Float.Input(
                    "dype_scale",
                    default=2.0,
                    min=0.0,
                    max=8.0,
                    step=0.5,
                ),
                io.Float.Input(
                    "dype_exponent",
                    default=2.0,
                    min=0.0,
                    max=1000.0,
                    step=0.5,
                ),
                io.Float.Input(
                    "base_shift",
                    default=0.5,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    optional=True,
                ),
                io.Float.Input(
                    "max_shift",
                    default=1.15,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    optional=True,
                ),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model: ModelPatcher,
        latent: dict,
        method: str,
        dype_start_sigma: float,
        dype_scale: float,
        dype_exponent: float,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> io.NodeOutput:

        *bct, h, w = latent["samples"].shape
        m = apply_dype_flux(
            model,
            w // 8,
            h // 8,
            method,
            dype_start_sigma,
            dype_scale,
            dype_exponent,
            base_shift,
            max_shift,
        )

        return io.NodeOutput(m)


class DyPEExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DyPEPatchModelFlux,
        ]


async def comfy_entrypoint():
    return DyPEExtension()
