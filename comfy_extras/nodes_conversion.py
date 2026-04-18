from enum import Enum
from typing import Any

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io
from server import PromptServer


class ConvertMode(Enum):
    Int = "int"
    Int_Round = "int (round)"
    Float = "float"
    String = "string"
    Boolean = "bool"
    Boolean_String = "bool (string)"


TRUTHFUL: set[str] = {"true", "yes", "t", "y", "1"}


class ConvertAny(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConvertAny",
            display_name="Type Conversion",
            description="""
Convert one primitive type to another
- int: Call the Python native int() function ; return the integer with decimal places removed
- int (round): Convert the value to float first, then call the Python native round() function ; return a rounded integer
- float: Call the Python native float() function ; return a decimal value
- string: Call the Python native str() function ; return the value as string or its string representation
- bool: Call the Python native bool() function ; return whether the value is considered true or false
- bool (string): Return whether the string is one of ["true", "yes", "t", "y", "1"] (case insensitive)
            """,
            category="utils",
            inputs=[
                io.AnyType.Input(
                    id="input_value",
                    display_name="Input Value",
                    tooltip="The value to convert from",
                ),
                io.Combo.Input(
                    id="output_type",
                    options=ConvertMode,
                    display_name="Output Type",
                    tooltip="The type to convert to",
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    id="output_value",
                    display_name="Output Value",
                    tooltip="The value in the converted type if successful ; otherwise the original value",
                )
            ],
            hidden=[io.Hidden.unique_id],
            search_aliases=[
                "convert",
                "to",
                "int",
                "float",
                "string",
                "bool",
            ],
        )

    @classmethod
    def execute(cls, input_value: Any, output_type: str) -> io.NodeOutput:
        try:
            match output_type:
                case ConvertMode.Int.value:
                    output_value = int(float(input_value))
                case ConvertMode.Int_Round.value:
                    output_value = round(float(input_value))
                case ConvertMode.Float.value:
                    output_value = float(input_value)
                case ConvertMode.String.value:
                    output_value = str(input_value)
                case ConvertMode.Boolean.value:
                    output_value = bool(input_value)
                case ConvertMode.Boolean_String.value:
                    if not isinstance(input_value, str):
                        raise TypeError("Input is not a string")
                    output_value = input_value.strip().lower() in TRUTHFUL
        except (TypeError, ValueError) as e:
            if cls.hidden.unique_id:
                PromptServer.instance.send_progress_text(
                    f'Failed to convert "{input_value}" ({type(input_value).__name__}) to "{output_type}"\n{e}',
                    cls.hidden.unique_id,
                )
            return io.NodeOutput(input_value)
        else:
            if cls.hidden.unique_id:
                PromptServer.instance.send_progress_text(
                    f'Successfully converted "{input_value}" ({type(input_value).__name__}) to "{output_value}" ({type(output_value).__name__})',
                    cls.hidden.unique_id,
                )
            return io.NodeOutput(output_value)


class ConversionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ConvertAny]


async def comfy_entrypoint():
    return ConversionExtension()
