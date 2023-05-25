import carb
import omni
import asyncio
import os
import argparse
from omni.isaac.kit import SimulationApp
async def convert(in_file, out_file, load_materials=False):
    # This import causes conflicts when global
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = not load_materials
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(in_file, out_file, progress_callback, converter_context)
    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success


if __name__ == "__main__":
    kit = SimulationApp()

    from omni.isaac.core.utils.extensions import enable_extension
    enable_extension("omni.kit.asset_converter")
    path_in = "/home/tk/Downloads/cad_model_colored/objfiles/"
    path_out = "/home/tk/Downloads/cad_model_colored/usdfiles/"

    for file  in os.listdir(path_in):
        file_usd = str(file.split('.')[0]) + ".usd"
        print(file_usd)

        status = asyncio.get_event_loop().run_until_complete(
            convert(
                str(path_in + file),
                str(path_out + file_usd), False)
        )

    # cleanup
    kit.close()
