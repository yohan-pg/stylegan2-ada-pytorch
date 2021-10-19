from inversion import *

from measure_reconstruction_quality import run_reconstruction_quality
from measure_image_editing_consistency import run_image_editing_consistency
from measure_interpolation_determinism import run_interpolation_determinism

# todo save everything into files
# todo avoid .cuda() in metrics. clean up interpolation result; want a core datastructure that doesn't store all the preds
# todo run once fully
# todo fix up progress bars
# todo error histogram
# todo make plots as a second pass (fuse plots)
# todo make a single page that shows everything (html?)
# todo double check that it is deterministic for a given seed, & stable for another (low variance)

def run(dataloader, variable_types, methods, num_steps):
    fresh_dir("eval")

    for variable_type in variable_types:
        for method_name, G in methods.items():
            experiment_name = method_name + '_' + variable_type.space_name
            print(experiment_name)

            inverter = Inverter(G, num_steps=num_steps, variable_type=variable_type)
            inverted_dataloader = InvertedDataloader(dataloader, inverter)

            run_reconstruction_quality(experiment_name, inverted_dataloader)
            run_interpolation_determinism(experiment_name, inverted_dataloader)
            run_image_editing_consistency(experiment_name, inverted_dataloader)

if __name__ == "__main__":
    run(
        dataloader = RealDataloader(
            "datasets/afhq2_cat128_test.zip", batch_size=4, num_images=8
        ),
        variable_types = [ZVariable, ZPlusVariable, WVariable, WPlusVariable],
        num_steps = 2,
        methods = {
            "AdaConv": open_generator("pretrained/adaconv-gamma-20-003800.pkl"),
            "AdaConvSlow": open_generator("pretrained/adaconv-slowdown-all.pkl"),
        }
    )

    
