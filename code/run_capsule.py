""" top level run script """
# GENERAL IMPORTS
import os

# this is needed to limit the number of scipy threads
# and let spikeinterface handle parallelization
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import pickle
import numpy as np
from pathlib import Path

import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.generation as sgen
import spikeinterface.widgets as sw


data_folder = Path("../data")
results_folder = Path("../results")


# Define argument parser
parser = argparse.ArgumentParser(description="Generate hybrid datasets")

complexity_group = parser.add_mutually_exclusive_group()
complexity_help = "Complexity of the hybrid cases. Can be a string with a single case, or comma-separated (e.g., 'easy,hard')"
complexity_group.add_argument("--complexity", help=complexity_help)
complexity_group.add_argument("static_complexity", nargs="?", default="medium", help=complexity_help)

num_units_group = parser.add_mutually_exclusive_group()
num_units_help = "Number of hybrid units for each case"
num_units_group.add_argument("--num-units", help=num_units_help)
num_units_group.add_argument("static_num_units", nargs="?", help=num_units_help)

num_cases_group = parser.add_mutually_exclusive_group()
num_cases_help = "Number of cases for each complexity"
num_cases_group.add_argument("--num-cases", help=num_cases_help)
num_cases_group.add_argument("static_num_cases", nargs="?", help=num_cases_help)

correct_motion_group = parser.add_mutually_exclusive_group()
correct_motion_help = "Whether to concatenate recordings (segments) or not. Default: False"
correct_motion_group.add_argument("--skip-correct-motion", action="store_true", help=correct_motion_help)
correct_motion_group.add_argument("static_correct_motion", nargs="?", help=correct_motion_help)

debug_group = parser.add_mutually_exclusive_group()
debug_help = "Whether to run in DEBUG mode"
debug_group.add_argument("--debug", action="store_true", help=debug_help)
debug_group.add_argument("static_debug", nargs="?", default="false", help=debug_help)

if __name__ == "__main__":
    args = parser.parse_args()

    recordings_output_folder = results_folder / "recordings"
    sortings_output_folder = results_folder / "sortings"
    figure_output_folder = results_folder / "figures"
    templates_figures_folder = figure_output_folder / "templates"
    
    recordings_output_folder.mkdir(exist_ok=True)
    sortings_output_folder.mkdir(exist_ok=True)
    figure_output_folder.mkdir(exist_ok=True)
    templates_figures_folder.mkdir(exist_ok=True)

    with open("params.json", "r") as f:
        params = json.load(f)

    COMPLEXITY = args.complexity or args.static_complexity
    COMPLEXITY = COMPLEXITY.split(",")
    if isinstance(COMPLEXITY, str):
        COMPLEXITY = [COMPLEXITY]

    NUM_UNITS = int(args.num_units or args.static_num_units or params["num_units_per_case"])
    NUM_CASES = int(args.num_cases or args.static_num_cases or params["num_cases"])
    DEBUG = args.debug or args.static_debug == "true"


    if args.skip_correct_motion:
        CORRECT_MOTION = False
    elif args.static_correct_motion is not None:
        CORRECT_MOTION = True if args.static_correct_motion == "true" else False
    else:
        CORRECT_MOTION = params["correct_motion"]

    print(f"COMPLEXITY: {COMPLEXITY}")
    print(f"NUM_UNITS: {NUM_UNITS}")
    print(f"NUM_CASES: {NUM_CASES}")
    print(f"CORRECT_MOTION: {CORRECT_MOTION}")

    # input json files
    # find raw data
    job_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
    job_dicts = []
    for job_json_file in job_json_files:
        with open(job_json_file) as f:
            job_dict = json.load(f)
        job_dicts.append(job_dict)
    print(f"Found {len(job_dicts)} JSON job files")
    if DEBUG:
        job_dicts = job_dicts[:2]
        print(f"DEBUG MODE: restricted to {len(job_dicts)} JSON job files")

    templates_info = sgen.fetch_templates_database_info()

    # TODO: check this at database creation!
    templates_info = templates_info.query("amplitude_uv > 0")
    
    # for each JSON file, we now create hybrid recordings
    for job_dict in job_dicts:
        recording_name = job_dict["recording_name"]
        print(f"Creating hybrid recordings for {recording_name}")
        recording = si.load_extractor(job_dict["recording_dict"], base_folder=data_folder)
        print(f"\t{recording}")

        # preprocess
        recording_preproc = spre.highpass_filter(recording)
        recording_preproc = spre.common_reference(recording_preproc)

        motion = None
        if CORRECT_MOTION:
            print("Estimating motion")
            motion_figures_folder = figure_output_folder / "motion"
            motion_figures_folder.mkdir(exist_ok=True)
            
            _, motion_info = spre.correct_motion(
                recording_preproc, preset="dredge_fast", n_jobs=-1, progress_bar=True, output_motion_info=True
            )
            motion = motion_info["motion"]
            w = sw.plot_motion_info(
                motion_info,
                recording_preproc,
                color_amplitude=True,
                scatter_decimate=10,
                amplitude_cmap="Greys_r"
            )
            w.figure.savefig(motion_figures_folder / f"recording_name.png", dpi=300)

        for complexity in COMPLEXITY:
            print(f"\tGenerating complexity: {complexity}")
            min_amplitude, max_amplitude = params["amplitudes"][complexity]
            for case in range(NUM_CASES):
                print(f"\t\tGenerating case: {case}")
                case_name = f"{recording_name}_{complexity}_{case}"

                # sample templates
                print(f"\t\t\tSelecting and fetching templates")
                templates_selected_indices = np.random.choice(templates_info.index, size=NUM_UNITS, replace=False)
                print(f"\t\t\tSelected indices: {list(templates_selected_indices)}")
                templates_selected_info = templates_info.loc[templates_selected_indices]

                # fetch templates
                templates_selected = sgen.query_templates_from_database(templates_selected_info)

                # scale templates
                print(f"\t\t\tScaling templates between {min_amplitude} and {max_amplitude}")
                templates_scaled = sgen.scale_template_to_range(
                    templates=templates_selected,
                    min_amplitude=min_amplitude,
                    max_amplitude=max_amplitude
                )

                print(f"\t\t\tConstructing hybrid recording")
                recording_hybrid, sorting_hybrid = sgen.generate_hybrid_recording(
                    recording=recording_preproc,
                    templates=templates_scaled,
                    motion=motion,
                    seed=None,
                )
                print(recording_hybrid)

                # rename hybrid units with selected indices for provenance
                sorting_hybrid = sorting_hybrid.rename_units(templates_selected_indices)

                # we construct here a pkl version of the job json because 
                # it needs to be compatible with the preprocessing capsule
                recording_dict = recording_hybrid.to_dict(
                    include_annotations=True,
                    include_properties=True,
                    relative_to=data_folder,
                    recursive=True,
                )
                dump_dict = {
                    "session_name": job_dict["session_name"],
                    "recording_name": case_name,
                    "recording_dict": recording_dict
                }
                file_path = recordings_output_folder / f"job_{case_name}.pkl"
                file_path.write_bytes(pickle.dumps(dump_dict))

                sorting_hybrid.dump_to_pickle(
                    sortings_output_folder / f"{case_name}.pkl",
                    relative_to=data_folder
                )

                # generate some plots!
                templates_obj = si.Templates(
                    recording_hybrid.templates,
                    channel_ids=templates_selected.channel_ids,
                    unit_ids=sorting_hybrid.unit_ids,
                    probe=recording.get_probe(), 
                    sampling_frequency=templates_selected.sampling_frequency,
                    nbefore=templates_selected.nbefore
                )
                sparsity = si.compute_sparsity(templates_obj)

                figsize = (7, 3*NUM_UNITS)
                w = sw.plot_unit_templates(
                    templates_obj,
                    sparsity=sparsity,
                    figsize=figsize,
                    ncols=2,
                )
                w.figure.savefig(templates_figures_folder / f"{case_name}.pdf")
