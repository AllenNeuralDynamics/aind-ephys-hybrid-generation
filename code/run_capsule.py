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

margin_from_border = 100


# Define argument parser
parser = argparse.ArgumentParser(description="Generate hybrid datasets")


min_amplitude_group = parser.add_mutually_exclusive_group()
min_amplitude_help = "Minimum amplitude to scale injected templates"
min_amplitude_group.add_argument("--min-amp", help=min_amplitude_help)
min_amplitude_group.add_argument("static_min_amp", nargs="?", default="50", help=min_amplitude_help)

max_amplitude_group = parser.add_mutually_exclusive_group()
max_amplitude_help = "Maximum amplitude to scale injected templates"
max_amplitude_group.add_argument("--max-amp", help=max_amplitude_help)
max_amplitude_group.add_argument("static_max_amp", nargs="?", default="200", help=max_amplitude_help)

min_depth_percentile_group = parser.add_mutually_exclusive_group()
min_depth_percentile_help = "Percentile of depths used as minimum depth"
min_depth_percentile_group.add_argument("--min-depth-percentile", help=min_depth_percentile_help)
min_depth_percentile_group.add_argument("static_min_depth_percentile", nargs="?", default="5", help=min_depth_percentile_help)

max_depth_percentile_group = parser.add_mutually_exclusive_group()
max_depth_percentile_help = "Percentile of depths used as maximum depth"
max_depth_percentile_group.add_argument("--max-depth-percentile", help=max_depth_percentile_help)
max_depth_percentile_group.add_argument("static_max_depth_percentile", nargs="?", default="95", help=max_depth_percentile_help)

num_units_group = parser.add_mutually_exclusive_group()
num_units_help = "Number of hybrid units for each case"
num_units_group.add_argument("--num-units", help=num_units_help)
num_units_group.add_argument("static_num_units", nargs="?", default="10", help=num_units_help)

num_cases_group = parser.add_mutually_exclusive_group()
num_cases_help = "Number of cases for each recording"
num_cases_group.add_argument("--num-cases", help=num_cases_help)
num_cases_group.add_argument("static_num_cases", nargs="?", default="5", help=num_cases_help)

correct_motion_group = parser.add_mutually_exclusive_group()
correct_motion_help = "Whether to skip motion correction."
correct_motion_group.add_argument("--skip-correct-motion", action="store_true", help=correct_motion_help)
correct_motion_group.add_argument("static_correct_motion", nargs="?", help=correct_motion_help)


if __name__ == "__main__":
    args = parser.parse_args()

    MIN_AMP = float(args.min_amp or args.static_min_amp)
    MAX_AMP = float(args.max_amp or args.static_max_amp)
    MIN_DEPTH_PERC = float(args.min_depth_percentile or args.static_min_depth_percentile)
    MAX_DEPTH_PERC = float(args.max_depth_percentile or args.static_max_depth_percentile)
    NUM_UNITS = int(args.num_units or args.static_num_units)
    NUM_CASES = int(args.num_cases or args.static_num_cases)
    
    if args.static_correct_motion is not None:
        CORRECT_MOTION = True if args.static_correct_motion == "true" else False
    elif args.skip_correct_motion:
        CORRECT_MOTION = False
    else:
        CORRECT_MOTION = True

    print(f"MIN_AMP: {MIN_AMP}")
    print(f"MAX_AMP: {MAX_AMP}")
    print(f"MIN_DEPTH_PERCENTILE: {MIN_DEPTH_PERC}")
    print(f"MAX_DEPTH_PERCENTILE: {MAX_DEPTH_PERC}")
    print(f"NUM_UNITS: {NUM_UNITS}")
    print(f"NUM_CASES: {NUM_CASES}")
    print(f"CORRECT_MOTION: {CORRECT_MOTION}")

    # this folder is used for parallelization
    recordings_folder = results_folder / "recordings"
    recordings_folder.mkdir(exist_ok=True)
    # this folder is used to collect the output
    flattened_folder = results_folder / "flattened"
    flattened_folder.mkdir(exist_ok=True)

    # input json files
    # find raw data
    job_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
    job_dicts = []
    for job_json_file in job_json_files:
        with open(job_json_file) as f:
            job_dict = json.load(f)
        job_dicts.append(job_dict)
    print(f"Found {len(job_dicts)} JSON job files")

    si.set_global_job_kwargs(n_jobs=-1, progress_bar=False)

    templates_info = sgen.fetch_templates_database_info()

    # TODO: check this at database creation!
    templates_info = templates_info.query("amplitude_uv > 0")
    
    # for each JSON file, we now create hybrid recordings
    for job_dict in job_dicts:
        recording_name = job_dict["recording_name"]
        print(f"\n\nCreating hybrid recordings for {recording_name}")
        recording = si.load_extractor(job_dict["recording_dict"], base_folder=data_folder)

        probes_info = recording.get_annotation("probes_info")
        model_name = None
        if probes_info is not None:
            model_name = probes_info[0].get("model_name")
            name = probes_info[0].get("name")
            if model_name is None:
                if name is not None and "Neuropixels" in name:
                    model_name = name

        if model_name is None:
            print("\tCould not load probes info. Selecting Neuropixels Ultra templates")
            templates_info = templates_info.query("probe == 'Neuropixels Ultra'")
            relocate_templates = True
            select_by_depth = False
        elif "1.0" in model_name:
            print("\tProbe model: {model_name}. Selecting Neuropixels 1.0 templates")
            templates_info = templates_info.query("probe == 'Neuropixels 1.0'")
            relocate_templates = False
            select_by_depth = True
        else:
            print("\tProbe model: {model_name}. Selecting Neuropixels Ultra templates")
            templates_info = templates_info.query("probe == 'Neuropixels Ultra'")
            relocate_templates = True
            select_by_depth = False

        print(f"\tSelected {len(templates_info)} templates from database")

        # skip times if non-monotonically increasing
        if job_dict["skip_times"]:
            recording.reset_times()
        print(f"\tRecording: {recording}")

        # preprocess
        recording_preproc = spre.highpass_filter(recording)
        recording_preproc = spre.common_reference(recording_preproc)

        motion = None
        if CORRECT_MOTION:
            print("\tEstimating motion")
            motion_folder = flattened_folder / f"motion_{recording_name}"
            motion_figure_file = flattened_folder / f"fig-motion_{recording_name}.png"

            recording_preproc_f = spre.astype(recording_preproc, "float")
            _, motion_info = spre.correct_motion(
                recording_preproc_f, preset="dredge_fast",output_motion_info=True, folder=motion_folder
            )
            motion = motion_info["motion"]
            w = sw.plot_motion_info(
                motion_info,
                recording_preproc,
                color_amplitude=True,
                scatter_decimate=10,
                amplitude_cmap="Greys_r"
            )
            w.figure.savefig(motion_figure_file, dpi=300)

        print(f"\tGenerating hybrid recordings")
        min_amplitude = MIN_AMP
        max_amplitude = MAX_AMP
        min_depth_percentile = MIN_DEPTH_PERC
        max_depth_percentile = MAX_DEPTH_PERC
        for case in range(NUM_CASES):
            print(f"\t\tGenerating case: {case}")
            case_name = f"{recording_name}_{case}"

            # sample templates
            print(f"\t\t\tSelecting and fetching templates")
            if CORRECT_MOTION and min_depth_percentile is not None and max_depth_percentile is not None:
                peak_depths = motion_info["peak_locations"]["y"]
                min_depth, max_depth = np.percentile(peak_depths, [min_depth_percentile, max_depth_percentile])
                print(f"Depth limits: {min_depth} - {max_depth} um")
            else:
                min_depth, max_depth = None, None

            if select_by_depth:
                print(f"Selecting templates using depth limits: {min_depth}-{max_depth}")
                templates_info = templates_info.query(f"{min_depth} <= depth_along_probe <= {max_depth}")
            templates_selected_indices = np.random.choice(templates_info.index, size=NUM_UNITS, replace=False)
            print(f"\t\t\tSelected indices: {list(templates_selected_indices)}")
            templates_selected_info = templates_info.loc[templates_selected_indices]

            # fetch templates
            templates_selected = sgen.query_templates_from_database(templates_selected_info)

            if relocate_templates:
                from spikeinterface.generation.drift_tools import move_dense_templates

                print(f"\t\t\tRelocating templates")

                source_probe = templates_selected.probe
                dest_probe = recording.get_probe()
                dest_num_channels = recording.get_num_channels()
                num_samples = templates_selected.num_samples

                channel_locations = recording.get_channel_locations()
                min_depth = min_depth or np.min(channel_locations[:, 1])
                max_depth = max_depth or np.max(channel_locations[:, 1])
                print(f"Relocating templates using depth limits: {min_depth}-{max_depth}")
                template_depths = templates_selected_info["depth_along_probe"].values
                templates_array_moved = np.zeros(
                    (templates_selected.num_units, templates_selected.num_samples, dest_num_channels)
                )
                for i, template in enumerate(templates_selected.templates_array):
                    starting_depth = template_depths[i]
                    final_depth = np.random.uniform(min_depth, max_depth)
                    random_displacement_depth = final_depth - starting_depth
                    displacements = np.array([[0, random_displacement_depth]])
                    template_moved = move_dense_templates(template[None], displacements, source_probe, dest_probe)
                    templates_array_moved[i] = np.squeeze(template_moved)
                templates_relocated = si.Templates(
                    templates_array=templates_array_moved,
                    nbefore=templates_selected.nbefore,
                    sampling_frequency=templates_selected.sampling_frequency,
                    unit_ids=templates_selected.unit_ids,
                    probe=dest_probe
                )
            else:
                templates_relocated = templates_selected

            # scale templates
            print(f"\t\t\tScaling templates between {min_amplitude} and {max_amplitude}")
            templates_scaled = sgen.scale_template_to_range(
                templates=templates_relocated,
                min_amplitude=min_amplitude,
                max_amplitude=max_amplitude
            )
            amplitudes = np.zeros(templates_scaled.num_units)
            extremum_channel_indices = list(si.get_template_extremum_channel(templates_scaled, outputs="index").values())
            for i in range(templates_scaled.num_units):
                amplitudes[i] = np.ptp(templates_scaled.templates_array[i, :, extremum_channel_indices[i]])
            print(f"\t\t\tScaled amplitudes: {np.round(amplitudes, 2)}")

            if templates_scaled.probe.device_channel_indices is None:
                templates_scaled.probe.set_device_channel_indices(np.arange(recording.get_num_channels()))

            print(f"\t\t\tConstructing hybrid recording")
            recording_hybrid, sorting_hybrid = sgen.generate_hybrid_recording(
                recording=recording_preproc,
                templates=templates_scaled,
                motion=motion,
                seed=None,
            )
            print(f"\t\t\t{recording_hybrid}")

            # rename hybrid units with selected indices for provenance
            sorting_hybrid = sorting_hybrid.rename_units(templates_selected_indices)

            print("\t\t\tComputing sorting analyzer")
            analyzer_gt = si.create_sorting_analyzer(
                sorting_hybrid,
                recording_hybrid,
                format="binary_folder",
                folder=flattened_folder / f"analyzer_{case_name}"
            )
            # needed for SNR
            analyzer_gt.compute(["noise_levels", "random_spikes", "templates"])

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
                "recording_dict": recording_dict,
                "template_indices": templates_selected_indices,
                "debug": job_dict["debug"],
                "skip_times": job_dict["skip_times"]
            }
            recording_file_path = recordings_folder / f"job_{case_name}.pkl"
            recording_file_path.write_bytes(pickle.dumps(dump_dict))
            flattened_file_path = flattened_folder / f"job_{case_name}.pkl"
            flattened_file_path.write_bytes(pickle.dumps(dump_dict))

            sorting_hybrid.dump_to_pickle(flattened_folder / f"gt_{case_name}.pkl")

            # generate some plots!
            if CORRECT_MOTION:
                templates_array = recording_hybrid.drifting_templates.templates_array

                # generate raster maps
                analyzer_gt.compute("spike_locations", method="grid_convolution")
                w = sw.plot_drift_raster_map(
                    peaks=motion_info["peaks"],
                    peak_locations=motion_info["peak_locations"],
                    recording=recording_hybrid,
                    cmap="Greys_r",
                    scatter_decimate=10,
                )
                ax = w.ax
                _ = sw.plot_drift_raster_map(
                    sorting_analyzer=analyzer_gt,
                    color_amplitude=False,
                    color="b",
                    scatter_decimate=10,
                    ax=w.ax
                )

                motion = motion_info["motion"]
                _ = ax.plot(
                    motion.temporal_bins_s[0],
                    motion.spatial_bins_um + motion.displacement[0],
                    color="y",
                    alpha=0.5
                )
                ax.set_title(case_name)

                w.figure.savefig(flattened_folder / f"fig-rasters_{case_name}.png", dpi=300)
            else:
                templates_array = recording_hybrid.templates

            # generate templates plots
            templates_obj = si.Templates(
                templates_array * recording.get_channel_gains()[0],
                channel_ids=recording.channel_ids,
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
                ncols=2
            )
            
            w.figure.savefig(flattened_folder / f"fig-templates_{case_name}.pdf")
