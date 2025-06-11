""" top level run script """
# GENERAL IMPORTS
import os

# this is needed to limit the number of scipy threads
# and let spikeinterface handle parallelization
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import warnings
import json
import pickle
import numpy as np
from pathlib import Path

import probeinterface as pi
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

win_step_norm_group = parser.add_mutually_exclusive_group()
win_step_norm_help = "Percent of win_step motion parameter with respect to probe span. Default: 0.1"
win_step_norm_group.add_argument("--win-step-norm", help=win_step_norm_help)
win_step_norm_group.add_argument("static_win_step_norm", nargs="?", default="0.1", help=win_step_norm_help)

win_scale_norm_group = parser.add_mutually_exclusive_group()
win_scale_norm_help = "Percent of win_scale motion parameter with respect to probe span. Default: 0.1"
win_scale_norm_group.add_argument("--win-scale-norm", help=win_scale_norm_help)
win_scale_norm_group.add_argument("static_win_scale_norm", nargs="?", default="0.1", help=win_scale_norm_help)


if __name__ == "__main__":
    args = parser.parse_args()

    MIN_AMP = float(args.min_amp or args.static_min_amp)
    MAX_AMP = float(args.max_amp or args.static_max_amp)
    MIN_DEPTH_PERC = float(args.min_depth_percentile or args.static_min_depth_percentile)
    MAX_DEPTH_PERC = float(args.max_depth_percentile or args.static_max_depth_percentile)
    NUM_UNITS = int(args.num_units or args.static_num_units)
    NUM_CASES = int(args.num_cases or args.static_num_cases)
    WIN_STEP_NORM = float(args.win_step_norm or args.static_win_step_norm)
    WIN_SCALE_NORM = float(args.win_scale_norm or args.static_win_scale_norm)
    
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
    print(f"WIN_STEP_NORM: {WIN_STEP_NORM}")
    print(f"WIN_SCALE_NORM: {WIN_SCALE_NORM}")

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
        recording = si.load(job_dict["recording_dict"], base_folder=data_folder)

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
        min_depth_percentile = MIN_DEPTH_PERC
        max_depth_percentile = MAX_DEPTH_PERC
        min_depth, max_depth = None, None
        if CORRECT_MOTION:
            print("\tEstimating motion")
            motion_folder = flattened_folder / f"motion_{recording_name}"
            motion_figure_file = flattened_folder / f"fig-motion_{recording_name}.png"

            probe_span = np.ptp(recording.get_channel_locations()[:, 1])

            estimate_motion_kwargs = {}
            win_step_um = WIN_STEP_NORM * probe_span
            estimate_motion_kwargs["win_step_um"] = win_step_um
            win_scale_um = WIN_SCALE_NORM * probe_span
            estimate_motion_kwargs["win_scale_um"] = win_scale_um
            print(f"\t\tEstimate motion kwargs: {estimate_motion_kwargs}")

            # use compute motion
            motion, motion_info = spre.compute_motion(
                recording_preproc.astype(float),
                preset="dredge_fast",
                folder=motion_folder,
                estimate_motion_kwargs=estimate_motion_kwargs,
                output_motion_info=True,
                raise_error=False
            )

            if motion is not None:
                w = sw.plot_motion_info(
                    motion_info,
                    recording_preproc,
                    color_amplitude=True,
                    scatter_decimate=10,
                    amplitude_cmap="Greys_r"
                )
                w.figure.savefig(motion_figure_file, dpi=300)

                if min_depth_percentile is not None and max_depth_percentile is not None:
                    peak_depths = motion_info["peak_locations"]["y"]
                    min_depth, max_depth = np.percentile(peak_depths, [min_depth_percentile, max_depth_percentile])
                    print(f"\t\t\tDepth limits: {np.round(min_depth, 2)} - {np.round(max_depth, 2)} um")


        print(f"\tGenerating hybrid recordings")
        min_amplitude = MIN_AMP
        max_amplitude = MAX_AMP
        for case in range(NUM_CASES):
            print(f"\t\tGenerating case: {case}")
            case_name = f"{recording_name}_{case}"

            # sample templates
            print(f"\t\t\tSelecting and fetching templates")

            if select_by_depth:
                print(f"\t\t\tSelecting templates using depth limits: {min_depth}-{max_depth}")
                templates_info = templates_info.query(f"{min_depth} <= depth_along_probe <= {max_depth}")

            if relocate_templates:
                from spikeinterface.generation.drift_tools import move_dense_templates

                # select more UNITS, to account for bad interpolations
                templates_selected_indices = np.random.choice(templates_info.index, size=2 * NUM_UNITS, replace=False)
                templates_selected_info = templates_info.loc[templates_selected_indices]

                # fetch templates
                templates_selected = sgen.query_templates_from_database(templates_selected_info)

                print(f"\t\t\tRelocating templates")

                source_probe = templates_selected.probe
                dest_probe = recording.get_probe()
                dest_num_channels = recording.get_num_channels()
                num_samples = templates_selected.num_samples

                # check if needs x-shift correction (e.g., on different shanks)
                src_x = np.min(source_probe.contact_positions[:, 0])
                dst_x = np.min(dest_probe.contact_positions[:, 0])
                if np.abs(dst_x - src_x) > 100:
                    print(f"\t\t\tShifting source probe by {dst_x - src_x} in x direction")
                    source_probe.move([dst_x - src_x, 0])

                sparsity = si.compute_sparsity(templates_selected, method="radius", radius_um=100)

                channel_locations = recording.get_channel_locations()
                min_depth = min_depth or np.min(channel_locations[:, 1])
                max_depth = max_depth or np.max(channel_locations[:, 1])
                print(f"\t\t\tRelocating templates using depth limits: {min_depth}-{max_depth}")
                template_depths = templates_selected_info["depth_along_probe"].values
                templates_array_moved = np.zeros(
                    (templates_selected.num_units, templates_selected.num_samples, dest_num_channels)
                )
                num_relocated_templates = 0
                for i, unit_id in enumerate(templates_selected.unit_ids):
                    if num_relocated_templates == NUM_UNITS:
                        break
                    template = templates_selected.templates_array[i]
                    starting_depth = template_depths[i]
                    final_depth = np.random.uniform(min_depth, max_depth)
                    random_displacement_depth = final_depth - starting_depth
                    displacements = np.array([[0, random_displacement_depth]])
                    # only select sparse channels for better signal quality
                    sparse_channel_mask = sparsity.mask[i]
                    template_sparse = template[:, sparse_channel_mask]
                    source_probe_sparse = pi.Probe.from_numpy(source_probe.to_numpy()[sparse_channel_mask])
                    template_moved = move_dense_templates(template_sparse[None], displacements, source_probe_sparse, dest_probe)
                    template_moved = np.squeeze(template_moved)
                    if np.ptp(template_moved) > 0.5 * np.ptp(template):
                        templates_array_moved[num_relocated_templates] = np.squeeze(template_moved)
                        num_relocated_templates += 1

                if num_relocated_templates < NUM_UNITS:
                    print(f"\t\t\tCould not relocate all templates. Only added {num_relocated_templates}")
                templates_array_moved = templates_array_moved[:num_relocated_templates]
                templates_selected_indices = templates_selected_indices[:num_relocated_templates]
                unit_ids = templates_selected.unit_ids[:num_relocated_templates]

                print(f"\t\t\tSelected indices: {[int(t) for t in templates_selected_indices]}")
                templates_relocated = si.Templates(
                    templates_array=templates_array_moved,
                    nbefore=templates_selected.nbefore,
                    sampling_frequency=templates_selected.sampling_frequency,
                    unit_ids=unit_ids,
                    probe=dest_probe
                )
            else:
                # select more UNITS, to account for bad interpolations
                templates_selected_indices = np.random.choice(templates_info.index, size=NUM_UNITS, replace=False)
                print(f"\t\t\tSelected indices: {[int(t) for t in templates_selected_indices]}")
                templates_selected_info = templates_info.loc[templates_selected_indices]

                # fetch templates
                templates_selected = sgen.query_templates_from_database(templates_selected_info)
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
