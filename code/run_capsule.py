""" top level run script """
# GENERAL IMPORTS
import os
import logging
import sys

# this is needed to limit the number of scipy threads
# and let spikeinterface handle parallelization
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
import json
import pickle
import numpy as np
from pathlib import Path

import probeinterface as pi
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.generation as sgen
import spikeinterface.curation as scur
import spikeinterface.widgets as sw


data_folder = Path("../data")
results_folder = Path("../results")
PARAMS_FILE = Path("../params.json")

margin_from_border = 100


def run() -> None:
    if not PARAMS_FILE.is_file():
        raise ValueError(f"Params file not found: {PARAMS_FILE}")
    with open(PARAMS_FILE, "r") as f:
        full_params = json.load(f)
    hybrid_generation_params = full_params["hybrid_generation"]
    LOGGING = full_params.get("logging", None)

    # setup logging before any other logging call
    if LOGGING is None:
        logging.basicConfig(level="INFO", stream=sys.stdout, format="%(message)s")
    else:
        if LOGGING["package"] == "logging":
            logging_cfg = LOGGING.get("logging_cfg", {})
            logging.basicConfig(stream=sys.stdout, **logging_cfg)
        elif LOGGING["package"] == "log-schema":
            import log_schema

            pipeline_name = LOGGING.get("pipeline_name", "AIND Ephys Hybrid Benchmark")
            acquisition_name = LOGGING.get("acquisition_name", "unknown")

            config = LOGGING.get("logging_cfg")
            if config is not None and len(config) == 0:
                config = None
            log_schema.setup_logging(
                config=config,
                model={
                    "pipeline_name": pipeline_name,
                    "acquisition_name": acquisition_name,
                    "process_name": "Hybrid Generation"
                }
            )

    logging.info("Begin processing...", extra={"event_type": "stage_start"})


    MIN_AMP = float(hybrid_generation_params.get("min_amp", 50))
    MAX_AMP = float(hybrid_generation_params.get("max_amp", 200))
    MIN_DEPTH_PERC = float(hybrid_generation_params.get("min_depth_percentile", 5))
    MAX_DEPTH_PERC = float(hybrid_generation_params.get("max_depth_percentile", 95))
    NUM_UNITS = int(hybrid_generation_params.get("num_units", 10))
    NUM_CASES = int(hybrid_generation_params.get("num_cases", 5))
    CORRECT_MOTION = hybrid_generation_params.get("correct_motion", True)

    logging.info(f"MIN_AMP: {MIN_AMP}")
    logging.info(f"MAX_AMP: {MAX_AMP}")
    logging.info(f"MIN_DEPTH_PERCENTILE: {MIN_DEPTH_PERC}")
    logging.info(f"MAX_DEPTH_PERCENTILE: {MAX_DEPTH_PERC}")
    logging.info(f"NUM_UNITS: {NUM_UNITS}")
    logging.info(f"NUM_CASES: {NUM_CASES}")
    logging.info(f"CORRECT_MOTION: {CORRECT_MOTION}")

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
    logging.info(f"Found {len(job_dicts)} JSON job files")

    # Use CO_CPUS/N_JOBS_EXT env variable if available
    N_JOBS_EXT = os.getenv("CO_CPUS") or os.getenv("N_JOBS_EXT")
    N_JOBS = int(N_JOBS_EXT) if N_JOBS_EXT is not None else -1
    si.set_global_job_kwargs(n_jobs=N_JOBS, progress_bar=False)

    templates_info = sgen.fetch_templates_database_info()

    # TODO: check this at database creation!
    templates_info = templates_info.query("amplitude_uv > 0")
    
    # for each JSON file, we now create hybrid recordings
    for job_dict in job_dicts:
        recording_name = job_dict["recording_name"]
        logging.info(f"\n\nCreating hybrid recordings for {recording_name}")
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
            logging.info("\tCould not load probes info. Selecting Neuropixels Ultra templates")
            templates_info = templates_info.query("probe == 'Neuropixels Ultra'")
            relocate_templates = True
            select_by_depth = False
        elif "1.0" in model_name:
            logging.info("\tProbe model: {model_name}. Selecting Neuropixels 1.0 templates")
            templates_info = templates_info.query("probe == 'Neuropixels 1.0'")
            relocate_templates = False
            select_by_depth = True
        else:
            logging.info("\tProbe model: {model_name}. Selecting Neuropixels Ultra templates")
            templates_info = templates_info.query("probe == 'Neuropixels Ultra'")
            relocate_templates = True
            select_by_depth = False

        logging.info(f"\tSelected {len(templates_info)} templates from database")

        # skip times if non-monotonically increasing
        if job_dict["skip_times"]:
            recording.reset_times()
        logging.info(f"\tRecording: {recording}")

        # preprocess
        recording_preproc = spre.highpass_filter(recording)
        recording_preproc = spre.common_reference(recording_preproc)

        motion = None
        min_depth_percentile = MIN_DEPTH_PERC
        max_depth_percentile = MAX_DEPTH_PERC
        min_depth, max_depth = None, None
        if CORRECT_MOTION:
            logging.info("\tEstimating motion")
            motion_folder = flattened_folder / f"motion_{recording_name}"
            motion_figure_file = flattened_folder / f"fig-motion_{recording_name}.png"

            # use compute motion
            motion, motion_info = spre.compute_motion(
                recording_preproc.astype(float),
                preset="dredge_fast",
                folder=motion_folder,
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
                    logging.info(f"\t\t\tDepth limits: {np.round(min_depth, 2)} - {np.round(max_depth, 2)} um")


        logging.info(f"\tGenerating hybrid recordings")
        min_amplitude = MIN_AMP
        max_amplitude = MAX_AMP
        for case in range(NUM_CASES):
            logging.info(f"\t\tGenerating case: {case}")
            case_name = f"{recording_name}_{case}"

            # sample templates
            logging.info(f"\t\t\tSelecting and fetching templates")

            if select_by_depth:
                logging.info(f"\t\t\tSelecting templates using depth limits: {min_depth}-{max_depth}")
                templates_info = templates_info.query(f"{min_depth} <= depth_along_probe <= {max_depth}")

            if relocate_templates:
                from spikeinterface.generation.drift_tools import move_dense_templates

                # select more UNITS, to account for bad interpolations
                templates_selected_indices = np.random.choice(templates_info.index, size=2 * NUM_UNITS, replace=False)
                templates_selected_info = templates_info.loc[templates_selected_indices]

                # fetch templates
                templates_selected = sgen.query_templates_from_database(templates_selected_info)

                logging.info(f"\t\t\tRelocating templates")

                source_probe = templates_selected.probe
                dest_probe = recording.get_probe()
                dest_num_channels = recording.get_num_channels()
                num_samples = templates_selected.num_samples

                # check if needs x-shift correction (e.g., on different shanks)
                src_x = np.min(source_probe.contact_positions[:, 0])
                dst_x = np.min(dest_probe.contact_positions[:, 0])
                if np.abs(dst_x - src_x) > 100:
                    logging.info(f"\t\t\tShifting source probe by {dst_x - src_x} in x direction")
                    source_probe.move([dst_x - src_x, 0])

                sparsity = si.compute_sparsity(templates_selected, method="radius", radius_um=100)

                channel_locations = recording.get_channel_locations()
                min_depth = min_depth or np.min(channel_locations[:, 1])
                max_depth = max_depth or np.max(channel_locations[:, 1])
                logging.info(f"\t\t\tRelocating templates using depth limits: {min_depth}-{max_depth}")
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
                    logging.info(f"\t\t\tCould not relocate all templates. Only added {num_relocated_templates}")
                templates_array_moved = templates_array_moved[:num_relocated_templates]
                templates_selected_indices = templates_selected_indices[:num_relocated_templates]
                unit_ids = templates_selected.unit_ids[:num_relocated_templates]

                logging.info(f"\t\t\tSelected indices: {[int(t) for t in templates_selected_indices]}")
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
                logging.info(f"\t\t\tSelected indices: {[int(t) for t in templates_selected_indices]}")
                templates_selected_info = templates_info.loc[templates_selected_indices]

                # fetch templates
                templates_selected = sgen.query_templates_from_database(templates_selected_info)
                templates_relocated = templates_selected

            # scale templates
            logging.info(f"\t\t\tScaling templates between {min_amplitude} and {max_amplitude}")
            templates_scaled = sgen.scale_template_to_range(
                templates=templates_relocated,
                min_amplitude=min_amplitude,
                max_amplitude=max_amplitude
            )
            amplitudes = np.zeros(templates_scaled.num_units)
            extremum_channel_indices = list(si.get_template_extremum_channel(templates_scaled, outputs="index").values())
            for i in range(templates_scaled.num_units):
                amplitudes[i] = np.ptp(templates_scaled.templates_array[i, :, extremum_channel_indices[i]])
            logging.info(f"\t\t\tScaled amplitudes: {np.round(amplitudes, 2)}")

            if templates_scaled.probe.device_channel_indices is None:
                templates_scaled.probe.set_device_channel_indices(np.arange(recording.get_num_channels()))

            logging.info(f"\t\t\tConstructing hybrid recording")
            recording_hybrid, sorting_hybrid = sgen.generate_hybrid_recording(
                recording=recording_preproc,
                templates=templates_scaled,
                are_templates_scaled=True,
                motion=motion,
                seed=None,
            )
            logging.info(f"\t\t\t{recording_hybrid}")

            # rename hybrid units with selected indices for provenance
            sorting_hybrid = sorting_hybrid.rename_units(templates_selected_indices)

            logging.info("\t\t\tComputing sorting analyzer")
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
    logging.info("Pipeline stage completed", extra={"event_type": "stage_complete"})


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logging.exception("Pipeline stage failed", extra={"event_type": "stage_error"})
        raise
