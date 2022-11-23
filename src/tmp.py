def print_experiment_info(experiments):
    for e in experiments:
        print(
            "- experiment_id: {}, name: {}, lifecycle_stage: {}".format(
                e.experiment_id, e.name, e.lifecycle_stage
            )
        )
