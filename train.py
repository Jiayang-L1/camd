from run import CAMD_run


# The 'dataset' variable determines the dataset to be used by the model.
# It can take two possible values: "gofundme" or "indiegogo".
dataset = 'gofundme'

CAMD_run(model_name='camd', dataset_name=dataset, seeds=[1], model_save_dir="./pt",
        log_dir="./log", mode='train', gpu_ids=[0]
        )

