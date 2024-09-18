from run import CAMD_run


# The 'dataset' variable determines the dataset to be used by the model.
# It can take two possible values: "gofundme" or "indiegogo".
dataset = 'gofundme'

# If 'mode' is set to "test", a 'model_path' variable is required to specify the path
# to the pre-trained model that will be used for testing.
model_path = './pt/xxx.pth'

CAMD_run(model_name='camd', dataset_name=dataset, seeds=[1], model_save_dir="./pt", model_path=model_path,
        log_dir="./log", mode='test', gpu_ids=[0]
        )

