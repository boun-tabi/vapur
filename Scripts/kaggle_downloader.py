import kaggle

kaggle.api.dataset_download_files('allen-institute-for-ai/CORD-19-research-challenge', path='../Data/Raw', unzip=True)
