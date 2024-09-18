import logging
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

__all__ = ['MMDataLoader']
logger = logging.getLogger('CAMD')


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'gofundme': self.__init_gofundme,
            'indiegogo': self.__init_indiegogo,
        }
        DATASET_MAP[args['dataset_name']]()

    def __init_gofundme(self):
        data = pd.read_pickle(f"{self.args['featurePath']}_{self.mode}.pkl")
        self.text = data['text_glove']
        self.image = data['image_emb']
        self.meta = data['metadata']
        self.label_re = np.array(data['target_re'])

        logger.info(f"{self.mode} samples: {self.label_re.shape}")

    def __init_indiegogo(self):
        data = pd.read_pickle(f"{self.args['featurePath']}_{self.mode}.pkl")
        self.text = data['text_glove']
        self.image = data['image_emb']
        self.meta = data['metadata']
        self.label_re = np.array(data['target_re'])

        logger.info(f"{self.mode} samples: {self.label_re.shape}")

    def __len__(self):
        return len(self.label_re)

    def __getitem__(self, index):
        sample = {
            'text': self.text[index],
            'image': self.image[index][0],
            'meta': self.meta[index],
            'labels': self.label_re[index],
        }

        return sample


def gofundme_collate_fn(batch):
    image = [batch[x]['image'] for x in range(len(batch))]
    text = [batch[x]['text'] for x in range(len(batch))]
    meta = [batch[x]['meta'] for x in range(len(batch))]
    labels = [batch[x]['labels'] for x in range(len(batch))]

    return meta, text, image, labels


def indiegogo_collate_fn(batch):
    image = [batch[x]['image'] for x in range(len(batch))]
    text = [batch[x]['text'] for x in range(len(batch))]
    meta = [batch[x]['meta'] for x in range(len(batch))]
    labels = [batch[x]['labels'] for x in range(len(batch))]

    return meta, text, image, labels


def MMDataLoader(args, num_workers):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if args['dataset_name'] == 'gofundme':
        dataLoader = {
            ds: DataLoader(datasets[ds],
                           batch_size=args['batch_size'],
                           num_workers=num_workers,
                           collate_fn=gofundme_collate_fn,
                           shuffle=True,
                           )
            for ds in datasets.keys()
        }
    elif args['dataset_name'] == 'indiegogo':
        dataLoader = {
            ds: DataLoader(datasets[ds],
                           batch_size=args['batch_size'],
                           num_workers=num_workers,
                           collate_fn=indiegogo_collate_fn,
                           shuffle=True)
            for ds in datasets.keys()
        }

    return dataLoader
