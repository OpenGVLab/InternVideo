import os
from tqdm import tqdm


def fix_listfile(file_split, phase):
    assert os.path.exists(file_split), 'File does not exist! %s'%(file_split)
    filename = file_split.split('/')[-1]
    file_split_new = os.path.join(dataset_path, 'temp_' + filename)
    with open(file_split_new, 'w') as fw:
        with open(file_split, 'r') as fr:
            for line in tqdm(fr.readlines(), desc=phase):
                foldername = line.split('/')[0]
                if foldername == phase:
                    continue
                fw.writelines(phase + '/' + line)
    os.remove(file_split)
    os.rename(file_split_new, file_split)


if __name__ == '__main__':
    dataset_path = '../../../data/mit'

    file_split = os.path.join(dataset_path, 'mit_train_list_videos.txt')
    fix_listfile(file_split, 'training')
    
    file_split = os.path.join(dataset_path, 'mit_val_list_videos.txt')
    fix_listfile(file_split, 'validation')