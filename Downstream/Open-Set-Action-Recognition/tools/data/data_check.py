import os, argparse
import cv2
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check the data')
    parser.add_argument('video_path', type=str, help='The video path')
    parser.add_argument('dataset_list', type=str, help='The list file of dataset.')
    parser.add_argument('split', type=str, choices=['train', 'test', 'val'], help='The split of the data.')
    args = parser.parse_args()
    
    # parse the filelist into list
    filelist, labels = [], []
    assert os.path.exists(args.dataset_list), 'File list does not exist! %s'%(args.dataset_list)
    with open(args.dataset_list, 'r') as f:
        for line in f.readlines():
            filename, label = line.strip().split(' ')
            filelist.append(filename)
            labels.append(label)
    # checking
    valid_files, invalid_files, valid_labels = [], [], []
    for filename, label in tqdm(zip(filelist, labels), total=len(filelist), desc=args.split):
        videofile = os.path.join(args.video_path, filename)
        if not os.path.exists(videofile):
            # file not exist
            invalid_files.append(filename)
        else:
            # file cannot be read
            cap = cv2.VideoCapture(videofile)
            ret, frame = cap.read()
            if not ret:
                invalid_files.append(filename)
            else:
                valid_files.append(filename)
                valid_labels.append(label)
            cap.release()
    # print
    print('Valid file number: %d, Invalid number: %d'%(len(valid_files), len(invalid_files)))

    if len(invalid_files) > 0:
        tmp_listfile = os.path.join(os.path.dirname(args.dataset_list), args.dataset_list.split('/')[-1][:-4] + '_temp.txt')
        with open(tmp_listfile, 'w') as f:
            for filename, label in zip(valid_files, valid_labels):
                f.writelines('%s %s\n'%(filename, label))

        print('\nFollowing files are invalid: ')
        for filename in invalid_files:
            print(invalid_files)
        