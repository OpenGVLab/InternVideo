import os, argparse
from sklearn.utils import shuffle



def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument('dataset', type=str, choices=['mimetics10', 'mimetics'], help='dataset to be built file list')
    parser.add_argument('src_folder', type=str, help='root directory for the frames or videos')
    parser.add_argument('list_file', type=str, help='file list result')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    filelist, labels = [], []
    for cls_id, labelname in enumerate(sorted(os.listdir(args.src_folder))):
        video_path = os.path.join(args.src_folder, labelname)
        for videoname in os.listdir(video_path):
            # get the video file
            video_file = os.path.join(labelname, videoname)
            filelist.append(video_file)
            # get the label
            labels.append(str(cls_id))

    filelist, labels = shuffle(filelist, labels)

    with open(args.list_file, 'w') as f:
        for filepath, label in zip(filelist, labels):
            print(filepath, label)
            f.writelines('%s %s\n'%(filepath, label))
    

