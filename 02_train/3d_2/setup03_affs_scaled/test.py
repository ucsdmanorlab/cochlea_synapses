import argparse
parser = argparse.ArgumentParser()
parser.add_argument("param_fi", help="file with checkpoint name and target voxel size")
args = parser.parse_args()
print(args.param_fi)

params = open(args.param_fi,'r')
checkpoint = params.readline().strip()
target_vox = params.readline().strip()
target_vox = tuple(map(int, target_vox.split(',')))

print(checkpoint, target_vox)
