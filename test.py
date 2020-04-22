import argparse

parser = argparse.ArgumentParser(description='Age and Gender Inference')
parser.add_argument('-i', '--input', type=str,
                    help='Path to input image or video file. Skip this argument to capture frames from a '
                         'camera.')
parser.add_argument('-ag', "--age-gender", type=str, required=True,
                    default="age", help="mention classification needs to be performed - age or gender")
args = parser.parse_args()

print(args.age_gender)