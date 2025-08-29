import joblib
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Load a motion file using joblib.")
    parser.add_argument('motion_file', type=str, help='Path to the motion file to load.')
    args = parser.parse_args()

    out_motion_data = joblib.load(args.motion_file)
    motion_data = next(iter(out_motion_data.values()))
    
    print(out_motion_data.keys())
    print(motion_data.keys())
    # print(motion_data)
    breakpoint()

if __name__ == "__main__":
    main()
