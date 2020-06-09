import pandas as pd
import os
import argparse

dtype_spec = {'id': str, 'user_id': str, 'reply_to_id': str, 'reply_to_user': str}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Concatenate CSV files written by the hydrate '
                                     'workers.'))
    parser.add_argument('input', type=str,
                        help='Path to data directory in which worker files were written')
    parser.add_argument('output', type=str,
                        help='Path to output CSV file')

    args = parser.parse_args()

    filenames = [os.path.join(args.input, path) for path in os.listdir(args.input)
                 if path.endswith(".csv") and not path.startswith(".")]
    df = pd.concat([pd.read_csv(filename, dtype=dtype_spec, lineterminator='\n')
                    for filename in filenames if filename != args.output])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')].reset_index(drop=True)
    df.to_csv(args.output, line_terminator='\n')
    print("Saved {} tweets to {}.".format(len(df), os.path.basename(args.output)))
