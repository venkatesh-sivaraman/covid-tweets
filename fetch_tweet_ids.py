import requests
import json
import wget
import sys
import os
import hashlib
import time
import argparse

# Adapted from zenodo_get: https://zenodo.org/record/1261813
def check_hash(filename, checksum):
    algorithm, value = checksum.split(':')
    if not os.path.exists(filename):
        return value, 'invalid'
    h = hashlib.new(algorithm)
    with open(filename, 'rb') as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            h.update(data)
    digest = h.hexdigest()
    return value, digest

def download_record_file(recordID, file_name, retry=0, pause_time=60, out_dir='./'):
    url = 'https://zenodo.org/api/records/'
    r = requests.get(url + recordID)
    if r.ok:
        js = json.loads(r.text)
        files = js['files']

        desired_file = next((f for f in files if f['links']['self'].endswith(file_name)), None)
        if not desired_file:
            print("Could not find file named {} in Zenodo record.".format(file_name))
            sys.exit(1)
        
        link = desired_file['links']['self']
        size = desired_file['size']/2**20
        print('Link: {}   size: {:.1f} MB'.format(link, size))
        fname = desired_file['key']
        checksum = desired_file['checksum']

        remote_hash, local_hash = check_hash(fname, checksum)

        if remote_hash == local_hash:
            print('{} is already downloaded correctly.'.format(fname))
            return fname

        for _ in range(retry+1):
            try:
                filename = wget.download(link, out=out_dir)
            except Exception:
                print('  Download error.')
                time.sleep(pause_time)
            else:
                break
        else:
            print('  Too many errors.')
            return None

        h1, h2 = check_hash(filename, checksum)
        if h1 == h2:
            print('Checksum is correct. ({})'.format(h1,))
            return os.path.join(out_dir, filename)
        else:
            print('Checksum is INCORRECT!({} got:{})'.format(h1, h2))
            return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Get the latest version of a Zenodo record containing tweets.'))
    parser.add_argument('--record', type=str, default='3723939', dest='record',
                        help='Zenodo record ID pointing to the latest version of the tweet dataset')
    parser.add_argument('--filename', type=str, default='full_dataset_clean.tsv.gz',
                        dest='file_name',
                        help='Name of the file within the record to download')
    parser.add_argument('--out', type=str, default='./',
                        help='Path to the output directory')
    args = parser.parse_args()

    download_record_file(args.record, args.file_name,
                         out_dir=args.out + '/' if not args.out.endswith('/') else args.out)