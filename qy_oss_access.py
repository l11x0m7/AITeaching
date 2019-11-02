from qingstor.sdk.service.qingstor import QingStor
from qingstor.sdk.config import Config

ACCESS_KEY_ID = 'KPXLUFSRVNVNZGFCEPDT'
SECRET_ACCESS_KEY = '9RW7JW2RsIDmArXSdeHhCjYt7A9vHPs6LBT8zSEp'

config = Config(ACCESS_KEY_ID, SECRET_ACCESS_KEY)
qingstor = QingStor(config)


# List all buckets.
output = qingstor.list_buckets()

# Print HTTP status code.
print(output.status_code)

# Print the buckets.
print(output['buckets'])


bucket = qingstor.Bucket('mrc-lxm', 'pek3b')

fname = 'main1.jpg'

with open('trendsetter/img/{}'.format(fname), 'rb') as f:
    output = bucket.put_object(
        fname, body=f
    )

# Print the HTTP status code.
# Example: 201
print(output.status_code)
