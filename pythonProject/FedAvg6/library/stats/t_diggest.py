from crick import TDigest
import numpy as np

# Simulate data on 3 clients
client_data = [
    np.random.normal(loc=0, scale=1, size=1000),
    np.random.normal(loc=5, scale=1, size=1000),
    np.random.normal(loc=10, scale=1, size=1000),
]

# Each client creates its TDigest
client_digests = []
for data in client_data:
    td = TDigest()
    for x in data:
        td.update(x)
    client_digests.append(td)

# Merge all into a global TDigest
global_digest = TDigest()
for td in client_digests:
    global_digest.merge(td)

# Query quantiles
for q in [0.25, 0.5, 0.75, 0.95]:
    print(f"{int(q*100)}th percentile: {global_digest.quantile(q):.2f}")
