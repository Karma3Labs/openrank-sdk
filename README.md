# OpenRank SDK

## EigenTrust by OpenRank
EigenTrust SDk is a python wrapper for https://github.com/Karma3Labs/go-eigentrust

### Input

You will need *local trust* and optionally *pre-trust.*  If pre-trust is not specified, each peer will have an equal weight. Both can be specified
using a CSV or an array of a dict with `i`, `j`, and `v`.

Sample local trust dict variable

```python
localtrust = [{
  "i": "ek",
  "j": "sd",
  "v": 100
}, {
  "i": "vm",
  "j": "sd",
  "v": 100
}, {
  "i": "ek",
  "j": "sd",
  "v": 75
}]
```

Sample local trust (`lt.csv`):

```csv
from,to,value
ek,sd,100
vm,sd,100
ek,vm,75
```

Here we have 3 peers: EK, VM, and SD.
Both EK and VM trust SD by 100.
EK also trusts VM, by 3/4 of how much he trusts SD.

Sample pre trust dict variable
```csv
pre = [{
  "i": "ek",
  "v": 50
}, {
  "i": "vm",
  "v": 100
}]
```

Sample pre-trust (`pt.csv`):

```csv
peer_id,value
ek,50
vm,100
```

Here, both EK and VM are pre-trusted by the network (*a priori* trust).
VM is trusted twice as much as EK.

### Running

To run EigenTrust using the above input:

```python
from openrank_sdk import EigenTrust

api_key = 'your_api_key'
a = EigenTrust(api_key=api_key)

# Option A - Use local variable
a.run_eigentrust(localtrust)
## run with pretrust you've defined rather than the one distributed equally
a.run_eigentrust(localtrust, pretrust)

# Option B - Use CSV
a.run_eigentrust_from_csv("./lt.csv")
## run with pretrust you've defined rather than the one distributed equally
a.run_eigentrust_from_csv("./lt.csv", "./pt.csv")

```

Outputs:
```python
[
  {'i': 'vm', 'v': 0.485969387755102},
  {'i': 'sd', 'v': 0.2933673469387755},
  {'i': 'ek', 'v': 0.22066326530612243}
]
```

Here, the EigenTrust algorithm distributed the network's trust onto the 3 peers:

* EK gets 22.0%
* SD gets 29.3%
* VM gets 48.5%

## Appendix

### Tweaking Alpha

The pre-trust input defines the *relative* ratio
by which the network distributes its *a priori* trust onto trustworthy peers,
in this case EK and VM.

You can also tweak the overall *absolute* strength of the pre-trust.
This parameter, named *alpha*,
represents the portion of the EigenTrust output taken from the pre-trust.
For example, with alpha of 0.2, the EigenTrust output is a blend of 20%
pre-trust and 80% peer-to-peer trust.

The CLI default for alpha is 0.5 (50%).  If you re-run EigenTrust using a lower
alpha of only 0.01 (1%):

```python
from openrank_sdk import EigenTrust

api_key = 'your_api_key'
a = EigenTrust(api_key=api_key, alpha=0.01)

a.run_eigentrust(localtrust, pretrust)
```

We get a different result:

```python
[
  {'address': 'vm', 'score': 0.39451931175281096},
  {'address': 'sd', 'score': 0.4401132971693594},
  {'address': 'ek', 'score': 0.16536739107782936}
]
```

EK and VM's trust shares got lower (EK 21.7% ⇒ 16.5%, VM 48.1% ⇒ 39.5%),
whereas SD's trust share soared (30.2% ⇒ 44%) despite not being pre-trusted.
This is because, with only 1% pre-trust level,
the peer-to-peer trust opinions (where SD is trusted by both EK and VM)
make up for a much larger portion of trust.