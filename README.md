# OpenRank SDK

## EigenTrust by OpenRank
EigenTrust is powered by [Karma3Labs/GoEigentrust](https://github.com/Karma3Labs/go-eigentrust). Documentation for the openrank API [can be found here](https://docs.openrank.com/).

### Installation

Get started by installing the package in your environment

```
pip install openrank-sdk
```

You can learn how to build your own openrank graphs using the [jupyter notebook linked here](https://github.com/Karma3Labs/openrank-sdk/blob/main/openrank_example.ipynb). Alternatively, you can play with the notebook [directly on hex.tech](https://app.hex.tech/e8198fd2-0779-4dff-a368-06371d3af467/hex/fa721eba-e774-4036-bda1-52323202ef9a/draft/logic?view=notebook).

### Input

You will need *local trust* and optionally *pre-trust.* If pre-trust is not specified, each peer will have an equal weight. Both can be specified using a CSV or an array of a dict with `i`, `j`, and `v`.

Sample local trust dict variable:

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

Here we have 3 peers: EK, VM, and SD. Both EK and VM trust SD by 100. EK also trusts VM, by 3/4 of how much he trusts SD.

Sample pre-trust dict variable:

```python
pretrust = [{
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

Here, both EK and VM are pre-trusted by the network (*a priori* trust). VM is trusted twice as much as EK.

### Tweaking Alpha

The pre-trust input defines the *relative* ratio by which the network distributes its *a priori* trust onto trustworthy peers, in this case EK and VM.

You can also tweak the overall *absolute* strength of the pre-trust. This parameter, named *alpha*, represents the portion of the EigenTrust output taken from the pre-trust. For example, with alpha of 0.2, the EigenTrust output is a blend of 20% pre-trust and 80% peer-to-peer trust.

The CLI default for alpha is 0.5 (50%). If you re-run EigenTrust using a lower alpha of only 0.01 (1%):

```python
from openrank_sdk import EigenTrust

a = EigenTrust(alpha=0.01)
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

EK and VM's trust shares got lower (EK 21.7% ⇒ 16.5%, VM 48.1% ⇒ 39.5%), whereas SD's trust share soared (30.2% ⇒ 44%) despite not being pre-trusted. This is because, with only 1% pre-trust level, the peer-to-peer trust opinions (where SD is trusted by both EK and VM) make up for a much larger portion of trust.

#### Example 1: Ranking Users in the Base Channel on Farcaster based on Token Transfers on Base

[Hex Notebook](https://app.hex.tech/e8198fd2-0779-4dff-a368-06371d3af467/hex/fa721eba-e774-4036-bda1-52323202ef9a)

**Objective:**
Identify top users on the Base channel based on peer-to-peer token transfer transactions on Base.

**Input Local Trust (i, j, v):**
Local trust is created based on stablecoin and native token transfers on Base, Optimism, and Ethereum mainnet among EOAs. To scope the graph down, included addresses from users who have cast in the /base channel on Farcaster.

**Input Seed Trust (I, V):**
Focus on the most active users in the /base channel. Calculated the Seed Trust score as total engagement multiplied by the number of distinct addresses they have transacted with.

**Output Rankings:**
The output will be a ranked list of EOAs based on their trust scores.

#### Example 2: DAO Votes

[Hex Notebook](https://app.hex.tech/f7590f16-5e23-4925-9e84-6656844ee649/hex/2a1f9e4b-76e2-49fe-979a-750043589cbb/draft/logic)
[Video demo explaining the same can be found here.](https://www.loom.com/share/483a3a1fab244207924595d98c056c70?sid=3112f668-3970-474d-9a3f-ed925231eb6e)

**Objective:**
Ranking proposers from Aave, Compound, dYdX, ENS, and Gitcoin DAO based on the equivalent USD amount of votes they receive as trust weight.

**Input Local Trust (i, j, v):**
This example uses a Dune query for generating the local trust matrix consisting of:
- `i`: EOA of the address who voted
- `j`: EOA of the proposer who got the vote
- `v`: Equivalent USD amount of votes received

**Input PreTrust (I, V):**
Here it is set to default, making all voters within the network have a common trust score.

**Output Rankings:**
The output will be a ranked list of proposers from the specified DAOs (Aave, Compound, dYdX, ENS, and Gitcoin) based on the total equivalent USD amount of votes they received, weighted by trust.


### Running

To run EigenTrust using the above input:

```python
from openrank_sdk import EigenTrust

a = EigenTrust()

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

### Methods

#### `__init__(self, **kwargs)`

Initialize the EigenTrust class with optional parameters.

Args:
- `alpha (float)`: The alpha value for EigenTrust.
- `host_url (str)`: The host URL for the EigenTrust service.
- `timeout (int)`: The timeout value in millisecond for the EigenTrust requests.
- `api_key (str)`: The API key for authentication.

Example:

```python
et = EigenTrust(alpha=0.5, host_url="https://example.com", timeout=900000, api_key="your_api_key")
```

#### `run_eigentrust(self, localtrust: List[IJV], pretrust: List[IV]=None) -> List[Score]`

Run the EigenTrust algorithm using the provided local trust and pre-trust data.

Args:
- `localtrust (List[IJV])`: List of local trust values.
- `pretrust (List[IV], optional)`: List of pre-trust values. Defaults to None.

Returns:
- `List[Score]`: List of computed scores.

Example:

```python
localtrust = [{'i': 'A', 'j': 'B', 'v': 0.5}, {'i': 'B', 'j': 'C', 'v': 0.6}]
pretrust = [{'i': 'A', 'v': 1.0}]
scores = et.run_eigentrust(localtrust, pretrust)
```

#### `run_eigentrust_from_csv(self, localtrust_filename: str, pretrust_filename: str = None) -> List[Score]`

Run the EigenTrust algorithm using local trust and pre-trust data from CSV files.

Args:
- `localtrust_filename (str)`: The filename of the local trust CSV file.
- `pretrust_filename (str, optional)`: The filename of the pre-trust CSV file. Defaults to None.

Returns:
- `List[Score]`: List of computed scores.

Example:

```python
scores = et.run_eigentrust_from_csv('localtrust.csv', 'pretrust.csv')
```

#### `_send_go_eigentrust_req(self, pretrust: list[dict], max_pt_id: int, localtrust: list[dict], max_lt_id: int)`

Send a request to the EigenTrust service to compute scores.

Args:
- `pretrust (list[dict])`: List of pre-trust values.
- `max_pt_id (int)`: The maximum pre-trust ID.
- `localtrust (list[dict])`: List of local trust values.
- `max_lt_id (int)`: The maximum local trust ID.

Returns:
- `List[dict]`: List of computed scores.

Example:

```python
scores = self._send_go_eigentrust_req(pretrust, max_pt_id, localtrust, max_lt_id)
```

#### `export_scores_to_csv(self, scores: List[Score], filepath: str, headers: List[str])`

Export the computed scores to a CSV file.

Args:
- `scores (List[Score])`: List of computed scores.
- `filepath (str)`: The path to the output CSV file.
- `headers (List[str])`: List of CSV headers.

Example:

```python
et.export_scores_to_csv(scores, 'scores.csv', ['i', 'v'])
```

#### `export_csv_to_dune(self, filepath: str, headers: List[str], tablename: str, description: str, is_private: bool, api_key: str)`

Export a CSV file to the Dune Analytics platform.

Args:
- `filepath (str)`: The path to the CSV file.
- `headers (List[str])`: List of CSV headers.
- `tablename (str)`: The name of the table on Dune Analytics.
- `description (str)`: Description of the table.
- `is_private (bool)`: Whether the table is private.
- `api_key (str)`: The API key for Dune Analytics.

Example:

```python
et.export_csv_to_dune('scores.csv', ['i', 'v'], 'my_table', 'Table description', False, 'your_api_key')
```

#### `_upload_csv(self, data: List[dict], headers: List[str], endpoint: str, overwrite: bool) -> str`

Upload CSV data to the backend server.

Args:
- `data (List[dict])`: List of data to be uploaded.
- `headers (List[str])`: List of CSV headers.
- `endpoint (str)`: The endpoint for the upload.
- `overwrite (bool)`: Whether to overwrite existing data.

Returns:
- `str`: URL of the uploaded data.

Example:

```python
data = [{'i': 'A', 'j': 'B', 'v': 0.5}, {'i': 'B', 'j': 'C', 'v': 0.6}]
url = et._upload_csv(data, ['i', 'j', 'v'], 'localtrust/123', True)
```

#### `_download_csv(self, endpoint: str) -> List[dict]`

Download CSV data from the backend server.

Args:
- `endpoint (str)`: The endpoint for the download.

Returns:
- `List[dict]`: List of downloaded data.

Example:

```python
data = et._download_csv('localtrust/123')
```

#### `_convert_to_ijv(self, data: List[dict]) -> List[IJV]`

Convert a list of dictionaries to a list of IJV objects.

Args:
- `data (List[dict])`: List of data to be converted.

Returns:
- `List[IJV]`: List of IJV objects.

Example:

```python
ijv_list = et._convert_to_ijv(data)
```

#### `_convert_to_iv(self, data: List[dict]) -> List[IV]`

Convert a

 list of dictionaries to a list of IV objects.

Args:
- `data (List[dict])`: List of data to be converted.

Returns:
- `List[IV]`: List of IV objects.

Example:

```python
iv_list = et._convert_to_iv(data)
```

#### `_convert_to_score(self, data: List[dict]) -> List[Score]`

Convert a list of dictionaries to a list of Score objects.

Args:
- `data (List[dict])`: List of data to be converted.

Returns:
- `List[Score]`: List of Score objects.

Example:

```python
score_list = et._convert_to_score(data)
```

#### `run_eigentrust_from_id(self, localtrust_id: str, pretrust_id: str = None) -> Tuple[List[Score], str]`

Run the EigenTrust algorithm using local trust and pre-trust data identified by their IDs.

Args:
- `localtrust_id (str)`: The ID of the local trust data.
- `pretrust_id (str, optional)`: The ID of the pre-trust data. Defaults to None.

Returns:
- `Tuple[List[Score], str]`: List of computed scores and the URL of the results.

Example:

```python
scores, url = et.run_eigentrust_from_id('localtrust123', 'pretrust123')
```

#### `run_and_publish_eigentrust_from_id(self, id: str, localtrust_id: str, pretrust_id: str = None, **kwargs) -> Tuple[List[Score], str]`

Run the EigenTrust algorithm using local trust and pre-trust data identified by their IDs, and publish the results.

Args:
- `id (str)`: The ID for publishing the results.
- `localtrust_id (str)`: The ID of the local trust data.
- `pretrust_id (str, optional)`: The ID of the pre-trust data. Defaults to None.

Returns:
- `Tuple[List[Score], str]`: List of computed scores and the URL of the published results.

Example:

```python
scores, publish_url = et.run_and_publish_eigentrust_from_id('result123', 'localtrust123', 'pretrust123')
```

#### `run_and_publish_eigentrust(self, id: str, localtrust: List[IJV], pretrust: List[IV] = None, **kwargs) -> Tuple[List[Score], str]`

Run the EigenTrust algorithm using local trust and pre-trust data, and publish the results.

Args:
- `id (str)`: The ID for publishing the results.
- `localtrust (List[IJV])`: List of local trust values.
- `pretrust (List[IV], optional)`: List of pre-trust values. Defaults to None.

Returns:
- `Tuple[List[Score], str]`: List of computed scores and the URL of the published results.

Example:

```python
localtrust = [{'i': 'A', 'j': 'B', 'v': 0.5}, {'i': 'B', 'j': 'C', 'v': 0.6}]
pretrust = [{'i': 'A', 'v': 1.0}]
scores, publish_url = et.run_and_publish_eigentrust('result123', localtrust, pretrust)
```

#### `publish_eigentrust(self, id: str, result: List[Score], **kwargs) -> str`

Publish the EigenTrust results.

Args:
- `id (str)`: The ID for publishing the results.
- `result (List[Score])`: List of computed scores.

Returns:
- `str`: URL of the published results.

Example:

```python
url = et.publish_eigentrust('result123', scores)
```

#### `fetch_eigentrust(self, id: str, **kwargs) -> List[Score]`

Fetch the EigenTrust results by ID.

Args:
- `id (str)`: The ID of the results to fetch.

Returns:
- `List[Score]`: List of fetched scores.

Example:

```python
scores = et.fetch_eigentrust('result123')
```

#### `publish_localtrust(self, id: str, result: List[IJV], **kwargs) -> str`

Publish the local trust data.

Args:
- `id (str)`: The ID for publishing the local trust data.
- `result (List[IJV])`: List of local trust values.

Returns:
- `str`: URL of the published local trust data.

Example:

```python
url = et.publish_localtrust('localtrust123', localtrust)
```

#### `fetch_localtrust(self, id: str, **kwargs) -> List[IJV]`

Fetch the local trust data by ID.

Args:
- `id (str)`: The ID of the local trust data to fetch.

Returns:
- `List[IJV]`: List of fetched local trust values.

Example:

```python
localtrust = et.fetch_localtrust('localtrust123')
```

#### `publish_pretrust(self, id: str, result: List[IV], **kwargs) -> str`

Publish the pre-trust data.

Args:
- `id (str)`: The ID for publishing the pre-trust data.
- `result (List[IV])`: List of pre-trust values.

Returns:
- `str`: URL of the published pre-trust data.

Example:

```python
url = et.publish_pretrust('pretrust123', pretrust)
```

#### `fetch_pretrust(self, id: str, **kwargs) -> List[IV]`

Fetch the pre-trust data by ID.

Args:
- `id (str)`: The ID of the pre-trust data to fetch.

Returns:
- `List[IV]`: List of fetched pre-trust values.

Example:

```python
pretrust = et.fetch_pretrust('pretrust123')
```