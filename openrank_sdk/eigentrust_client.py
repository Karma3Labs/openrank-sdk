from __future__ import annotations

import csv
import enum
import io
import logging
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass, fields, replace
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Literal, Optional, \
    Tuple, Union

import httpx
import pandas as pd
from dataclasses_json import LetterCase, dataclass_json
from dataclasses_json.core import Json

from openrank_sdk import trust
from openrank_sdk.trust import IJV, IJV_CSV_HEADERS, IV, IV_CSV_HEADERS, \
    SCORE_CSV_HEADERS, Score

SchemaType = Literal["inline", "objectstorage", "stored"]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ComputeRequestBody:
    """Compute request body."""
    local_trust: trust.Matrix
    initial_trust: Optional[trust.Vector] = None
    pre_trust: Optional[trust.Vector] = None
    alpha: Optional[float] = None
    epsilon: Optional[float] = None
    flat_tail: Optional[int] = None
    num_leaders: Optional[int] = None
    max_iterations: Optional[int] = None
    min_iterations: Optional[int] = None
    check_freq: Optional[int] = None

    def encode(self) -> Json:
        d = self.to_dict()
        null_keys = {k for k, v in d.items() if v is None}
        for k in null_keys:
            del d[k]
        return d


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ComputeWithStatsResponse:
    """Successful ``/compute-with-stats`` endpoint response."""
    eigen_trust: trust.Vector
    flat_tail_stats: FlatTailStats


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class FlatTailStats:
    """Flat-tail algorithm stats and peer ranking."""
    length: int
    threshold: int
    delta_norm: float
    ranking: List[int]


class ScoreScale(enum.Enum):
    """EigenTrust score scale."""
    RAW = 'raw'
    """Raw EigenTrust score value in [0..1] range.

    The number represents the peer's share of the overall trust circulating
    in the network, e.g. 0.1 means 10% of the total trust is bestowed
    on the peer.
    """

    PERCENT = 'percent'
    """Same as `RAW` but in percents, in [0..100] range."""

    LOG = 'log'
    """Log distance from the full trust, in [0..inf] range.

    Defined as `log(raw_value, 0.1)`, this represents the distance from
    the full trust: 0/1/2/3/... represents 100%/10%/1%/0.1%/... of the total
    trust circulating in the network.  Lower distance means higher trust.

    Score values in this scale tend to form a bell curve.
    """


_SCORE_SCALERS: Dict[ScoreScale, Callable[[float], float]] = {
    ScoreScale.RAW: lambda v: v,
    ScoreScale.PERCENT: lambda v: v * 100,
    ScoreScale.LOG: lambda v: math.log(v, 0.1),
}


def snake2camel(s: str) -> str:
    return ''.join(
        w.capitalize() if i > 0 else w.lower()
        for i, w in enumerate(s.split('_'))
    )


def replace_with_kwargs(obj, kwargs: Dict[str, Any]):
    """
    Replace fields of a dataclass instance obj with given kwargs.

    Pop and use only the kwargs that match a field name, leave others intact.
    """
    changes = {}
    for f in fields(obj):
        try:
            v = kwargs.pop(f.name)
        except KeyError:
            pass
        else:
            changes[f.name] = v
    return replace(obj, **changes)


def assert_empty_kwargs(kwargs):
    if kwargs:
        msg = (f"extra keyword arguments: "
               f"{', '.join(f'{k}={v!r}' for k, v in kwargs.items())}")
        raise TypeError(msg)


@dataclass
class ComputeReqParams:
    """
    EigenTrust compute request parameters.

    `alpha` and `epsilon` are core EigenTrust parameters.

    `max_iterations`, `min_iterations`, and `check_freq` are
    exit criteria parameters.
    They can be used with bipartite or otherwise heterogeneous trust graphs
    such as hubs-and-authorities affinity graphs,
    to handle convergence over fixed-cycle oscillations.

    `flat_tail` and `num_leaders` are flat-tail detection parameters,
    which can be used when ranking stability should be part of exit criteria.
    """

    alpha: Optional[float] = None
    """
    Seed trust strength, between 0 and 1 inclusive.

    Higher values skew the result toward the seed trust bias
    and away from the local trust opinions.
    Lower values skew the result toward the local trust opinions
    and away from the seed trust (thus more "democratic"),
    but tends to be slower.

    When unsure, start with 0.5, observe the result,
    then tune higher/lower as needed.
    """

    epsilon: Optional[float] = None
    """
    Convergence termination threshold, between 0 and 1 inclusive.

    Leave untouched for the most part, except when using flat-tail (advanced).
    """

    max_iterations: Optional[int] = None
    """
    The maximum number of iterations.

    Iteration loop stops after this many iterations
    even if other termination criteria are not met.

    0 (default) means no limit.
    """

    min_iterations: Optional[int] = None
    """
    The minimum number of iterations.

    The loop performs at least this many iterations
    even if other termination criteria are met.

    Defaults to check_freq, which in turn defaults to 1.
    """

    check_freq: Optional[int] = None
    """
    Exit criteria check frequency, in number of iterations.

    The loop checks exit criteria only every this many iterations.
    It can be used with `min_iterations` for "modulo n" behavior,
    e.g. with ``min_iterations=7`` and ``check_freq=5``
    exit criteria are checked after 7/12/17/... iterations.

    Default is 1: exit criteria are checked after every iteration.
    """

    flat_tail: Optional[int] = None
    """
    Flat tail length.

    This is the number of consecutive iterations with
    ranking unchanged from previous iteration
    that must be seen before terminating the recursion.

    0 (default) means a flat tail need not be seen,
    and the recursion is terminated solely based upon convergence check.
    """

    num_leaders: Optional[int] = None
    """
    The number of top-ranking for flat tail detection.

    0 (default) means everyone.
    """

    def update_req(self, req: ComputeRequestBody):
        for k, v in asdict(self).items():
            if v is not None:
                setattr(req, k, v)


DEFAULT_COMPUTE_PARAMS = ComputeReqParams()  # none yet


@dataclass
class ClientParams:
    """EigenTrust API client parameters."""

    host_url: Optional[str] = None
    """The host URL for the EigenTrust service."""

    timeout: Optional[int] = None
    """EigenTrust request timeout, in milliseconds."""

    api_key: Optional[str] = None
    """The API key for authentication."""

    s3_bucket: Optional[str] = None
    """The S3 bucket to use S3-based local trust upload."""


DEFAULT_CLIENT_PARAMS = ClientParams(
    host_url="https://openrank-sdk-api.k3l.io",
    timeout=(15 * 60 * 1000),
    s3_bucket="openrank-sdk-dev-cache",
    api_key='',
)


class _NoEdgesFoundForAddresses(RuntimeError):
    """No edges found for addresses."""

    def __str__(self):
        addresses = self.args[0]
        return f"No edges found for {addresses}"


# noinspection PyMethodMayBeStatic
class EigenTrust:
    """
    EigenTrust client.

    Can be instantiated also with various request parameters.
    These values are used when they are not specified to individual request
    method calls such as `run_eigentrust()`.
    If a parameter is neither specified here nor to the request method call,
    server picks/uses the default value.

    See `ComputeReqParams` and `ClientParams` for the list of parameters.

    Example:

    >>> et = EigenTrust(alpha=0.5, epsilon=1.0, max_iterations=50, flat_tail=2,
    ...                 host_url="https://example.com", timeout=900000,
    ...                 api_key="your_api_key")
    """

    def __init__(self, *poargs, **kwargs):
        compute_params = replace_with_kwargs(DEFAULT_COMPUTE_PARAMS, kwargs)
        client_params = replace_with_kwargs(DEFAULT_CLIENT_PARAMS, kwargs)
        super().__init__(*poargs, **kwargs)
        self.compute_params = compute_params
        self.client_params = client_params
        self.http = httpx.Client(timeout=self.client_params.timeout / 1000)

    @property
    def alpha(self):
        return self.compute_params.alpha

    @property
    def go_eigentrust_host_url(self):
        return self.client_params.host_url

    @property
    def api_key(self):
        return self.client_params.api_key

    @property
    def s3_bucket(self):
        return self.client_params.s3_bucket

    @staticmethod
    def normalize_trust(
            localtrust: List[IJV], pretrust: List[IV] = None,
    ) -> [List[Score], List[Score]]:
        return localtrust, pretrust

    def run(self,
            localtrust: trust.Matrix, pretrust: Optional[trust.Vector], *,
            scale: Union[ScoreScale, str] = ScoreScale.LOG,
            **kwargs) -> trust.Vector:
        """
        Run the EigenTrust algorithm using the provided local trust and
        pre-trust data.

        :param localtrust: local trust.
        :param pretrust: pre-trust.
        :param scale: how to scale the output scores.
            See `ScoreScale` documentation for details.
        :returns: the computed scores.

        Example:

        >>> et = EigenTrust()
        >>> lt_df = pd.DataFrame([  # uses string peer ids
        ...     dict(i='ek', j='sd', v=100),
        ...     dict(i='vm', j='sd', v=100),
        ...     dict(i='ek', j='vm', v=75),
        ... ])
        >>> pt_df = pd.DataFrame([  # uses string peer ids
        ...     dict(i='ek', v=50),
        ...     dict(i='vm', v=100),
        ...     dict(i='gw', v=100),
        ... ])
        >>> id2idx, idx2id = trust.make_peer_map()  # peer id <-> index maps
        >>> lt = trust.InlineMatrix.from_df(
        ...     lt_df, coord_map=id2idx,
        ...     on_missing='allocate',  # create peer entries in id2idx/idx2id
        ... )
        >>> dict(id2idx)  # peers that appear in local trust
        {'ek': 0, 'sd': 1, 'vm': 2}
        >>> pt = trust.InlineVector.from_df(
        ...     pt_df, coord_map=id2idx,
        ...     on_missing='drop',  # drops gw because she's not in local trust
        ... )
        >>> gt = et.run(lt, pt, scale='raw')
        >>> assert isinstance(gt, trust.InlineVector)
        >>> gt_df = gt.to_df(coord_map=idx2id)
        >>> gt_df
            i         v
        0  vm  0.480620
        1  sd  0.302326
        2  ek  0.217054
        """
        scale = ScoreScale(scale)
        scaler = _SCORE_SCALERS[scale]
        reverse = scale != ScoreScale.LOG

        logging.debug("calling go_eigentrust")
        globaltrust = self._send_go_eigentrust_req(pretrust=pretrust,
                                                   localtrust=localtrust,
                                                   **kwargs)
        if isinstance(globaltrust, trust.InlineVector):
            globaltrust.entries['v'] = [scaler(v) for v in
                                        globaltrust.entries['v']]
            globaltrust.entries = globaltrust.entries.sort_values('v',
                                                                  ascending=not reverse)
        return globaltrust

    def _prepare_input(self, localtrust: List[IJV],
                       pretrust: Optional[List[IV]]):
        lt = []
        for entry in localtrust:
            if entry['v'] <= 0.0:
                logging.warning(f"v cannot be less than or equal to 0, "
                                f"skipping this entry: {entry}")
            elif entry['i'] == entry['j']:
                logging.warning(f"i and j cannot be same, "
                                f"skipping this entry: {entry}")
            else:
                lt.append(entry)
        localtrust[:] = lt
        if pretrust is not None:
            pt = []
            for p in pretrust:
                if p['v'] <= 0.0:
                    logging.warning(f"v cannot be less than or equal to 0, "
                                    f"skipping this entry: {p}")
                else:
                    pt.append(p)
            pretrust[:] = pt

    @staticmethod
    def _compat_scale(scale: Optional[Union[ScoreScale, str]]) -> ScoreScale:
        if scale is None:
            msg = (
                "Defaulting to the 'raw' score scale. "
                "The default scale will change to 'log' in a future version; "
                "add score='raw' to keep the current behavior "
                "(and silence this warning)")
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
            scale = ScoreScale.RAW
        return scale

    def run_eigentrust(
            self, localtrust: List[IJV], pretrust: List[IV] = None, *,
            scale: Optional[Union[ScoreScale, str]] = None, **kwargs,
    ) -> List[Score]:
        """
        Run the EigenTrust algorithm using the provided local trust and
        pre-trust data.

        This method is for small in-line local trust and pre-trust data;
        for larger input data (~1M entries or more),
        the `run()` method should be used directly with `pd.DataFrame`.
        See the `run()` documentation for details.

        Example:

        >>> from openrank_sdk import EigenTrust
        >>> et = EigenTrust()
        >>> lt = [
        ...     dict(i='ek', j='sd', v=100),
        ...     dict(i='vm', j='sd', v=100),
        ...     dict(i='ek', j='vm', v=75),
        ... ]
        >>> pt = [
        ...     dict(i='ek', v=50),
        ...     dict(i='vm', v=100), dict(i='gw', v=100),
        ... ]
        >>> gt = et.run_eigentrust(lt, pt, scale='raw')
        >>> from pprint import pprint
        >>> pprint(gt)
        [{'i': 'vm', 'v': 0.4806201780180134},
         {'i': 'sd', 'v': 0.3023255429103218},
         {'i': 'ek', 'v': 0.21705427907166472}]


        Args:
            localtrust (List[IJV]): List of local trust values.
            pretrust (List[IV], optional): List of pre-trust values. Defaults
            to None.
            scale (ScoreScale): How to scale the output scores.  See
            `ScoreScale` documentation for details.

        Returns:
            List[Score]: List of computed scores.
        """
        scale = self._compat_scale(scale)

        try:
            self._prepare_input(localtrust, pretrust)
        except _NoEdgesFoundForAddresses as e:
            print(e)
            return []

        id2idx, idx2id = trust.make_peer_map()
        lt_matrix = trust.InlineMatrix.from_entries(localtrust, id2idx,
                                                    on_missing='allocate')
        if pretrust is None:
            pt_vector = None
        else:
            pt_vector = trust.InlineVector.from_entries(pretrust, id2idx,
                                                        on_missing='drop')
        globaltrust = self.run(lt_matrix, pt_vector, scale=scale, **kwargs)
        return list(globaltrust.load().to_entries(coord_map=idx2id))

    def _read_scores_from_csv(
            self, localtrust_filename: str, pretrust_filename: str = None, *,
            coord_map: trust.PeerId2Index,
            on_lt_missing: trust.OnMissingPeer,
            on_pt_missing: trust.OnMissingPeer,
    ) -> Tuple[trust.InlineMatrix, trust.InlineVector]:
        lt_df = pd.read_csv(localtrust_filename)
        lt_columns = tuple(lt_df.columns)
        pt_df = pd.read_csv(pretrust_filename)
        pt_columns = tuple(pt_df.columns)
        lt = trust.InlineMatrix.from_df(lt_df, lt_columns[-1], lt_columns[:-1],
                                        coord_map, on_missing=on_lt_missing)
        pt = trust.InlineVector.from_df(pt_df, pt_columns[-1], pt_columns[:-1],
                                        coord_map, on_missing=on_pt_missing)

        return lt, pt

    def run_eigentrust_from_csv(
            self, localtrust_filename: str, pretrust_filename: str = None,
            scale: Optional[Union[ScoreScale, str]] = None, **kwargs,
    ) -> List[Score]:
        """
        Run the EigenTrust algorithm using local trust and pre-trust data
        from CSV files.

        Args:
            localtrust_filename (str):
                The filename of the local trust CSV file.
            pretrust_filename (str, optional):
                The filename of the pre-trust CSV file. Defaults to None.
            scale (ScoreScale or str):
                How to scale the output scores.
                See `ScoreScale` documentation for details.

        Returns:
            List[Score]: List of computed scores.

        Example:
            scores = et.run_eigentrust_from_csv('localtrust.csv',
                                                'pretrust.csv')
        """
        scale = self._compat_scale(scale)
        id2idx, idx2id = trust.make_peer_map()
        lt_url = 'file:' + os.path.realpath(localtrust_filename)
        lt = trust.ObjectStorageMatrix(lt_url).load(coord_map=id2idx,
                                                    on_missing='allocate')
        pt_url = 'file:' + os.path.realpath(pretrust_filename)
        pt = trust.ObjectStorageVector(pt_url).load(coord_map=id2idx,
                                                    on_missing='drop')
        gt = self.run(lt, pt, scale=scale, **kwargs)
        return list(gt.load().to_entries(coord_map=idx2id))

    def _send_go_eigentrust_req(
            self,
            pretrust: Optional[trust.Vector],
            localtrust: trust.Matrix,
            **kwargs,
    ) -> trust.Vector:
        compute_params = replace_with_kwargs(self.compute_params, kwargs)
        client_params = replace_with_kwargs(self.client_params, kwargs)
        assert_empty_kwargs(kwargs)
        req = ComputeRequestBody(
            local_trust=localtrust,
            pre_trust=pretrust,
        )
        compute_params.update_req(req)

        start_time = time.perf_counter()
        try:
            response = self.http.post(
                f"{client_params.host_url}/basic/v1/compute",
                headers={
                    'Accept': 'application/json',
                    'API-Key': self.api_key,
                },
                json=req.encode(),
                follow_redirects=True,
            )
            logging.debug(
                f"go-eigentrust took {time.perf_counter() - start_time} secs")

            if response.status_code != 200:
                resp_data = response.text
                msg = f"Server returned HTTP {response.status_code}: {resp_data}"
                logging.error(msg)
                raise RuntimeError(msg)

            return trust.Vector.decode(response.json())
        except Exception:
            logging.error('Error while sending a request to go-eigentrust',
                          exc_info=True)
            raise

    @staticmethod
    def export_scores_to_csv(
            scores: List[Score], filepath: str, headers: List[str]):
        """
        Export the computed scores to a CSV file.

        Args:
            scores (List[Score]): List of computed scores.
            filepath (str): The path to the output CSV file.
            headers (List[str]): List of CSV headers.

        Example:
            et.export_scores_to_csv(scores, 'scores.csv', ['i', 'v'])
        """
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for line in scores:
                item = []
                for h in headers:
                    # noinspection PyTypedDict
                    item.append(line[h])
                writer.writerow(item)

    def export_csv_to_dune(
            self,
            filepath: str,
            headers: List[str],
            table_name: str,
            description: str,
            is_private: bool,
            api_key: str,
    ):
        """
        Export a CSV file to the Dune Analytics platform.

        Args:
            filepath (str): The path to the CSV file.
            headers (List[str]): List of CSV headers.
            table_name (str): The name of the table on Dune Analytics.
            description (str): Description of the table.
            is_private (bool): Whether the table is private.
            api_key (str): The API key for Dune Analytics.

        Example:
            et.export_csv_to_dune('scores.csv', ['i', 'v'], 'my_table',
                                  'Table description', False, 'your_api_key')
        """
        csv_header = ""
        for h in headers:
            csv_header += f"{h},"
        lines = [csv_header]
        with open(filepath, "r") as f:
            reader = csv.reader(f, delimiter=',')
            for _, line in enumerate(reader):
                header = ""
                for field in line:
                    header += f'{field},'
                lines.append(header)
        data = '\n'.join(lines)
        req = {
            "data": data,
            "description": description,
            "table_name": table_name,
            "is_private": is_private,
        }

        start_time = time.perf_counter()
        try:
            response = self.http.request(
                'POST',
                "https://api.dune.com/api/v1/table/upload/csv",
                headers={
                    'Accept': 'application/json',
                    'X-DUNE-API-KEY': api_key,
                },
                json=req,
                # timeout=30 * 1000,
            )
            resp_dict = response.json()

            if response.status_code != 200:
                resp_data = response.text
                msg = f"Server returned HTTP {response.status_code}: {resp_data}"
                logging.error(msg)
                raise RuntimeError(msg)

            return resp_dict
        except Exception as e:
            logging.error(
                'error while sending a request to dune-upload-csv', e)
        logging.debug(
            f"dune-upload-csv took {time.perf_counter() - start_time} secs ")

    def _save_dict_to_csv(self, data: List[dict], filename: str):
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    def run_eigentrust_from_s3(
            self, localtrust_filename: str, pretrust_filename: str = None,
            scale: Optional[Union[ScoreScale, str]] = None, **kwargs,
    ) -> List[Score]:
        scale = self._compat_scale(scale)
        id2idx, idx2id = trust.make_peer_map()
        with NamedTemporaryFile() as f:
            trust.ObjectStorageMatrix.from_path(localtrust_filename).load(
                coord_map=id2idx, on_missing='allocate',
            ).entries.to_csv(f, index=False)
            f.flush()
            f.seek(0)
            lt = trust.ObjectStorageMatrix.from_path(f.name) \
                .upload_to_s3(self.s3_bucket)
        with NamedTemporaryFile() as f:
            trust.ObjectStorageVector.from_path(pretrust_filename).load(
                coord_map=id2idx, on_missing='drop',
            ).entries.to_csv(f, index=False)
            f.flush()
            f.seek(0)
            pt = trust.ObjectStorageVector.from_path(f.name) \
                .upload_to_s3(self.s3_bucket)
        gt = self.run(lt, pt, scale=scale, **kwargs)
        return list(gt.load().to_entries(coord_map=idx2id))

    # New methods to interact with the backend server
    def _upload_csv(
            self, data: List[dict], headers: List[str], endpoint: str,
            overwrite: bool,
    ) -> str:
        """
        Upload CSV data to the backend server.

        Args:
            data (List[dict]): List of data to be uploaded.
            headers (List[str]): List of CSV headers.
            endpoint (str): The endpoint for the upload.
            overwrite (bool): Whether to overwrite existing data.

        Returns:
            str: URL of the uploaded data.

        Example:
            data = [
                {'i': 'A', 'j': 'B', 'v': 0.5},
                {'i': 'B', 'j': 'C', 'v': 0.6},
            ]
            url = et._upload_csv(
                data, ['i', 'j', 'v'], 'localtrust/123', True)
        """
        # Create an in-memory file-like object for the CSV data
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)

        # Write CSV headers
        writer.writerow(headers)

        # Write CSV rows
        for item in data:
            writer.writerow(item.values())

        # Send the CSV data to the server
        response = self.http.post(
            f'{self.go_eigentrust_host_url}/upload/{endpoint}?overwrite={overwrite}',
            headers={'Content-Type': 'text/csv'},
            content=csv_buffer.getvalue().encode('utf-8'),
        )

        if response.status_code != 200:
            raise Exception(
                f"Failed to upload CSV: {response.text}")

        return f'{self.go_eigentrust_host_url}/download/{endpoint}'

    def _download_csv(self, endpoint: str) -> List[dict]:
        """
        Download CSV data from the backend server.

        Args:
            endpoint (str): The endpoint for the download.

        Returns:
            List[dict]: List of downloaded data.

        Example:
            data = et._download_csv('localtrust/123')
        """
        response = self.http.get(
            f'{self.go_eigentrust_host_url}/download/{endpoint}',
            headers={'Accept': 'text/csv'}
        )

        if response.status_code != 200:
            raise Exception(
                f"Failed to download CSV: {response.text}")
        data = response.text.splitlines()
        reader = csv.DictReader(data)
        return list(reader)

    @staticmethod
    def _convert_to_ijv(data: List[dict]) -> List[IJV]:
        """
        Convert a list of dictionaries to a list of IJV objects.

        Args:
            data (List[dict]): List of data to be converted.

        Returns:
            List[IJV]: List of IJV objects.

        Example:
            ijv_list = et._convert_to_ijv(data)
        """
        return [{'i': row['i'], 'j': row['j'],
                 'v': float(row['v'])} for row in data]

    @staticmethod
    def _convert_to_iv(data: List[dict]) -> List[IV]:
        """
        Convert a list of dictionaries to a list of IV objects.

        Args:
            data (List[dict]): List of data to be converted.

        Returns:
            List[IV]: List of IV objects.

        Example:
            iv_list = et._convert_to_iv(data)
        """
        return [{'i': row['i'], 'v': float(row['v'])} for row in data]

    @staticmethod
    def _convert_to_score(data: List[dict]) -> List[Score]:
        """
        Convert a list of dictionaries to a list of Score objects.

        Args:
            data (List[dict]): List of data to be converted.

        Returns:
            List[Score]: List of Score objects.

        Example:
            score_list = et._convert_to_score(data)
        """
        return [{'i': row['i'], 'v': float(row['v'])} for row in data]

    def run_eigentrust_from_id(
            self, localtrust_id: str, pretrust_id: str = None,
    ) -> List[Score]:
        """
        Run the EigenTrust algorithm using local trust and pre-trust data
        identified by their IDs.

        Args:
            localtrust_id (str): The ID of the local trust data.
            pretrust_id (str, optional):
                The ID of the pre-trust data. Defaults to None.

        Returns:
            Tuple[List[Score], str]:
                List of computed scores and the URL of the results.

        Example:
            scores, url = et.run_eigentrust_from_id('localtrust123',
                                                    'pretrust123')
        """
        data = {
            'localtrust_id': localtrust_id,
            'alpha': self.alpha,
        }
        if pretrust_id:
            data['pretrust_id'] = pretrust_id

        response = self.http.post(
            f'{self.go_eigentrust_host_url}/compute_from_id',
            json=data
        )

        if response.status_code != 200:
            raise Exception(
                f"Failed to run eigentrust: {response.text}")

        resp_dict = response.json()
        scores = [Score(i=item['i'], v=item['v'])
                  for item in resp_dict['scores']]
        return scores

    def run_and_publish_eigentrust_from_id(
            self, id_: str, localtrust_id: str, pretrust_id: str = None,
            **kwargs,
    ) -> Tuple[List[Score], str]:
        """
        Run the EigenTrust algorithm using local trust and pre-trust data
        identified by their IDs, and publish the results.

        Args:
            id_ (str): The ID for publishing the results.
            localtrust_id (str): The ID of the local trust data.
            pretrust_id (str, optional):
                The ID of the pre-trust data. Defaults to None.

        Returns:
            Tuple[List[Score], str]:
                List of computed scores and the URL of the published results.

        Example:
            scores, publish_url = et.run_and_publish_eigentrust_from_id(
                'result123', 'localtrust123', 'pretrust123')
        """
        scores = self.run_eigentrust_from_id(localtrust_id, pretrust_id)
        publish_url = self.publish_eigentrust(id_, scores, **kwargs)
        return scores, publish_url

    def run_and_publish_eigentrust(
            self, id_: str, localtrust: List[IJV], pretrust: List[IV] = None,
            **kwargs,
    ) -> Tuple[List[Score], str]:
        """
        Run the EigenTrust algorithm using local trust and pre-trust data,
        and publish the results.

        Args:
            id_ (str): The ID for publishing the results.
            localtrust (List[IJV]): List of local trust values.
            pretrust (List[IV], optional):
                List of pre-trust values. Defaults to None.

        Returns:
            Tuple[List[Score], str]:
                List of computed scores and the URL of the published results.

        Example:
            localtrust = [
                {'i': 'A', 'j': 'B', 'v': 0.5},
                {'i': 'B', 'j': 'C', 'v': 0.6},
            ]
            pretrust = [{'i': 'A', 'v': 1.0}]
            scores, publish_url = et.run_and_publish_eigentrust(
                'result123', localtrust, pretrust)
        """
        overwrite = kwargs.get('overwrite', False)
        scores = self.run_eigentrust(localtrust, pretrust)
        publish_url = self.publish_eigentrust(id_, scores,
                                              overwrite=overwrite)
        return scores, publish_url

    def publish_eigentrust(
            self, id_: str, result: List[Score], **kwargs) -> str:
        """
        Publish the EigenTrust results.

        Args:
            id_ (str): The ID for publishing the results.
            result (List[Score]): List of computed scores.

        Returns:
            str: URL of the published results.

        Example:
            url = et.publish_eigentrust('result123', scores)
        """
        overwrite = kwargs.get('overwrite', False)
        return self._upload_csv(result, SCORE_CSV_HEADERS,
                                f'eigentrust/{id_}', overwrite)

    def fetch_eigentrust(self, id_: str, **_) -> List[Score]:
        """
        Fetch the EigenTrust results by ID.

        Args:
            id_ (str): The ID of the results to fetch.

        Returns:
            List[Score]: List of fetched scores.

        Example:
            scores = et.fetch_eigentrust('result123')
        """
        return self._convert_to_score(self._download_csv(f'eigentrust/{id_}'))

    def publish_localtrust(
            self, id_: str, result: List[IJV], **kwargs) -> str:
        """
        Publish the local trust data.

        Args:
            id_ (str): The ID for publishing the local trust data.
            result (List[IJV]): List of local trust values.

        Returns:
            str: URL of the published local trust data.

        Example:
            url = et.publish_localtrust('localtrust123', localtrust)
        """
        overwrite = kwargs.get('overwrite', False)
        return self._upload_csv(result, IJV_CSV_HEADERS,
                                f'localtrust/{id_}', overwrite)

    def fetch_localtrust(self, id_: str, **_) -> List[IJV]:
        """
        Fetch the local trust data by ID.

        Args:
            id_ (str): The ID of the local trust data to fetch.

        Returns:
            List[IJV]: List of fetched local trust values.

        Example:
            localtrust = et.fetch_localtrust('localtrust123')
        """
        return self._convert_to_ijv(self._download_csv(f'localtrust/{id_}'))

    def publish_pretrust(self, id_: str, result: List[IV], **kwargs) -> str:
        """
        Publish the pre-trust data.

        Args:
            id_ (str): The ID for publishing the pre-trust data.
            result (List[IV]): List of pre-trust values.

        Returns:
            str: URL of the published pre-trust data.

        Example:
            url = et.publish_pretrust('pretrust123', pretrust)
        """
        overwrite = kwargs.get('overwrite', False)
        return self._upload_csv(result, IV_CSV_HEADERS,
                                f'pretrust/{id_}', overwrite)

    def fetch_pretrust(self, id_: str, **_) -> List[IV]:
        """
        Fetch the pre-trust data by ID.

        Args:
            id_ (str): The ID of the pre-trust data to fetch.

        Returns:
            List[IV]: List of fetched pre-trust values.

        Example:
            pretrust = et.fetch_pretrust('pretrust123')
        """
        return self._convert_to_iv(self._download_csv(f'pretrust/{id_}'))
