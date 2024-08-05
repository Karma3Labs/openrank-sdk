import csv
import enum
import io
import json
import logging
import math
import os
import random
import string
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, \
    Union

import boto3
import urllib3


class IJV(TypedDict):
    i: str
    j: str
    v: float


IJV_CSV_HEADERS = ['i', 'j', 'v']


class IV(TypedDict):
    i: str
    v: float


IV_CSV_HEADERS = ['i', 'v']


class Score(TypedDict):
    i: str
    v: float


SCORE_CSV_HEADERS = ['i', 'v']

SchemaType = Literal["inline", "objectstorage"]


class ScoreScale(enum.Enum):
    """EigenTrust score scale."""
    LEGACY = 'legacy'
    """Legacy behavior (default).

    Behaves same as `RAW` but emits a deprecation warning until the default
    changes to `LOG`."""

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

    Defined as `log(raw_value, 0.1)-1`, this represents the distance from
    the full trust: 0/1/2/3/... represents 100%/10%/1%/0.1%/... of the total
    trust circulating in the network.  Lower distance means higher trust.

    Score values in this scale tend to form a bell curve.
    """


_SCORE_SCALERS: Dict[ScoreScale, Callable[[float], float]] = {
    ScoreScale.RAW: lambda v: v,
    ScoreScale.PERCENT: lambda v: v * 100,
    ScoreScale.LOG: lambda v: math.log(v, 0.1) - 1,
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
    They can be used with bipartite or otherwise heterogenous trust graphs
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

    Defauls to check_freq, which in turn defaults to 1.
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

    def update_req(self, req: Dict[str, Any]):
        req.update((snake2camel(k), v)
                   for k, v in asdict(self).items()
                   if v is not None)


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
        self.http = urllib3.PoolManager()

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

    def run_eigentrust(
            self, localtrust: List[IJV], pretrust: List[IV] = None, *,
            scale: Union[ScoreScale, str] = ScoreScale.LEGACY, **kwargs,
    ) -> List[Score]:
        """
        Run the EigenTrust algorithm using the provided local trust and
        pre-trust data.

        Args:
            localtrust (List[IJV]): List of local trust values.
            pretrust (List[IV], optional): List of pre-trust values. Defaults
            to None.
            scale (ScoreScale): How to scale the output scores.  See
            `ScoreScale` documentation for details.

        Returns:
            List[Score]: List of computed scores.

        Example:
            localtrust = [
                {'i': 'A', 'j': 'B', 'v': 0.5},
                {'i': 'B', 'j': 'C', 'v': 0.6},
            ]
            pretrust = [{'i': 'A', 'v': 1.0}]
            scores = et.run_eigentrust(localtrust, pretrust)
        """
        scale = ScoreScale(scale)
        if scale == ScoreScale.LEGACY:
            warnings.warn(
                "Defaulting to the 'raw' score scale. "
                "The default scale will change to 'log' in a future version; "
                "add score='raw' to keep the current behavior "
                "(and silence this warning)"
            )
            scale = ScoreScale.RAW
        scaler = _SCORE_SCALERS[scale]
        reverse = scale != ScoreScale.LOG
        start_time = time.perf_counter()

        lt = []
        for entry in localtrust:
            if entry['v'] <= 0.0:
                logging.warning(
                    f"v cannot be less than or equal to 0, "
                    f"skipping this entry: {entry}")
            elif entry['i'] == entry['j']:
                logging.warning(
                    f"i and j cannot be same, skipping this entry: {entry}")
            else:
                lt.append(entry)
        localtrust = lt

        addresses = set()
        for entry in localtrust:
            addresses.add(entry["i"])
            addresses.add(entry["j"])
        if len(addresses) <= 0:
            print(f"No edges found for {addresses}")
            return []

        addr_to_int_map = {}
        int_to_addr_map = {}
        for idx, addr in enumerate(addresses):
            addr_to_int_map[addr] = idx
            int_to_addr_map[idx] = addr

        if not pretrust:
            pt_len = len(addresses)
            logging.debug(
                f"generating pretrust from localtrust with equally weighted "
                f"pretrusted value")
            pretrust = [{'i': addr_to_int_map[addr], 'v': 1 / pt_len}
                        for addr in addresses]
        else:
            pt = []
            for p in pretrust:
                if p['v'] <= 0.0:
                    logging.warning(
                        f"v cannot be less than or equal to 0, "
                        f"skipping this entry: {p}")
                elif not p['i'] in addresses:
                    logging.warning(
                        f"i entry not found in localtrust, "
                        f"skipping this entry: {p}")
                else:
                    pt.append(p)
            pretrust = pt
            pretrust = [{'i': addr_to_int_map[p['i']], 'v': p['v']}
                        for p in pretrust]

        logging.debug(f"generating localtrust with "
                      f"{len(addresses)} addresses")
        localtrust = [{'i': addr_to_int_map[entry['i']],
                       'j': addr_to_int_map[entry['j']],
                       'v': entry['v']} for entry in localtrust]
        max_id = len(addresses)

        logging.debug("calling go_eigentrust")
        i_scores = self._send_go_eigentrust_req(pretrust=pretrust,
                                                max_pt_id=max_id,
                                                localtrust=localtrust,
                                                max_lt_id=max_id,
                                                **kwargs)

        addr_scores = sorted(({'i': int_to_addr_map[i_score['i']],
                               'v': scaler(i_score['v'])}
                              for i_score in i_scores),
                             key=lambda x: x['v'], reverse=reverse)
        logging.info(f"eigentrust compute took "
                     f"{time.perf_counter() - start_time} secs")
        return addr_scores

    def _read_scores_from_csv(
            self, localtrust_filename: str, pretrust_filename: str = None,
    ) -> [List[IJV], List[IV]]:
        localtrust = []
        with open(localtrust_filename, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                i, j, v = line[0], line[1], line[2]
                # is header
                if not v.isnumeric():
                    continue
                localtrust.append({'i': str(i), 'j': str(j), 'v': float(v)})

        pretrust = None
        if pretrust_filename:
            pretrust = []
            with open(pretrust_filename, "r") as f:
                reader = csv.reader(f, delimiter=",")
                for i, line in enumerate(reader):
                    i, v = line[0], line[1]
                    # is header
                    if not v.isnumeric():
                        continue
                    pretrust.append({'i': str(i), 'v': float(v)})

        return localtrust, pretrust

    def run_eigentrust_from_csv(
            self, localtrust_filename: str, pretrust_filename: str = None,
            **kwargs,
    ) -> List[Score]:
        """
        Run the EigenTrust algorithm using local trust and pre-trust data
        from CSV files.

        Args:
            localtrust_filename (str):
                The filename of the local trust CSV file.
            pretrust_filename (str, optional):
                The filename of the pre-trust CSV file. Defaults to None.

        Returns:
            List[Score]: List of computed scores.

        Example:
            scores = et.run_eigentrust_from_csv('localtrust.csv',
                                                'pretrust.csv')
        """

        localtrust, pretrust = self._read_scores_from_csv(localtrust_filename,
                                                          pretrust_filename)
        return self.run_eigentrust(localtrust, pretrust, **kwargs)

    def _send_go_eigentrust_req(
            self,
            pretrust: list[dict],
            max_pt_id: int,
            localtrust: list[dict],
            max_lt_id: int,
            req: dict = None,
            **kwargs,
    ):
        """
        Send a request to the EigenTrust service to compute scores.

        Args:
            pretrust (list[dict]): List of pre-trust values.
            max_pt_id (int): The maximum pre-trust ID.
            localtrust (list[dict]): List of local trust values.
            max_lt_id (int): The maximum local trust ID.

        Returns:
            List[dict]: List of computed scores.

        Example:
            scores = self._send_go_eigentrust_req(pretrust, max_pt_id, localtrust, max_lt_id)
        """
        compute_params = replace_with_kwargs(self.compute_params, kwargs)
        client_params = replace_with_kwargs(self.client_params, kwargs)
        assert_empty_kwargs(kwargs)
        if req is None:
            req = {
                "pretrust": {
                    "scheme": 'inline',
                    # np.int64 doesn't serialize; cast to int
                    "size": int(max_pt_id) + 1,
                    "entries": pretrust,
                },
                "localTrust": {
                    "scheme": 'inline',
                    # np.int64 doesn't serialize; cast to int
                    "size": int(max_lt_id) + 1,
                    "entries": localtrust,
                },
            }
            compute_params.update_req(req)

        start_time = time.perf_counter()
        try:
            encoded_data = json.dumps(req).encode('utf-8')

            response = self.http.request(
                'POST',
                f"{client_params.host_url}/basic/v1/compute",
                headers={
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    'API-Key': self.api_key,
                },
                body=encoded_data,
                timeout=client_params.timeout,
            )
            logging.debug(
                f"go-eigentrust took {time.perf_counter() - start_time} secs")
            if response.status != 200:
                try:
                    resp_data = response.data.decode('UTF-8')
                except UnicodeDecodeError:
                    resp_data = response.data
                msg = f"Server returned HTTP {response.status}: {resp_data}"
                logging.error(msg)
                raise RuntimeError(msg)

            resp_dict = json.loads(response.data)
            return resp_dict["entries"]
        except Exception:
            logging.error('error while sending a request to go-eigentrust',
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
            tablename: str,
            description: str,
            is_private: bool,
            api_key: str,
    ):
        """
        Export a CSV file to the Dune Analytics platform.

        Args:
            filepath (str): The path to the CSV file.
            headers (List[str]): List of CSV headers.
            tablename (str): The name of the table on Dune Analytics.
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
            "table_name": tablename,
            "is_private": is_private,
        }

        start_time = time.perf_counter()
        try:
            encoded_data = json.dumps(req)

            response = self.http.request(
                'POST',
                "https://api.dune.com/api/v1/table/upload/csv",
                headers={
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    'X-DUNE-API-KEY': api_key,
                },
                body=encoded_data,
                # timeout=30 * 1000,
            )
            resp_dict = json.loads(response.data.decode('utf-8'))

            if response.status != 200:
                logging.error(f"Server error: {response.status}:"
                              f"{resp_dict} {resp_dict}")
                raise {
                    "statusCode": response.status,
                    "body": str(resp_dict)
                }

            return resp_dict
        except Exception as e:
            logging.error(
                'error while sending a request to dune-upload-csv', e)
        logging.debug(
            f"dune-upload-csv took {time.perf_counter() - start_time} secs ")

    def _save_dict_to_csv(self, data: list[dict], filename: str):
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    def run_eigentrust_from_s3(
            self, localtrust_filename: str, pretrust_filename: str = None,
            **kwargs,
    ) -> List[Score]:
        start_time = time.perf_counter()
        localtrust_tmp_filename = "localtrust-tmp.csv"
        pretrust_tmp_filename = "pretrust-tmp.csv"

        localtrust, pretrust = self._read_scores_from_csv(localtrust_filename,
                                                          pretrust_filename)

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
        localtrust = lt

        addresses = set()
        for entry in localtrust:
            addresses.add(entry["i"])
            addresses.add(entry["j"])
        if len(addresses) <= 0:
            print(f"No edges found for {addresses}")
            return []

        addr_to_int_map = {}
        int_to_addr_map = {}
        for idx, addr in enumerate(addresses):
            addr_to_int_map[addr] = idx
            int_to_addr_map[idx] = addr

        if not pretrust:
            pt_len = len(addresses)
            logging.debug(f"generating pretrust from localtrust "
                          f"with equally weighted pretrusted value")
            pretrust = [{'i': addr_to_int_map[addr], 'v': 1 / pt_len}
                        for addr in addresses]
        else:
            pt = []
            for p in pretrust:
                if p['v'] <= 0.0:
                    logging.warning(f"v cannot be less than or equal to 0, "
                                    f"skipping this entry: {p}")
                elif not p['i'] in addresses:
                    logging.warning(f"i entry not found in localtrust, "
                                    f"skipping this entry: {p}")
                else:
                    pt.append(p)
            pretrust = pt
            pretrust = [{'i': addr_to_int_map[p['i']], 'v': p['v']}
                        for p in pretrust]

        logging.debug(f"generating localtrust with "
                      f"{len(addresses)} addresses")
        localtrust = [{'i': addr_to_int_map[l['i']],
                       'j': addr_to_int_map[l['j']],
                       'v': l['v']}
                      for l in localtrust]
        max_id = len(addresses)

        logging.debug("calling go_eigentrust")

        self._save_dict_to_csv(localtrust, localtrust_tmp_filename)
        if pretrust_filename is not None:
            self._save_dict_to_csv(pretrust, pretrust_tmp_filename)

        localtrust_s3 = self._upload_csv_to_s3(localtrust_tmp_filename)
        if pretrust_filename is not None:
            pretrust_s3 = self._upload_csv_to_s3(pretrust_tmp_filename)

        req = {
            "localTrust": {
                "scheme": 'objectstorage',
                "format": "csv",
                "url": f"s3://{self.s3_bucket}/{localtrust_s3}",
            },
        }

        if pretrust_filename is not None:
            req["pretrust"] = {
                "scheme": 'objectstorage',
                "format": "csv",
                "url": f"s3://{self.s3_bucket}/{pretrust_s3}",
            }
        compute_params = replace_with_kwargs(self.compute_params, kwargs)
        compute_params.update_req(req)

        i_scores = self._send_go_eigentrust_req(pretrust=pretrust,
                                                max_pt_id=max_id,
                                                localtrust=localtrust,
                                                max_lt_id=max_id,
                                                req=req,
                                                **kwargs)

        addr_scores = [{'i': int_to_addr_map[i_score['i']], 'v': i_score['v']}
                       for i_score in i_scores]
        logging.info(f"eigentrust compute took "
                     f"{time.perf_counter() - start_time} secs ")
        addr_scores.sort(key=lambda x: x['v'], reverse=True)

        os.remove(localtrust_tmp_filename)
        if pretrust_filename is not None:
            os.remove(pretrust_tmp_filename)

        return addr_scores

    def _upload_csv_to_s3(self, file_name) -> str:
        if not file_name.lower().endswith('.csv'):
            logging.error("Error: The file name must end with '.csv'.")
            return

        object_name = (
                ''.join(random.choice(string.ascii_letters + string.digits)
                        for _ in range(8)) +
                '_' +
                datetime.now().strftime("%Y%m%d_%H%M%S") +
                '.csv'
        )
        s3_client = boto3.client('s3')

        try:
            s3_client.upload_file(file_name, self.s3_bucket, object_name)
            logging.info(f"File {file_name} uploaded to "
                         f"{self.s3_bucket}/{object_name} successfully.")
        except Exception as e:
            logging.error('error while sending a request to aws s3', e)

        return object_name

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
        response = self.http.request(
            'POST',
            (f'{self.go_eigentrust_host_url}/upload/{endpoint}'
             f'?overwrite={overwrite}'),
            headers={'Content-Type': 'text/csv'},
            body=csv_buffer.getvalue().encode('utf-8'),
        )

        if response.status != 200:
            raise Exception(
                f"Failed to upload CSV: {response.data.decode('utf-8')}")

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
        response = self.http.request(
            'GET',
            f'{self.go_eigentrust_host_url}/download/{endpoint}',
            headers={'Accept': 'text/csv'}
        )
        if response.status != 200:
            raise Exception(
                f"Failed to download CSV: {response.data.decode('utf-8')}")
        data = response.data.decode('utf-8').splitlines()
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
    ) -> Tuple[List[Score], str]:
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

        response = self.http.request(
            'POST',
            f'{self.go_eigentrust_host_url}/compute_from_id',
            headers={'Content-Type': 'application/json'},
            body=json.dumps(data).encode('utf-8')
        )

        if response.status != 200:
            raise Exception(
                f"Failed to run eigentrust: {response.data.decode('utf-8')}")

        resp_dict = json.loads(response.data.decode('utf-8'))
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
