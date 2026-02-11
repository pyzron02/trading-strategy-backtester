#!/usr/bin/env python3
"""Tests for the CustomJSONEncoder in engine/serialization.py."""
import json
import datetime

import numpy as np
import pandas as pd
import pytest

from engine.serialization import CustomJSONEncoder


class TestCustomJSONEncoder:
    """Tests for CustomJSONEncoder handling of non-standard types."""

    def _encode(self, obj):
        """Helper to JSON-encode *obj* using the custom encoder."""
        return json.loads(json.dumps(obj, cls=CustomJSONEncoder))

    def _encode_raw(self, obj):
        """Helper to get the raw JSON string without parsing back."""
        return json.dumps(obj, cls=CustomJSONEncoder)

    # -- numpy integer types ------------------------------------------------

    def test_numpy_int64(self):
        result = self._encode({"val": np.int64(42)})
        assert result["val"] == 42
        assert isinstance(result["val"], int)

    def test_numpy_int32(self):
        result = self._encode({"val": np.int32(7)})
        assert result["val"] == 7

    # -- numpy floating types -----------------------------------------------

    def test_numpy_float64(self):
        result = self._encode({"val": np.float64(3.14)})
        assert abs(result["val"] - 3.14) < 1e-10

    def test_numpy_float_nan(self):
        # The encoder returns the string "NaN" from default(), which produces
        # a raw NaN token in JSON output.  Python's json.loads parses this
        # back into float('nan'), so we verify the raw encoded string.
        raw = self._encode_raw({"val": np.float64("nan")})
        assert "NaN" in raw

    def test_numpy_float_inf(self):
        raw = self._encode_raw({"val": np.float64("inf")})
        assert "Infinity" in raw

    def test_numpy_float_neg_inf(self):
        raw = self._encode_raw({"val": np.float64("-inf")})
        assert "-Infinity" in raw

    # -- numpy arrays -------------------------------------------------------

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = self._encode({"val": arr})
        assert result["val"] == [1, 2, 3]

    def test_numpy_array_2d(self):
        arr = np.array([[1, 2], [3, 4]])
        result = self._encode({"val": arr})
        assert result["val"] == [[1, 2], [3, 4]]

    # -- datetime / pandas timestamps ---------------------------------------

    def test_datetime(self):
        dt = datetime.datetime(2023, 6, 15, 12, 30, 0)
        result = self._encode({"val": dt})
        assert result["val"] == "2023-06-15T12:30:00"

    def test_date(self):
        d = datetime.date(2023, 6, 15)
        result = self._encode({"val": d})
        assert result["val"] == "2023-06-15"

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2023-06-15 12:30:00")
        result = self._encode({"val": ts})
        assert "2023-06-15" in result["val"]

    # -- pandas Series / DataFrame ------------------------------------------

    def test_pandas_series(self):
        s = pd.Series({"a": 1, "b": 2})
        result = self._encode({"val": s})
        assert result["val"]["a"] == 1
        assert result["val"]["b"] == 2

    def test_pandas_dataframe(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = self._encode({"val": df})
        assert isinstance(result["val"], list)
        assert len(result["val"]) == 2
        assert result["val"][0]["x"] == 1

    # -- pandas NA / None ---------------------------------------------------

    def test_pandas_na(self):
        result = self._encode({"val": pd.NA})
        assert result["val"] is None

    # -- standard types pass through ----------------------------------------

    def test_plain_dict(self):
        result = self._encode({"a": 1, "b": "hello"})
        assert result == {"a": 1, "b": "hello"}

    def test_plain_list(self):
        result = self._encode([1, 2, 3])
        assert result == [1, 2, 3]

    # -- unsupported type raises TypeError ----------------------------------

    def test_unsupported_type(self):
        class Foo:
            pass

        with pytest.raises(TypeError):
            json.dumps(Foo(), cls=CustomJSONEncoder)
