import pytest

from elliptic_toolkit.log_parser import _parse_value, _parse_search_cv_log_lines

class TestValueParsing:
    def test_boolean_parsing(self):
        assert _parse_value('True') is True
        assert _parse_value('False') is False

    def test_none_parsing(self):
        assert _parse_value('None') is None

    def test_integer_parsing(self):
        assert _parse_value('42') == 42
        assert _parse_value('-7') == -7
        assert _parse_value('0') == 0

    def test_float_parsing(self):
        assert _parse_value('3.14') == 3.14
        assert _parse_value('-0.001') == -0.001
        assert _parse_value('2e10') == 2e10
        assert _parse_value('-1.5E-3') == -1.5e-3

    def test_time_parsing(self):
        assert _parse_value('10s') == 10
        assert _parse_value('5min') == 300
        assert _parse_value('5m') == 300
        assert _parse_value('2h') == 7200
        assert _parse_value('-3.5h') == -12600

    def test_string_parsing(self):
        assert _parse_value('hello') == 'hello'
        assert _parse_value('123abc') == '123abc'
        assert _parse_value('3.14abc') == '3.14abc'
        assert _parse_value('10seconds') == '10seconds'

class TestLogParser:
    _cv_line = "[CV 1/3] END class_weights=balanced, learning_rate=0.01, max_depth=5, n_estimators=100,; acc=0.95, ap=0.5 total time=  12.3s"

    def test_line(self):
        res = _parse_search_cv_log_lines([self._cv_line], trim=False)
        assert res.shape == (1, 8)
        assert res['cv_fold'][0] == 1
        assert res['class_weights'][0] == 'balanced'
        assert res['learning_rate'][0] == 0.01
        assert res['max_depth'][0] == 5
        assert res['n_estimators'][0] == 100
        assert res['acc'][0] == 0.95
        assert res['ap'][0] == 0.5
        assert res['time'][0] == 12.3

if __name__ == "__main__":
    pytest.main([__file__])