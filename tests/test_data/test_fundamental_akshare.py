"""Tests for AkShareFundamentalProvider — mocked AkShare calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pangu.data.fundamental.akshare import AkShareFundamentalProvider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_financial_df() -> pd.DataFrame:
    """Simulate ak.stock_financial_analysis_indicator() return."""
    return pd.DataFrame([
        {
            "日期": "2025-06-30",
            "摊薄每股收益(元)": "1.20",
            "每股净资产_调整前(元)": "8.50",
            "净资产收益率(%)": "14.12",
            "主营业务收入增长率(%)": "18.50",
            "净利润增长率(%)": "22.30",
            "主营业务利润(元)": "30000000000",
        },
        {
            "日期": "2025-09-30",
            "摊薄每股收益(元)": "1.80",
            "每股净资产_调整前(元)": "9.20",
            "净资产收益率(%)": "19.57",
            "主营业务收入增长率(%)": "20.10",
            "净利润增长率(%)": "25.60",
            "主营业务利润(元)": "45000000000",
        },
    ])


# ---------------------------------------------------------------------------
# get_financial_indicator
# ---------------------------------------------------------------------------


class TestGetFinancialIndicator:
    @patch("akshare.stock_financial_analysis_indicator")
    def test_returns_standardized_df(self, mock_fin: MagicMock) -> None:
        mock_fin.return_value = _fake_financial_df()
        provider = AkShareFundamentalProvider(request_interval=0)
        df = provider.get_financial_indicator("601899")

        assert len(df) == 2
        assert "roe_ttm" in df.columns
        assert "revenue_yoy" in df.columns
        assert "profit_yoy" in df.columns
        assert "symbol" in df.columns
        assert "date" in df.columns

    @patch("akshare.stock_financial_analysis_indicator")
    def test_roe_conversion(self, mock_fin: MagicMock) -> None:
        mock_fin.return_value = _fake_financial_df()
        provider = AkShareFundamentalProvider(request_interval=0)
        df = provider.get_financial_indicator("601899")

        # ROE should be decimal (19.57% → 0.1957)
        latest = df.iloc[-1]
        assert latest["roe_ttm"] == pytest.approx(0.1957, rel=0.01)
        assert latest["revenue_yoy"] == pytest.approx(0.201, rel=0.01)
        assert latest["profit_yoy"] == pytest.approx(0.256, rel=0.01)

    @patch("akshare.stock_financial_analysis_indicator")
    def test_empty_returns_empty_df(self, mock_fin: MagicMock) -> None:
        mock_fin.return_value = pd.DataFrame()
        provider = AkShareFundamentalProvider(request_interval=0)
        df = provider.get_financial_indicator("601899")
        assert df.empty

    @patch("akshare.stock_financial_analysis_indicator")
    def test_exception_returns_empty_df(self, mock_fin: MagicMock) -> None:
        mock_fin.side_effect = ConnectionError("down")
        provider = AkShareFundamentalProvider(request_interval=0)
        from pangu.utils import CircuitBreaker
        provider._circuit = CircuitBreaker(threshold=100, cooldown=0)
        df = provider.get_financial_indicator("601899")
        assert df.empty

    @patch("akshare.stock_financial_analysis_indicator")
    def test_handles_nan_fields(self, mock_fin: MagicMock) -> None:
        """Missing/empty financial fields should become None."""
        df_raw = pd.DataFrame([{
            "日期": "2025-09-30",
            "摊薄每股收益(元)": "",
            "每股净资产_调整前(元)": "",
            "净资产收益率(%)": "",
            "主营业务收入增长率(%)": "",
            "净利润增长率(%)": "",
            "主营业务利润(元)": "",
        }])
        mock_fin.return_value = df_raw
        provider = AkShareFundamentalProvider(request_interval=0)
        df = provider.get_financial_indicator("601899")

        assert len(df) == 1
        assert df.iloc[0]["roe_ttm"] is None
        assert df.iloc[0]["revenue_yoy"] is None
        assert df.iloc[0]["profit_yoy"] is None

    @patch("akshare.stock_financial_analysis_indicator")
    def test_zero_values_preserved(self, mock_fin: MagicMock) -> None:
        """Numeric 0 should be preserved as 0.0, not converted to None."""
        df_raw = pd.DataFrame([{
            "日期": "2025-09-30",
            "摊薄每股收益(元)": "0",
            "每股净资产_调整前(元)": "0",
            "净资产收益率(%)": "0",
            "主营业务收入增长率(%)": "0",
            "净利润增长率(%)": "0",
            "主营业务利润(元)": "0",
        }])
        mock_fin.return_value = df_raw
        provider = AkShareFundamentalProvider(request_interval=0)
        df = provider.get_financial_indicator("601899")

        assert df.iloc[0]["roe_ttm"] == 0.0
        assert df.iloc[0]["revenue_yoy"] == 0.0
        assert df.iloc[0]["profit_yoy"] == 0.0

    @patch("akshare.stock_financial_analysis_indicator")
    def test_nan_values_become_none(self, mock_fin: MagicMock) -> None:
        """NaN values from AkShare should become None, not propagate."""
        df_raw = pd.DataFrame([{
            "日期": "2025-09-30",
            "摊薄每股收益(元)": float("nan"),
            "每股净资产_调整前(元)": float("nan"),
            "净资产收益率(%)": float("nan"),
            "主营业务收入增长率(%)": float("nan"),
            "净利润增长率(%)": float("nan"),
            "主营业务利润(元)": float("nan"),
        }])
        mock_fin.return_value = df_raw
        provider = AkShareFundamentalProvider(request_interval=0)

        df = provider.get_financial_indicator("601899")
        assert df.iloc[0]["roe_ttm"] is None


# ---------------------------------------------------------------------------
# fetch_gross_margin_batch
# ---------------------------------------------------------------------------


class TestFetchGrossMarginBatch:
    @patch("akshare.stock_yjbb_em")
    def test_returns_symbol_to_margin_mapping(self, mock_yjbb: MagicMock) -> None:
        mock_yjbb.return_value = pd.DataFrame([
            {"股票代码": "600519", "股票简称": "贵州茅台", "销售毛利率": 91.87},
            {"股票代码": "000001", "股票简称": "平安银行", "销售毛利率": 29.77},
        ])
        provider = AkShareFundamentalProvider(request_interval=0)
        result = provider.fetch_gross_margin_batch("20240331")
        assert result == pytest.approx({"600519": 0.9187, "000001": 0.2977})

    @patch("akshare.stock_yjbb_em")
    def test_skips_nan_values(self, mock_yjbb: MagicMock) -> None:
        mock_yjbb.return_value = pd.DataFrame([
            {"股票代码": "600519", "销售毛利率": 91.87},
            {"股票代码": "000001", "销售毛利率": float("nan")},
            {"股票代码": "300750", "销售毛利率": None},
        ])
        provider = AkShareFundamentalProvider(request_interval=0)
        result = provider.fetch_gross_margin_batch("20240331")
        assert list(result.keys()) == ["600519"]

    @patch("akshare.stock_yjbb_em")
    def test_empty_response(self, mock_yjbb: MagicMock) -> None:
        mock_yjbb.return_value = pd.DataFrame()
        provider = AkShareFundamentalProvider(request_interval=0)
        assert provider.fetch_gross_margin_batch("20240331") == {}

    @patch("akshare.stock_yjbb_em")
    def test_api_failure_raises(self, mock_yjbb: MagicMock) -> None:
        mock_yjbb.side_effect = ConnectionError("down")
        provider = AkShareFundamentalProvider(request_interval=0)
        from pangu.utils import CircuitBreaker
        provider._circuit = CircuitBreaker(threshold=100, cooldown=0)
        with pytest.raises(ConnectionError):
            provider.fetch_gross_margin_batch("20240331")
