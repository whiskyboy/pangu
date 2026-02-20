"""Tests for AkShareFundamentalProvider — mocked AkShare calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_agent.data.fundamental_akshare import AkShareFundamentalProvider
from trading_agent.data.storage import Database

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db() -> Database:
    d = Database(":memory:")
    d.init_tables()
    return d


def _fake_individual_info_df() -> pd.DataFrame:
    """Simulate ak.stock_individual_info_em() return."""
    return pd.DataFrame({
        "item": ["最新", "股票代码", "股票简称", "总股本", "流通股", "总市值", "流通市值", "行业", "上市时间"],
        "value": [50.0, "601899", "紫金矿业", 2644116028.0, 2644116028.0,
                  1.32e11, 1.32e11, "黄金", "20030414"],
    })


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
# get_valuation
# ---------------------------------------------------------------------------


class TestGetValuation:
    @patch("akshare.stock_financial_analysis_indicator")
    @patch("akshare.stock_individual_info_em")
    def test_returns_valuation_dict(
        self, mock_info: MagicMock, mock_fin: MagicMock,
    ) -> None:
        mock_info.return_value = _fake_individual_info_df()
        mock_fin.return_value = _fake_financial_df()
        provider = AkShareFundamentalProvider(request_interval=0)
        val = provider.get_valuation("601899")

        assert val["market_cap"] == 1.32e11
        assert val["pe_ttm"] is not None
        assert val["pe_ttm"] > 0
        assert val["pb"] is not None
        assert val["pb"] > 0

    @patch("akshare.stock_financial_analysis_indicator")
    @patch("akshare.stock_individual_info_em")
    def test_pe_calculation(
        self, mock_info: MagicMock, mock_fin: MagicMock,
    ) -> None:
        mock_info.return_value = _fake_individual_info_df()
        mock_fin.return_value = _fake_financial_df()
        provider = AkShareFundamentalProvider(request_interval=0)
        val = provider.get_valuation("601899")

        # PE = market_cap / (EPS * total_shares) = 1.32e11 / (1.80 * 2644116028)
        expected_pe = 1.32e11 / (1.80 * 2644116028.0)
        assert val["pe_ttm"] == pytest.approx(expected_pe, rel=0.01)

    @patch("akshare.stock_financial_analysis_indicator")
    @patch("akshare.stock_individual_info_em")
    def test_pb_calculation(
        self, mock_info: MagicMock, mock_fin: MagicMock,
    ) -> None:
        mock_info.return_value = _fake_individual_info_df()
        mock_fin.return_value = _fake_financial_df()
        provider = AkShareFundamentalProvider(request_interval=0)
        val = provider.get_valuation("601899")

        # PB = market_cap / (BPS * total_shares) = 1.32e11 / (9.20 * 2644116028)
        expected_pb = 1.32e11 / (9.20 * 2644116028.0)
        assert val["pb"] == pytest.approx(expected_pb, rel=0.01)

    @patch("akshare.stock_financial_analysis_indicator")
    @patch("akshare.stock_individual_info_em")
    def test_empty_info_returns_partial(
        self, mock_info: MagicMock, mock_fin: MagicMock,
    ) -> None:
        mock_info.return_value = pd.DataFrame()
        mock_fin.return_value = _fake_financial_df()
        provider = AkShareFundamentalProvider(request_interval=0)
        val = provider.get_valuation("601899")

        assert val["market_cap"] is None
        assert val["pe_ttm"] is None

    @patch("akshare.stock_financial_analysis_indicator")
    @patch("akshare.stock_individual_info_em")
    def test_empty_fin_returns_market_cap_only(
        self, mock_info: MagicMock, mock_fin: MagicMock,
    ) -> None:
        mock_info.return_value = _fake_individual_info_df()
        mock_fin.return_value = pd.DataFrame()
        provider = AkShareFundamentalProvider(request_interval=0)
        val = provider.get_valuation("601899")

        assert val["market_cap"] == 1.32e11
        assert val["pe_ttm"] is None
        assert val["pb"] is None

    @patch("akshare.stock_financial_analysis_indicator")
    @patch("akshare.stock_individual_info_em")
    def test_exception_returns_none_values(
        self, mock_info: MagicMock, mock_fin: MagicMock,
    ) -> None:
        mock_info.side_effect = ConnectionError("down")
        provider = AkShareFundamentalProvider(request_interval=0)
        from trading_agent.utils import CircuitBreaker
        provider._circuit = CircuitBreaker(threshold=100, cooldown=0)
        val = provider.get_valuation("601899")

        assert val["pe_ttm"] is None
        assert val["market_cap"] is None

    @patch("akshare.stock_financial_analysis_indicator")
    @patch("akshare.stock_individual_info_em")
    def test_persists_to_storage(
        self, mock_info: MagicMock, mock_fin: MagicMock, db: Database,
    ) -> None:
        mock_info.return_value = _fake_individual_info_df()
        mock_fin.return_value = _fake_financial_df()
        provider = AkShareFundamentalProvider(storage=db, request_interval=0)
        provider.get_valuation("601899")

        stored = db.load_fundamentals("601899", "2020-01-01", "2030-01-01")
        assert len(stored) >= 1


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
        from trading_agent.utils import CircuitBreaker
        provider._circuit = CircuitBreaker(threshold=100, cooldown=0)
        df = provider.get_financial_indicator("601899")
        assert df.empty

    @patch("akshare.stock_financial_analysis_indicator")
    def test_persists_to_storage(
        self, mock_fin: MagicMock, db: Database,
    ) -> None:
        mock_fin.return_value = _fake_financial_df()
        provider = AkShareFundamentalProvider(storage=db, request_interval=0)
        provider.get_financial_indicator("601899")

        stored = db.load_fundamentals("601899", "2020-01-01", "2030-01-01")
        assert len(stored) == 2

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
    @patch("akshare.stock_individual_info_em")
    def test_nan_values_become_none(
        self, mock_info: MagicMock, mock_fin: MagicMock,
    ) -> None:
        """NaN values from AkShare should become None, not propagate."""
        mock_info.return_value = _fake_individual_info_df()
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

        val = provider.get_valuation("601899")
        assert val["pe_ttm"] is None
        assert val["pb"] is None

        df = provider.get_financial_indicator("601899")
        assert df.iloc[0]["roe_ttm"] is None
