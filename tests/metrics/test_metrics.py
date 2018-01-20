import numpy as np
import pandas as pd

from zipline import api
from zipline.assets.synthetic import make_commodity_future_info
from zipline.testing import parameter_space
from zipline.testing.fixtures import (
    WithMakeAlgo,
    WithConstantEquityMinuteBarData,
    WithConstantFutureMinuteBarData,
    ZiplineTestCase,
)
from zipline.testing.predicates import assert_equal, wildcard


def T(cs):
    return pd.Timestamp(cs, tz='utc')


class TestConstantPrice(WithConstantEquityMinuteBarData,
                        WithConstantFutureMinuteBarData,
                        WithMakeAlgo,
                        ZiplineTestCase):
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE = True
    FUTURE_DAILY_BAR_SOURCE_FROM_MINUTE = True

    ASSET_FINDER_EQUITY_SIDS = [ord('A')]

    EQUITY_MINUTE_CONSTANT_LOW = 1.0
    EQUITY_MINUTE_CONSTANT_OPEN = 1.0
    EQUITY_MINUTE_CONSTANT_CLOSE = 1.0
    EQUITY_MINUTE_CONSTANT_HIGH = 1.0
    EQUITY_MINUTE_CONSTANT_VOLUME = 100.0

    FUTURE_MINUTE_CONSTANT_LOW = 1.0
    FUTURE_MINUTE_CONSTANT_OPEN = 1.0
    FUTURE_MINUTE_CONSTANT_CLOSE = 1.0
    FUTURE_MINUTE_CONSTANT_HIGH = 1.0
    FUTURE_MINUTE_CONSTANT_VOLUME = 100.0

    START_DATE = T('2014-01-06')
    END_DATE = T('2014-01-10')

    # note: class attributes after this do not configure fixtures, they are
    # just used in this test suite

    # we use a contract multiplier to make sure we are correctly calculating
    # exposure as price * multiplier
    FUTURE_CONTRACT_MULTIPLIER = 2

    # this is the expected exposure for a position of one contract
    FUTURE_CONSTANT_EXPOSURE = (
        FUTURE_MINUTE_CONSTANT_CLOSE * FUTURE_CONTRACT_MULTIPLIER
    )

    @classmethod
    def make_futures_info(cls):
        return make_commodity_future_info(
            first_sid=ord('Z'),
            root_symbols=['Z'],
            years=[cls.START_DATE.year],
            multiplier=cls.FUTURE_CONTRACT_MULTIPLIER,
        )

    @classmethod
    def init_class_fixtures(cls):
        super(TestConstantPrice, cls).init_class_fixtures()

        cls.equity = cls.asset_finder.retrieve_asset(
            cls.asset_finder.equities_sids[0],
        )
        cls.future = cls.asset_finder.retrieve_asset(
            cls.asset_finder.futures_sids[0],
        )

        cls.closes = pd.Index(
            cls.trading_calendar.session_closes_in_range(
                cls.START_DATE,
                cls.END_DATE,
            ),
        )
        cls.closes.name = None

    @parameter_space(direction=['long', 'short'])
    def test_equity_single_position(self, direction):
        if direction not in ('long', 'short'):
            raise ValueError(
                'direction must be either long or short, got: %r' % direction,
            )

        shares = 1 if direction == 'long' else -1

        def initialize(context):
            api.set_benchmark(self.equity)

            api.set_slippage(api.slippage.NoSlippage())
            api.set_commission(api.commission.NoCommission())

            context.ordered = False

        def handle_data(context, data):
            if not context.ordered:
                api.order(self.equity, shares)
                context.ordered = True

        perf = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
        )

        zeros = pd.Series(0.0, index=self.closes)
        all_zero_fields = [
            'algorithm_period_return',
            'benchmark_period_return',
            'benchmark_returns',
            'downside_risk',
            'excess_return',
            'max_drawdown',
            'treasury_period_return',
        ]
        if direction == 'long':
            all_zero_fields.extend((
                'short_value',
                'shorts_count',
            ))
        else:
            all_zero_fields.extend((
                'long_value',
                'longs_count',
            ))
        for field in all_zero_fields:
            assert_equal(
                perf[field],
                zeros,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        ones = pd.Series(1, index=self.closes)
        if direction == 'long':
            count_field = 'longs_count'
        else:
            count_field = 'shorts_count'

        assert_equal(
            perf[count_field],
            ones,
            check_names=False,
            msg=field,
        )

        if direction == 'long':
            expected_exposure = pd.Series(
                self.EQUITY_MINUTE_CONSTANT_CLOSE,
                index=self.closes,
            )
            for field in 'long_value', 'long_exposure':
                assert_equal(
                    perf[field],
                    expected_exposure,
                    check_names=False,
                )
        else:
            expected_exposure = pd.Series(
                -self.EQUITY_MINUTE_CONSTANT_CLOSE,
                index=self.closes,
            )
            for field in 'short_value', 'short_exposure':
                assert_equal(
                    perf[field],
                    expected_exposure,
                    check_names=False,
                )

        nan_then_zero = pd.Series(0.0, index=self.closes)
        nan_then_zero[0] = float('nan')
        nan_then_zero_fields = (
            'algo_volatility',
            'algorithm_volatility',
            'benchmark_volatility',
        )
        for field in nan_then_zero_fields:
            assert_equal(
                perf[field],
                nan_then_zero,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        capital_base_series = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.closes,
        )

        # with no commissions, slippage, or returns our portfolio value stays
        # constant (at the capital base)
        assert_equal(
            perf['portfolio_value'],
            capital_base_series,
            check_names=False,
        )

        # leverage is gross market exposure / current notional capital
        # gross market exposure is
        # sum(long_exposure) + sum(abs(short_exposure))
        # current notional capital is the current portfolio value
        expected_max_leverage = (
            # we are exposed to only one share, the portfolio value is the
            # capital_base because we have no commissions, slippage, or
            # returns
            self.EQUITY_MINUTE_CONSTANT_CLOSE / capital_base_series
        )
        assert_equal(
            perf['max_leverage'],
            expected_max_leverage,
            check_names=False,
        )

        expected_cash = capital_base_series.copy()
        if direction == 'long':
            # we purchased one share on the first day
            cash_modifier = -self.EQUITY_MINUTE_CONSTANT_CLOSE
        else:
            # we sold one share on the first day
            cash_modifier = +self.EQUITY_MINUTE_CONSTANT_CLOSE

        expected_cash[1:] += cash_modifier

        assert_equal(
            perf['starting_cash'],
            expected_cash,
            check_names=False,
        )

        expected_cash[0] += cash_modifier
        assert_equal(
            perf['ending_cash'],
            expected_cash,
            check_names=False,
        )

        # we purchased one share on the first day
        expected_capital_used = pd.Series(0.0, index=self.closes)
        expected_capital_used[0] += cash_modifier

        assert_equal(
            perf['capital_used'],
            expected_capital_used,
            check_names=False,
        )

        # we hold one share so our positions exposure is that one share's price
        expected_position_exposure = pd.Series(
            -cash_modifier,
            index=self.closes,
        )
        for field in 'ending_value', 'ending_exposure':
            # for equities, position value and position exposure are the same
            assert_equal(
                perf[field],
                expected_position_exposure,
                check_names=False,
                msg=field,
            )

        # we don't start with any positions; the first day has no starting
        # exposure
        expected_position_exposure[0] = 0
        for field in 'starting_value', 'starting_exposure':
            # for equities, position value and position exposure are the same
            assert_equal(
                perf[field],
                expected_position_exposure,
                check_names=False,
                msg=field,
            )

        assert_equal(
            perf['trading_days'],
            pd.Series(
                np.arange(len(self.closes)) + 1,
                index=self.closes,
            ),
            check_names=False,
        )

        all_nan = pd.Series(np.nan, index=self.closes)
        all_nan_fields = (
            'alpha',
            'beta',
            'sortino',
        )
        for field in all_nan_fields:
            assert_equal(
                perf[field],
                all_nan,
                check_names=False,
                msg=field,
            )

        orders = perf['orders']

        # we only order on the first day
        expected_orders = [
            [{
                'amount': shares,
                'commission': 0.0,
                'created': T('2014-01-06 14:31'),
                'dt': T('2014-01-06 14:32'),
                'filled': shares,
                'id': wildcard,
                'limit': None,
                'limit_reached': False,
                'reason': None,
                'sid': self.equity,
                'status': 1,
                'stop': None,
                'stop_reached': False
            }],
        ] + [[]] * (len(self.closes) - 1)

        assert_equal(
            orders.tolist(),
            expected_orders,
            check_names=False,
        )
        assert_equal(
            orders.index,
            self.closes,
            check_names=False,
        )

        transactions = perf['transactions']

        # since we only order on the first day, we should only transact on the
        # first day
        expected_transactions = [
            [{
                'amount': shares,
                'commission': None,
                'dt': T('2014-01-06 14:32'),
                'order_id': wildcard,
                'price': 1.0,
                'sid': self.equity,
            }],
        ] + [[]] * (len(self.closes) - 1)

        assert_equal(
            transactions.tolist(),
            expected_transactions,
            check_names=False,
        )
        assert_equal(
            transactions.index,
            self.closes,
            check_names=False,
        )

    @parameter_space(direction=['long', 'short'])
    def test_future_single_position(self, direction):
        if direction not in ('long', 'short'):
            raise ValueError(
                'direction must be either long or short, got: %r' % direction,
            )

        if direction == 'long':
            contracts = 1
            expected_exposure = self.FUTURE_CONSTANT_EXPOSURE
        else:
            contracts = -1
            expected_exposure = -self.FUTURE_CONSTANT_EXPOSURE

        def initialize(context):
            # still set the equity as the benchmark
            api.set_benchmark(self.equity)

            api.set_slippage(us_futures=api.slippage.NoSlippage())
            api.set_commission(us_futures=api.commission.NoCommission())

            context.ordered = False

        def handle_data(context, data):
            if not context.ordered:
                api.order(self.future, contracts)
                context.ordered = True

        perf = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
        )

        zeros = pd.Series(0.0, index=self.closes)
        all_zero_fields = [
            'algorithm_period_return',
            'benchmark_period_return',
            'benchmark_returns',
            'downside_risk',
            'excess_return',
            'max_drawdown',
            'treasury_period_return',

            # futures contracts have no value, just exposure
            'starting_value',
            'ending_value',
            'long_value',
            'short_value',
        ]
        if direction == 'long':
            all_zero_fields.extend((
                'short_value',
                'shorts_count',
            ))
        else:
            all_zero_fields.extend((
                'long_value',
                'longs_count',
            ))

        for field in all_zero_fields:
            assert_equal(
                perf[field],
                zeros,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        ones = pd.Series(1, index=self.closes)
        count_field = direction + 's_count'
        assert_equal(
            perf[count_field],
            ones,
            check_names=False,
            msg=count_field,
        )

        expected_exposure_series = pd.Series(
            expected_exposure,
            index=self.closes,
        )
        exposure_field = direction + '_exposure'
        assert_equal(
            perf[exposure_field],
            expected_exposure_series,
            check_names=False,
            msg=exposure_field,
        )

        nan_then_zero = pd.Series(0.0, index=self.closes)
        nan_then_zero[0] = float('nan')
        nan_then_zero_fields = (
            'algo_volatility',
            'algorithm_volatility',
            'benchmark_volatility',
        )
        for field in nan_then_zero_fields:
            assert_equal(
                perf[field],
                nan_then_zero,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        # with no commissions, entering or exiting a future position does not
        # affect your cash
        capital_base_series = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.closes,
        )

        # leverage is gross market exposure / current notional capital
        # gross market exposure is
        # sum(long_exposure) + sum(abs(short_exposure))
        # current notional capital is the current portfolio value
        expected_max_leverage = (
            self.FUTURE_CONSTANT_EXPOSURE / capital_base_series
        )
        assert_equal(
            perf['max_leverage'],
            expected_max_leverage,
            check_names=False,
        )

        # with no commissions, slippage, or returns our portfolio value stays
        # constant (at the capital base)
        for field in 'starting_cash', 'ending_cash', 'portfolio_value':
            assert_equal(
                perf[field],
                capital_base_series,
                check_names=False,
                msg=field,
            )

        # with no commissions, entering or exiting a future position does not
        # affect your cash; thus no capital gets used
        expected_capital_used = pd.Series(0.0, index=self.closes)

        assert_equal(
            perf['capital_used'],
            expected_capital_used,
            check_names=False,
        )

        # we hold one contract so our positions exposure is that one
        # contract's price
        expected_position_exposure = pd.Series(
            expected_exposure,
            index=self.closes,
        )
        assert_equal(
            perf['ending_exposure'],
            expected_position_exposure,
            check_names=False,
            check_dtype=False,
        )

        # we don't start with any positions; the first day has no starting
        # exposure
        expected_position_exposure[0] = 0
        assert_equal(
            perf['starting_exposure'],
            expected_position_exposure,
            check_names=False,
        )

        assert_equal(
            perf['trading_days'],
            pd.Series(
                np.arange(len(self.closes)) + 1,
                index=self.closes,
            ),
            check_names=False,
        )

        all_nan = pd.Series(np.nan, index=self.closes)
        all_nan_fields = (
            'alpha',
            'beta',
            'sortino',
        )
        for field in all_nan_fields:
            assert_equal(
                perf[field],
                all_nan,
                check_names=False,
                msg=field,
            )

        orders = perf['orders']

        # we only order on the first day
        expected_orders = [
            [{
                'amount': contracts,
                'commission': 0.0,
                'created': T('2014-01-06 14:31'),
                'dt': T('2014-01-06 14:32'),
                'filled': contracts,
                'id': wildcard,
                'limit': None,
                'limit_reached': False,
                'reason': None,
                'sid': self.future,
                'status': 1,
                'stop': None,
                'stop_reached': False
            }],
        ] + [[]] * (len(self.closes) - 1)

        assert_equal(
            orders.tolist(),
            expected_orders,
            check_names=False,
        )
        assert_equal(
            orders.index,
            self.closes,
            check_names=False,
        )

        transactions = perf['transactions']

        # since we only order on the first day, we should only transact on the
        # first day
        expected_transactions = [
            [{
                'amount': contracts,
                'commission': None,
                'dt': T('2014-01-06 14:32'),
                'order_id': wildcard,
                'price': 1.0,
                'sid': self.future,
            }],
        ] + [[]] * (len(self.closes) - 1)

        assert_equal(
            transactions.tolist(),
            expected_transactions,
            check_names=False,
        )
        assert_equal(
            transactions.index,
            self.closes,
            check_names=False,
        )
