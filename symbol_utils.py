from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


FX_CONTRACT_SIZE = 100_000


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    base_currency: str
    quote_currency: str
    pip_size: float
    contract_size: int = FX_CONTRACT_SIZE


def symbol_spec(symbol: str) -> SymbolSpec:
    clean = symbol.upper().strip()
    if len(clean) != 6:
        raise ValueError(f"Unsupported FX symbol '{symbol}'. Expected a 6-letter pair like EURUSD.")
    base = clean[:3]
    quote = clean[3:]
    pip_size = 0.01 if quote == "JPY" else 0.0001
    return SymbolSpec(
        symbol=clean,
        base_currency=base,
        quote_currency=quote,
        pip_size=pip_size,
    )


def pip_size_for_symbol(symbol: str) -> float:
    return symbol_spec(symbol).pip_size


def contract_size_for_symbol(symbol: str) -> int:
    return symbol_spec(symbol).contract_size


def pips_to_price(symbol: str, pips: float) -> float:
    return float(pips) * pip_size_for_symbol(symbol)


def price_to_pips(symbol: str, price_delta: float) -> float:
    return float(price_delta) / pip_size_for_symbol(symbol)


def quote_currency_per_pip(symbol: str) -> float:
    spec = symbol_spec(symbol)
    return spec.contract_size * spec.pip_size


def convert_quote_to_account(
    amount_in_quote: float,
    *,
    quote_currency: str,
    account_currency: str,
    reference_price: float,
    conversion_rates: Mapping[str, float] | None = None,
) -> float:
    if quote_currency == account_currency:
        return float(amount_in_quote)

    direct_pair = f"{quote_currency}{account_currency}"
    inverse_pair = f"{account_currency}{quote_currency}"
    if conversion_rates:
        if direct_pair in conversion_rates:
            return float(amount_in_quote) * float(conversion_rates[direct_pair])
        if inverse_pair in conversion_rates:
            rate = float(conversion_rates[inverse_pair])
            if rate <= 0:
                raise ValueError(f"Invalid inverse conversion rate for {inverse_pair}: {rate}")
            return float(amount_in_quote) / rate

    if account_currency == "USD":
        if quote_currency == "JPY":
            if reference_price <= 0:
                raise ValueError("JPY->USD conversion requires a positive reference price.")
            return float(amount_in_quote) / float(reference_price)
        if quote_currency == "USD":
            return float(amount_in_quote)

    raise ValueError(
        "Missing quote-to-account conversion rate for "
        f"{quote_currency}->{account_currency}. Provide conversion_rates explicitly."
    )


def pip_value_per_lot(
    symbol: str,
    *,
    price: float,
    account_currency: str = "USD",
    conversion_rates: Mapping[str, float] | None = None,
) -> float:
    spec = symbol_spec(symbol)
    quote_value = quote_currency_per_pip(symbol)
    return convert_quote_to_account(
        quote_value,
        quote_currency=spec.quote_currency,
        account_currency=account_currency,
        reference_price=price,
        conversion_rates=conversion_rates,
    )


def pip_value_for_volume(
    symbol: str,
    *,
    price: float,
    volume_lots: float,
    account_currency: str = "USD",
    conversion_rates: Mapping[str, float] | None = None,
) -> float:
    return pip_value_per_lot(
        symbol,
        price=price,
        account_currency=account_currency,
        conversion_rates=conversion_rates,
    ) * float(volume_lots)
