import feedparser
import time
from datetime import datetime, timezone

RSS_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"
BTC_KEYWORDS = ("bitcoin", "btc", "satoshi", "halving")


def get_btc_headlines(limit=10):
    """Fetch and return the most recent Bitcoin-related headlines from CoinDesk RSS."""
    feed = feedparser.parse(RSS_URL)

    if feed.bozo and not feed.entries:
        raise RuntimeError(f"Error parsing RSS feed: {feed.bozo_exception}")

    btc_entries = [
        entry
        for entry in feed.entries
        if any(kw in entry.get("title", "").lower() for kw in BTC_KEYWORDS)
    ]

    return btc_entries[:limit]


def format_date(entry):
    """Format an entry's published date as a clean UTC string."""
    published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if published_parsed:
        return datetime.fromtimestamp(
            time.mktime(published_parsed), tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
    return entry.get("published", entry.get("updated", ""))


def main():
    print("Fetching Bitcoin-related headlines from CoinDesk RSS feed...")
    try:
        btc_entries = get_btc_headlines(limit=10)
    except RuntimeError as e:
        print(e)
        return

    if not btc_entries:
        print("No Bitcoin-related headlines found in the feed.")
        return

    print()
    for i, entry in enumerate(btc_entries, start=1):
        title = entry.get("title", "Untitled")
        date_str = format_date(entry)
        print(f"{i}. [{date_str}] {title}")


if __name__ == "__main__":
    main()

