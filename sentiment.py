import json

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from fetch_news import format_date, get_btc_headlines

OUTPUT_FILE = "sentiment.json"


def sentiment_label(compound):
    """Convert a VADER compound score to a sentiment label."""
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def main():
    analyzer = SentimentIntensityAnalyzer()

    print("Fetching Bitcoin-related headlines from CoinDesk RSS feed...")
    try:
        headlines = get_btc_headlines(limit=10)
    except RuntimeError as e:
        print(e)
        return

    if not headlines:
        print("No Bitcoin-related headlines found in the feed.")
        return

    print()
    scores = []
    results = []
    for i, entry in enumerate(headlines, start=1):
        title = entry.get("title", "Untitled")
        date_str = format_date(entry)
        vs = analyzer.polarity_scores(title)
        compound = vs["compound"]
        label = sentiment_label(compound)
        scores.append(compound)
        results.append(
            {
                "headline": title,
                "date": date_str,
                "sentiment": label,
                "compound": compound,
            }
        )
        print(f"{i}. [{label}] (compound: {compound:+.2f}) {title}")

    overall = sum(scores) / len(scores)
    print()
    print(f"Overall average sentiment score: {overall:+.4f}")

    # Save results to sentiment.json
    output = {
        "overall_average_sentiment": overall,
        "articles": results,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()