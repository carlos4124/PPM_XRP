def run():
    import mwclient
    import time

    #Connect to Wikipedia
    site = mwclient.Site("en.wikipedia.org")
    page = site.pages["XRP Ledger"]

    #Retrieve the revision history of the page
    revs = list(page.revisions())

    #Sort revisions from oldest to newest
    revs = sorted(revs, key=lambda rev: rev["timestamp"])

    #Load pre-trained sentiment analysis model
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")

     #Analyzes the sentiment of the first 250 characters of a given text
    def find_sentiment(text):
        sent = sentiment_pipeline([text[:250]])[0]
        score = sent["score"]
        if sent["label"] == "NEGATIVE":
            score *= -1
        return score

    edits = {}

    for rev in revs:
        date = time.strftime("%Y-%m-%d", rev["timestamp"])

        #If there is no revision on a certain date then create an empty sentiment for that date
        if date not in edits:
            edits[date] = dict(sentiments=list(), edit_count = 0)

        #Count number of edits for the date
        edits[date]["edit_count"] += 1

        #Get the edit comment
        comment = rev["comment"]
        #Analyze sentiment of comment
        edits[date]["sentiments"].append(find_sentiment(comment))

    from statistics import mean

    for key in edits:
        if len(edits[key]["sentiments"]) > 0:
            #Calculate the average sentiment score for the day's edits
            edits[key]["sentiment"] = mean(edits[key]["sentiments"])
            #Calculate the proportion of negative sentiment edits
            edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s < 0]) / len(edits[key]["sentiments"])
        else:
            #If no sentiment data, set values to 0
            edits[key]["sentiment"] = 0
            edits[key]["neg_sentiment"] = 0

        del edits[key]["sentiments"]

    #Convert previous data into a dataframe
    import pandas as pd

    #Convert dictionary to DataFrame with dates as index
    edits_df = pd.DataFrame.from_dict(edits, orient="index")
    edits_df.index = pd.to_datetime(edits_df.index)

    from datetime import datetime

    #Create a date range from the first known edit to today
    dates = pd.date_range(start="2007-12-19", end=datetime.today())

    #Make so the data frame includes all dates, filling missing ones with zero
    edits_df = edits_df.reindex(dates, fill_value=0)

    #Compute rolling 30-day average for edit count and sentiment
    rolling_edits = edits_df.rolling(30).mean()
    rolling_edits = rolling_edits.dropna()

    #Save results to a file
    rolling_edits.to_csv("wikipedia_edits.csv")

    print(f"File saved to: {"wikipedia_edits.csv"}")

run()
