def run():
    import yfinance as yf
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')

    #Fetch historical price data for XRP/USD from Yahoo Finance
    xrp_ticker = yf.Ticker("XRP-USD")
    xrp = xrp_ticker.history(period="max")

    #Ensure datetime index is timezone-naive for consistency
    xrp.index = pd.to_datetime(xrp.index)
    xrp.index = xrp.index.tz_localize(None)

    #Remove unnecessary columns
    del xrp["Dividends"]
    del xrp["Stock Splits"]

    #Change name of colums to lowercase to maintain a uniform format
    xrp.columns = [c.lower() for c in xrp.columns]

    #Plot the closing price of XRP over time and save the graph
    xrp.plot.line(y="close", use_index=True)
    plt.savefig("graph.png")

    #Load Wikipedia edits data from the previously generated CSV file
    wiki = pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates = True)

    #Merge Wikipedia edits data with XRP price data based on date
    xrp = xrp.merge(wiki, left_index=True, right_index=True)

    #Make another section where the next day's opening price is this day's "tomorrow price"
    xrp["tomorrow"] = xrp["close"].shift(-1)

    #If price of the next day was higher than the closing price from the present day: column = 1. If not: column = 0
    xrp["target"] = (xrp["tomorrow"] > xrp["close"]).astype(int)

    # Define predictor variables for the model
    predictors = ["close", "volume", "open", "high", "low", "edit_count", "sentiment", "neg_sentiment"]

    from sklearn.metrics import precision_score

    def predict(train, test, predictors, model):
        #Allow the model to learn from the historical data
        model.fit(train[predictors], train["target"])

        #Make predictions on test data
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index, name="predictions")

        #Combine actual targets with predictions
        combined = pd.concat([test["target"], preds], axis=1)
        return combined

        #Perform rolling backtesting (see how well it predicts data that has already passed)
    def backtest(data, model, predictors, start=1095, step=150):
        all_predictions = []

        for i in range (start, data.shape[0], step):
            train = data.iloc[0:i].copy() #Starts training the model from day i (in this case day 1095)
            test = data.iloc[i:(i + step)] #Test in periods of 150 days of data
            #Uses this data to predict the next 150 days
            predictions = predict(train, test, predictors, model)
            all_predictions.append(predictions)

        return pd.concat(all_predictions)

    #Import M.L. model
    from xgboost import XGBClassifier
    #Initiate model
    model = XGBClassifier(random_state=1, learning_rate=.1, n_estimators=200)
    #Run backtesting
    predictions = backtest(xrp, model, predictors)

    def compute_rolling(xrp):
        #Find the price averages for different time periods
        horizons = [2,7,60,365]
        new_predictors = ["close", "sentiment", "neg_sentiment"]

        for horizon in horizons:
            rolling_averages = xrp.rolling(horizon, min_periods=1).mean()

            #Create a new column for each time period
            ratio_column = f"close_ratio_{horizon}"
            #Compares most recent day's price with the average from the last 2 days, 7 days, etc.
            xrp[ratio_column] = xrp["close"] / rolling_averages["close"]

            #Shows the average number of wikipedia edits from last x days (shows if there has been notable attention towards the cryptocurrency)
            xrp[edit_column] = rolling_averages["edit_count"]

            # Shows if the price from the last x days has been predominantly up or down
            rolling = xrp.rolling(horizon, closed = "left", min_periods = 1).mean()
            trend_column = f"trend_{horizon}"
            xrp[trend_column] = rolling["target"]

            #Makes these last 3 criterias as predictors for the dataframe
            new_predictors += [ratio_column, trend_column, edit_column]
        return xrp, new_predictors

    #Adds the predictors to the dataframe
    xrp, new_predictors = compute_rolling(xrp.copy())

    #Backtests again with the new criteria
    predictions = backtest(xrp, model, new_predictors)

    #Show the percentage of precision
    print(precision_score(predictions["target"], predictions["predictions"]))
    #Show past predictions and tomorrow's one
    print(predictions)

run()
