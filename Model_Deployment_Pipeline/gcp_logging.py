from google.cloud import bigquery

def log_prediction(review: str, sentiment: str):
    try:
        client = bigquery.Client()
        table_id = "mlops-project-test-448822.sentiment_data.user_logs"  # replace with your actual table
        row = [{
            "review": str(review),
            "sentiment": str(sentiment)
        }]
        errors = client.insert_rows_json(table_id, row)
        if errors:
            print("🚨 BigQuery insert errors:", errors)
    except Exception as e:
        print("🚨 Error logging prediction to BigQuery:", e)
