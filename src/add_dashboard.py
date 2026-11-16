from pathlib import Path

app_file = Path("app/app.py")
source = app_file.read_text(encoding="utf-8")

if '@app.route("/dashboard")' in source:
    print("Dashboard route already exists. No changes made.")
else:
    dashboard_code = r'''

@app.route("/dashboard")
def dashboard():
    import json

    total_records = len(df)
    average_price = float(df["Daily_Modal_Price"].mean())
    highest_price = float(df["Daily_Modal_Price"].max())
    total_markets = int(df["Market"].nunique())

    monthly = (
        df.groupby(df["Arrival_Date"].dt.to_period("M"))["Daily_Modal_Price"]
        .mean()
        .round(2)
    )

    state_data = (
        df.groupby("State")["Daily_Modal_Price"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .round(2)
    )

    monthly_labels = [str(x) for x in monthly.index]
    monthly_values = monthly.tolist()

    state_labels = state_data.index.astype(str).tolist()
    state_values = state_data.tolist()

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agricultural Analytics Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: #f1f5f2;
                color: #26352d;
            }
            header {
                background: #174d35;
                color: white;
                padding: 25px;
                text-align: center;
            }
            .container {
                max-width: 1150px;
                margin: 25px auto;
                padding: 0 15px;
            }
            .cards {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
            }
            .card, .chart {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 3px 12px #00000010;
            }
            .card h3 { color: #64746a; font-size: 14px; }
            .number {
                color: #16804e;
                font-size: 25px;
                font-weight: bold;
            }
            .charts {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 18px;
                margin-top: 20px;
            }
            .chart { min-width: 0; }
            .chart h2 { font-size: 18px; }
            .back {
                display: inline-block;
                margin: 20px 0;
                color: #16804e;
                font-weight: bold;
                text-decoration: none;
            }
            footer {
                text-align: center;
                padding: 25px;
                color: #64746a;
            }
            @media(max-width: 750px) {
                .cards { grid-template-columns: 1fr 1fr; }
                .charts { grid-template-columns: 1fr; }
            }
            @media(max-width: 420px) {
                .cards { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <header>
            <h1>🌾 Agricultural Analytics Dashboard</h1>
            <p>Onion Price Analysis Using Machine Learning</p>
        </header>

        <div class="container">
            <a class="back" href="/">← Back to Price Prediction</a>

            <div class="cards">
                <div class="card">
                    <h3>Total Records</h3>
                    <div class="number">RECORDS</div>
                </div>
                <div class="card">
                    <h3>Average Onion Price</h3>
                    <div class="number">AVERAGE</div>
                    <p>Per quintal</p>
                </div>
                <div class="card">
                    <h3>Highest Recorded Price</h3>
                    <div class="number">HIGHEST</div>
                    <p>Per quintal</p>
                </div>
                <div class="card">
                    <h3>Total Markets</h3>
                    <div class="number">MARKETS</div>
                </div>
            </div>

            <div class="charts">
                <div class="chart">
                    <h2>Monthly Average Onion Price</h2>
                    <canvas id="monthlyChart"></canvas>
                </div>
                <div class="chart">
                    <h2>Top 10 States by Average Price</h2>
                    <canvas id="stateChart"></canvas>
                </div>
            </div>
        </div>

        <footer>
            Agricultural Price Analysis and Prediction Using Machine Learning
            <br>Developed by P. Naresh | Saveetha University, Chennai
        </footer>

        <script>
            const monthlyLabels = MONTHLY_LABELS;
            const monthlyValues = MONTHLY_VALUES;
            const stateLabels = STATE_LABELS;
            const stateValues = STATE_VALUES;

            new Chart(document.getElementById("monthlyChart"), {
                type: "line",
                data: {
                    labels: monthlyLabels,
                    datasets: [{
                        label: "Average Price (₹)",
                        data: monthlyValues,
                        borderWidth: 2,
                        tension: 0.3
                    }]
                },
                options: { responsive: true }
            });

            new Chart(document.getElementById("stateChart"), {
                type: "bar",
                data: {
                    labels: stateLabels,
                    datasets: [{
                        label: "Average Price (₹)",
                        data: stateValues,
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: "y",
                    responsive: true
                }
            });
        </script>
    </body>
    </html>
    """

    replacements = {
        "RECORDS": f"{total_records:,}",
        "AVERAGE": f"₹{average_price:,.2f}",
        "HIGHEST": f"₹{highest_price:,.2f}",
        "MARKETS": f"{total_markets:,}",
        "MONTHLY_LABELS": json.dumps(monthly_labels),
        "MONTHLY_VALUES": json.dumps(monthly_values),
        "STATE_LABELS": json.dumps(state_labels),
        "STATE_VALUES": json.dumps(state_values)
    }

    for key, value in replacements.items():
        html = html.replace(key, value)

    return html

'''

    marker = '\nif __name__ == "__main__":'
    if marker not in source:
        raise SystemExit("Could not find the main application section.")

    source = source.replace(marker, dashboard_code + marker, 1)
    app_file.write_text(source, encoding="utf-8")
    print("Dashboard route added successfully.")
