from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import date

app = Flask(__name__)
app.secret_key = "agricultural-price-project-key"

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "onion_advanced_features.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_historical_experiment.pkl"

print("Loading historical agricultural data...")
df = pd.read_csv(DATA_PATH)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])
df = df.sort_values("Arrival_Date")

print("Loading trained XGBoost model...")
model = joblib.load(MODEL_PATH)

categorical = ["State", "District", "Market", "Variety", "Grade"]

numeric = [
    "Month", "Day", "DayOfWeek", "DayOfYear", "Quarter",
    "Lag_1", "Lag_2", "Rolling_Mean_7",
    "Rolling_Mean_30", "Previous_Price_Change"
]

features = categorical + numeric

PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Agricultural Price Prediction</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
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
            padding: 28px 15px;
            text-align: center;
        }
        header h1 { margin: 0 0 8px; }
        main {
            max-width: 760px;
            margin: 30px auto;
            padding: 0 15px;
        }
        .card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 3px 15px #00000012;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        label {
            display: block;
            font-weight: bold;
            margin-bottom: 7px;
        }
        input {
            width: 100%;
            padding: 12px;
            border: 1px solid #ccd5ce;
            border-radius: 6px;
            margin-bottom: 5px;
        }
        button {
            width: 100%;
            padding: 14px;
            background: #16804e;
            border: 0;
            color: white;
            border-radius: 7px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
        }
        button:hover { background: #105f39; }
        .result {
            background: #e5f5e9;
            border-left: 5px solid #16804e;
            padding: 18px;
            margin-top: 22px;
            border-radius: 7px;
        }
        .price {
            font-size: 32px;
            color: #12663d;
            font-weight: bold;
        }
        .error {
            background: #ffe9e9;
            padding: 15px;
            border-radius: 7px;
            color: #9b2222;
            margin-top: 15px;
        }
        .note {
            color: #64746a;
            font-size: 13px;
            line-height: 1.6;
            margin-top: 18px;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #64746a;
            font-size: 13px;
        }
        @media(max-width: 550px) {
            .grid { grid-template-columns: 1fr; gap: 8px; }
        }
    </style>
</head>
<body>
<header>
    <h1>🌾 Agricultural Price Prediction</h1>
    <p>Onion Market Price Analysis Using Machine Learning</p>
</header>

<main>
<div class="card">
    <h2>Predict Onion Price</h2>
    <p>Enter market details to estimate the next recorded price.</p>

    <form method="POST">
        <div class="grid">
            <div>
                <label>State</label>
                <input name="state" list="states" required
                       value="{{ values.get('state','') }}">
                <datalist id="states">
                    {% for x in states %}<option value="{{ x }}">{% endfor %}
                </datalist>
            </div>

            <div>
                <label>District</label>
                <input name="district" list="districts" required
                       value="{{ values.get('district','') }}">
                <datalist id="districts">
                    {% for x in districts %}<option value="{{ x }}">{% endfor %}
                </datalist>
            </div>

            <div>
                <label>Market</label>
                <input name="market" list="markets" required
                       value="{{ values.get('market','') }}">
                <datalist id="markets">
                    {% for x in markets %}<option value="{{ x }}">{% endfor %}
                </datalist>
            </div>

            <div>
                <label>Variety</label>
                <input name="variety" list="varieties" required
                       value="{{ values.get('variety','') }}">
                <datalist id="varieties">
                    {% for x in varieties %}<option value="{{ x }}">{% endfor %}
                </datalist>
            </div>

            <div>
                <label>Grade</label>
                <input name="grade" list="grades" required
                       value="{{ values.get('grade','') }}">
                <datalist id="grades">
                    {% for x in grades %}<option value="{{ x }}">{% endfor %}
                </datalist>
            </div>

            <div>
                <label>Prediction Date</label>
                <input type="date" name="prediction_date"
                       min="{{ today }}" value="{{ values.get('prediction_date',today) }}"
                       required>
            </div>
        </div>

        <button type="submit">Predict Onion Price</button>
    </form>

    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}

    {% if result %}
    <div class="result">
        <h3>Estimated Onion Price</h3>
        <div class="price">₹{{ "%.2f"|format(result.price) }}</div>
        <p>Estimated price per quintal</p>
        <p>Last observed price date: {{ result.last_date }}</p>
        <p>Previous observed price: ₹{{ "%.2f"|format(result.last_price) }}</p>
        <p>Prediction date: {{ result.prediction_date }}</p>
        <p>Model: XGBoost</p>
    </div>
    {% endif %}

    <p class="note">
        Note: This is a machine-learning estimate based on historical
        market records. It is not an official market quotation or a
        guaranteed future price. The prediction uses the latest available
        historical observations for the selected market.
    </p>
</div>
</main>

<footer>
    Agricultural Price Analysis and Prediction Using Machine Learning
    <br>Developed by P. Naresh | Saveetha University, Chennai
</footer>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None
    values = {}

    def choices(column):
        return sorted(df[column].dropna().astype(str).unique().tolist())

    if request.method == "POST":

        values = {
            "state": request.form.get("state", "").strip(),
            "district": request.form.get("district", "").strip(),
            "market": request.form.get("market", "").strip(),
            "variety": request.form.get("variety", "").strip(),
            "grade": request.form.get("grade", "").strip(),
            "prediction_date": request.form.get("prediction_date", "")
        }

        try:
            target_date = pd.to_datetime(
                values["prediction_date"],
                errors="raise"
            )

            if target_date.date() < date.today():
                raise ValueError("Choose today or a future prediction date.")

            if target_date.year < 2026:
                raise ValueError("Select a prediction date in 2026 or later.")

            history = df.copy()

            for column in categorical:
                history = history[
                    history[column].astype(str) == values[column.lower()]
                ]

            history = history[
                history["Arrival_Date"] < target_date
            ].sort_values("Arrival_Date")

            if len(history) < 7:
                raise ValueError(
                    "Not enough historical records for this exact market, "
                    "variety and grade combination. Try another selection."
                )

            prices = history["Daily_Modal_Price"].to_numpy()

            lag_1 = float(prices[-1])
            lag_2 = float(prices[-2])

            input_data = {
                "State": values["state"],
                "District": values["district"],
                "Market": values["market"],
                "Variety": values["variety"],
                "Grade": values["grade"],
                "Month": target_date.month,
                "Day": target_date.day,
                "DayOfWeek": target_date.dayofweek,
                "DayOfYear": target_date.dayofyear,
                "Quarter": target_date.quarter,
                "Lag_1": lag_1,
                "Lag_2": lag_2,
                "Rolling_Mean_7": float(np.mean(prices[-7:])),
                "Rolling_Mean_30": float(np.mean(prices[-30:])),
                "Previous_Price_Change": lag_1 - lag_2
            }

            input_df = pd.DataFrame([input_data])[features]

            predicted_price = float(
                np.maximum(model.predict(input_df)[0], 0)
            )

            result = {
                "price": predicted_price,
                "last_date": history["Arrival_Date"].iloc[-1].strftime("%Y-%m-%d"),
                "last_price": lag_1,
                "prediction_date": target_date.strftime("%Y-%m-%d")
            }

        except ValueError as exc:
            error = str(exc)
        except Exception as exc:
            error = "Prediction failed. Check the selected market details. " + str(exc)

    return render_template_string(
        PAGE,
        states=choices("State"),
        districts=choices("District"),
        markets=choices("Market"),
        varieties=choices("Variety"),
        grades=choices("Grade"),
        today=date.today().isoformat(),
        values=values,
        result=result,
        error=error
    )



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


if __name__ == "__main__":
    print("\nAgricultural Price Prediction App is starting...")
    app.run(debug=True, port=5001)
