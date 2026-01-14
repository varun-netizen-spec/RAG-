import pandas as pd
from datetime import datetime, timedelta
import random
import os

os.makedirs("data", exist_ok=True)

products = ["milk", "eggs", "goat meat"]
locations = ["Chennai", "Coimbatore", "Madurai"]

rows = []
start_date = datetime.today() - timedelta(days=20)

for i in range(20):
    for product in products:
        rows.append({
            "date": (start_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            "product": product,
            "price": random.randint(40, 55) if product == "milk" else random.randint(5, 8),
            "location": random.choice(locations)
        })

df = pd.DataFrame(rows)
df.to_csv("data/market_prices.csv", index=False)

with open("data/seasonal_notes.txt", "w") as f:
    f.write(
        "Festival seasons increase milk and meat demand by 10 to 20 percent.\n"
        "Summer reduces milk yield.\n"
        "Weekends show higher egg demand."
    )

print("✅ Dataset generated successfully")
