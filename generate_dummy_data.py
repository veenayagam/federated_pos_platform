import random
import polars as pl
from core.config import TRUE_COMBOS

random.seed(42)
store_combo_slices = {1: TRUE_COMBOS[0:5], 2: TRUE_COMBOS[5:10], 3: TRUE_COMBOS[10:15]}

for rest_id in [1, 2, 3]:
    records = []
    local_combos = store_combo_slices[rest_id]
    for order_idx in range(220):
        order_id = f"CLV-R{rest_id}-{1000 + order_idx}"
        combo = random.choice(local_combos) if random.random() < 0.85 else random.choice(TRUE_COMBOS)
        for item in combo:
            records.append({
                "Order ID": order_id,
                "Item Name": item,
                "Price": round(random.uniform(4.0, 18.0), 2),
                "Customer Name": f"Redacted_{random.randint(1,999)}",
            })
    pl.DataFrame(records).write_csv(f"restaurant_{rest_id}_clover.csv")
    print(f"Generated restaurant_{rest_id}_clover.csv")