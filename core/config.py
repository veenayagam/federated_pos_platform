MENU = [
    "Chicken Tikka Masala", "Butter Chicken", "Garlic Naan", "Plain Naan",
    "Basmati Rice", "Vegetable Samosa", "Mango Lassi", "Masala Dosa",
    "Idli Sambar", "Filter Coffee", "Biryani", "Gulab Jamun",
    "Tandoori Chicken", "Lamb Vindaloo", "Paneer Tikka"
]

ITEM_TO_ID = {name: idx for idx, name in enumerate(MENU)}
NUM_ITEMS = len(MENU)

TRUE_COMBOS = [
    ("Chicken Tikka Masala", "Garlic Naan"), ("Chicken Tikka Masala", "Basmati Rice"),
    ("Butter Chicken", "Garlic Naan"), ("Butter Chicken", "Plain Naan"),
    ("Paneer Tikka", "Mango Lassi"), ("Masala Dosa", "Filter Coffee"),
    ("Idli Sambar", "Filter Coffee"), ("Masala Dosa", "Idli Sambar"),
    ("Vegetable Samosa", "Mango Lassi"), ("Vegetable Samosa", "Masala Dosa"),
    ("Biryani", "Gulab Jamun"), ("Tandoori Chicken", "Plain Naan"),
    ("Lamb Vindaloo", "Basmati Rice"), ("Biryani", "Tandoori Chicken"),
    ("Gulab Jamun", "Mango Lassi"),
]

VALID_PAIR_SET = set()
for a, b in TRUE_COMBOS:
    VALID_PAIR_SET.add((ITEM_TO_ID[a], ITEM_TO_ID[b]))
    VALID_PAIR_SET.add((ITEM_TO_ID[b], ITEM_TO_ID[a]))