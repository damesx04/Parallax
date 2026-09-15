"""Page templates for the generator. Each template returns a dict:
{name, css, body, elements: [{id, kind, label, row}], rows: [row_id, ...]}
where elements lists every interactive element (id = DOM id), kind in
{button, link, checkbox, input, select}, and `row` names the horizontal
button row an element belongs to (None otherwise). Targets are chosen among
buttons and links. Everything fits in the 896x672 viewport without scrolling.
"""
from __future__ import annotations

import html
import random

BASE_CSS = """
*{box-sizing:border-box} body{margin:0;font-family:-apple-system,Helvetica,Arial,sans-serif;color:#1f2933;background:#f7f8fa;font-size:14px}
.wrap{padding:16px 24px} h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:14px 0 6px}
.card{background:#fff;border:1px solid #d9dee5;border-radius:6px;padding:12px 14px;margin-bottom:10px}
.btn{display:inline-block;padding:7px 14px;border:1px solid #9aa5b1;border-radius:4px;background:#fff;color:#1f2933;font:inherit;cursor:pointer;line-height:18px}
.btn.primary{background:%(accent)s;border-color:%(accent)s;color:#fff}
.btn.danger{background:#fff;border-color:#d64545;color:#b91c1c}
.row{display:flex;gap:10px;align-items:center;margin:8px 0;flex-wrap:nowrap}
.field{display:flex;gap:8px;align-items:center;margin:6px 0} .field label{width:120px;color:#52606d}
input[type=text],input[type=email],select{padding:5px 8px;border:1px solid #9aa5b1;border-radius:4px;font:inherit;width:200px;background:#fff}
a{color:%(accent)s} .muted{color:#6b7785;font-size:13px} .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.item{border:1px solid #d9dee5;border-radius:6px;padding:10px;background:#fff} .item .price{font-weight:600;margin:4px 0}
.list .msg{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #eef1f4}
.stat{display:inline-block;min-width:120px;padding:8px 10px;border:1px solid #d9dee5;border-radius:6px;background:#fff;margin-right:8px}
nav a{margin-right:14px}
"""

ACCENTS = ["#2563eb", "#0f766e", "#7c3aed", "#b45309", "#be123c", "#1d4ed8"]

FIRST = ["Alex", "Priya", "Marta", "Kenji", "Sam", "Noor", "Luis", "Ida", "Tomas", "Yara"]
PRODUCTS = ["Desk lamp", "Trail shoes", "Notebook set", "Water bottle", "Wool scarf", "Bike pump",
            "Canvas tote", "Coffee grinder", "Headphones", "Plant stand", "Yoga mat", "Wall clock"]
SUBJECTS = ["Invoice #4821", "Team offsite", "Design review", "Password reset", "Shipping update",
            "Weekly digest", "Contract draft", "Lunch order", "Bug triage", "Quarterly numbers"]
PRIMARY_ACTIONS = ["Place order", "Confirm purchase", "Pay now", "Complete checkout", "Submit order"]
SAVE_ACTIONS = ["Save changes", "Apply settings", "Update profile", "Save preferences"]
MISC_BUTTONS = ["Cancel", "Back", "Edit", "Delete", "Archive", "Export", "Refresh", "Share", "Duplicate",
                "Print", "Reset", "Preview", "Reply", "Forward", "Snooze", "Add to cart", "Compare", "Details"]
LINKS = ["Home", "Orders", "Help", "Account", "Returns", "Contact", "Terms", "Privacy", "Settings", "Reports"]


class Builder:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.k = 0
        self.elements: list[dict] = []
        self.rows: list[str] = []

    def nid(self) -> str:
        self.k += 1
        return f"e{self.k}"

    def button(self, label: str, cls: str = "", row: str | None = None) -> str:
        i = self.nid()
        self.elements.append({"id": i, "kind": "button", "label": label, "row": row})
        klass = ("btn " + cls).strip()
        return f'<button type="button" id="{i}" class="{klass}">{html.escape(label)}</button>'

    def link(self, label: str) -> str:
        i = self.nid()
        self.elements.append({"id": i, "kind": "link", "label": label, "row": None})
        return f'<a href="#{i}" id="{i}">{html.escape(label)}</a>'

    def checkbox(self, label: str, checked: bool = False) -> str:
        i = self.nid()
        self.elements.append({"id": i, "kind": "checkbox", "label": label, "row": None})
        return f'<label><input type="checkbox" id="{i}"{" checked" if checked else ""}> {html.escape(label)}</label>'

    def input(self, label: str, value: str = "", typ: str = "text") -> str:
        i = self.nid()
        self.elements.append({"id": i, "kind": "input", "label": label, "row": None})
        return (f'<div class="field"><label for="{i}">{html.escape(label)}</label>'
                f'<input type="{typ}" id="{i}" value="{html.escape(value)}"></div>')

    def select(self, label: str, options: list[str]) -> str:
        i = self.nid()
        self.elements.append({"id": i, "kind": "select", "label": label, "row": None})
        opts = "".join(f"<option>{html.escape(o)}</option>" for o in options)
        return f'<div class="field"><label for="{i}">{html.escape(label)}</label><select id="{i}">{opts}</select></div>'

    def row(self, *parts: str) -> str:
        rid = f"row{len(self.rows) + 1}"
        self.rows.append(rid)
        return f'<div class="row" id="{rid}">{" ".join(parts)}</div>'

    def button_row(self, labels: list[tuple[str, str]]) -> str:
        rid = f"row{len(self.rows) + 1}"
        parts = [self.button(lab, cls, row=rid) for lab, cls in labels]
        self.rows.append(rid)
        return f'<div class="row" id="{rid}">{" ".join(parts)}</div>'


def nav(b: Builder, rng: random.Random) -> str:
    return "<nav>" + " ".join(b.link(l) for l in rng.sample(LINKS, 4)) + "</nav>"


def t_checkout(b: Builder, rng: random.Random) -> str:
    name = rng.choice(FIRST)
    items = rng.sample(PRODUCTS, 2)
    misc = rng.sample(["Cancel", "Back", "Edit cart"], 2)
    primary = rng.choice(PRIMARY_ACTIONS)
    parts = [
        f'<h1>Checkout</h1>{nav(b, rng)}',
        '<div class="card"><h2>Contact</h2>',
        b.input("Full name", name + " " + rng.choice(["Rivera", "Okafor", "Lind", "Sato"])),
        b.input("Email", name.lower() + "@example.com", "email"),
        b.select("Country", ["United States", "Canada", "Germany", "Japan"]),
        "</div>",
        '<div class="card"><h2>Delivery</h2>',
        b.checkbox("Gift wrap", rng.random() < 0.5),
        " " + b.checkbox("Leave at door", rng.random() < 0.5),
        b.select("Speed", ["Standard (5 days)", "Express (2 days)", "Overnight"]),
        "</div>",
        '<div class="card"><h2>Items</h2>',
        f'<p class="muted">{html.escape(items[0])} x1, {html.escape(items[1])} x2 - total ${rng.randint(20, 300)}.{rng.randint(10, 99)}</p>',
        b.button_row([(misc[0], ""), (misc[1], ""), (primary, "primary")]),
        "</div>",
    ]
    return "\n".join(parts)


def t_settings(b: Builder, rng: random.Random) -> str:
    save = rng.choice(SAVE_ACTIONS)
    parts = [
        f'<h1>Account settings</h1>{nav(b, rng)}',
        '<div class="card"><h2>Profile</h2>',
        b.input("Display name", rng.choice(FIRST)),
        b.select("Language", ["English", "Deutsch", "Espanol", "Nihongo"]),
        b.select("Time zone", ["UTC", "Europe/Berlin", "America/New_York", "Asia/Tokyo"]),
        "</div>",
        '<div class="card"><h2>Notifications</h2>',
        b.checkbox("Email me about new sign-ins", rng.random() < 0.5), " ",
        b.checkbox("Weekly summary", rng.random() < 0.5), " ",
        b.checkbox("Product news", rng.random() < 0.5),
        "</div>",
        '<div class="card"><h2>Danger zone</h2>',
        b.button_row([(rng.choice(["Export data", "Download archive"]), ""), ("Delete account", "danger")]),
        "</div>",
        b.button_row([("Cancel", ""), (save, "primary")]),
    ]
    return "\n".join(parts)


def t_products(b: Builder, rng: random.Random) -> str:
    prods = rng.sample(PRODUCTS, 6)
    cards = []
    for p in prods:
        price = f"${rng.randint(8, 120)}.{rng.choice(['00', '50', '99'])}"
        cards.append(
            f'<div class="item"><div>{html.escape(p)}</div><div class="price">{price}</div>'
            + b.button_row([(f"Add {p.split()[0].lower()}", ""), ("Details", "")]) + "</div>"
        )
    parts = [f'<h1>Shop</h1>{nav(b, rng)}', b.select("Sort by", ["Relevance", "Price: low to high", "Newest"]),
             '<div class="grid">' + "".join(cards) + "</div>"]
    return "\n".join(parts)


def t_inbox(b: Builder, rng: random.Random) -> str:
    subs = rng.sample(SUBJECTS, 5)
    rows = []
    for s in subs:
        who = rng.choice(FIRST)
        rows.append(f'<div class="msg"><span>{html.escape(who)}: {html.escape(s)}</span>'
                    + b.button_row([(rng.choice(["Reply", "Archive", "Snooze"]), ""), ("Open", "")]) + "</div>")
    parts = [f'<h1>Inbox</h1>{nav(b, rng)}',
             b.row(b.button("Compose", "primary"), b.button("Refresh"), b.button("Mark all read")),
             '<div class="card list">' + "".join(rows) + "</div>"]
    return "\n".join(parts)


def t_dashboard(b: Builder, rng: random.Random) -> str:
    parts = [
        f'<h1>Dashboard</h1>{nav(b, rng)}',
        '<div class="card">'
        + "".join(f'<span class="stat"><span class="muted">{k}</span><br><b>{v}</b></span>' for k, v in [
            ("Revenue", f"${rng.randint(10, 99)},{rng.randint(100, 999)}"), ("Orders", str(rng.randint(100, 999))),
            ("Refunds", str(rng.randint(1, 40))), ("Visitors", f"{rng.randint(1, 9)}.{rng.randint(1, 9)}k")])
        + "</div>",
        '<div class="card"><h2>Reports</h2>',
        b.select("Period", ["Last 7 days", "Last 30 days", "This quarter"]),
        b.button_row([("Export CSV", ""), ("Generate report", "primary"), ("Schedule", "")]),
        "</div>",
        '<div class="card"><h2>Alerts</h2>',
        f'<p class="muted">{rng.randint(2, 9)} low-stock items need attention.</p>',
        b.button_row([("Dismiss all", ""), ("Review stock", "primary")]),
        "</div>",
    ]
    return "\n".join(parts)


TEMPLATES = {"checkout": t_checkout, "settings": t_settings, "products": t_products,
             "inbox": t_inbox, "dashboard": t_dashboard}


def build(template: str, rng: random.Random) -> dict:
    b = Builder(rng)
    accent = rng.choice(ACCENTS)
    body = TEMPLATES[template](b, rng)
    return {"name": template, "css": BASE_CSS % {"accent": accent}, "body": f'<div class="wrap">{body}</div>',
            "elements": b.elements, "rows": b.rows}
