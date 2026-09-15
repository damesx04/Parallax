"""Extension: extension loads; stable IDs; annotated tree; overlay; fidelity."""
import io
from pathlib import Path

from PIL import Image

from parallax.browser import goto, VIEWPORT
from parallax.tree import draw_overlay, get_tree, render_tree

FIX = Path(__file__).parent / "fixtures"


def ids_by_dom_id(page):
    return page.evaluate(
        "() => Object.fromEntries([...document.querySelectorAll('[id]')].map(e => [e.id, window.__parallax.idOf(e)]))"
    )


def test_extension_loads_headless(page):
    goto(page, str(FIX / "static.html"))
    assert page.evaluate("() => typeof window.__parallax.tree") == "function"
    assert page.evaluate("() => window.__parallax.version") == "0.1.0"


def test_ids_assigned_in_document_order_and_mirrored(page):
    goto(page, str(FIX / "static.html"))
    ids = ids_by_dom_id(page)
    assert ids["title"] < ids["lnk1"] < ids["lnk2"] < ids["email"] < ids["place"]
    mirrored = page.evaluate("() => document.getElementById('place').getAttribute('data-pxid')")
    assert int(mirrored) == ids["place"]


def test_ids_stable_under_reflow_restyle_reorder_reattach(page):
    goto(page, str(FIX / "dynamic.html"))
    before = ids_by_dom_id(page)
    tree_before = {e["id"]: e for e in get_tree(page)}
    page.evaluate("() => reflow(150)")
    page.evaluate("() => restyle()")
    page.evaluate("() => reorder()")
    page.evaluate("() => detachReattach()")
    after = ids_by_dom_id(page)
    for k in ("b1", "b2", "b3", "list", "spacer"):
        assert before[k] == after[k], k
    tree_after = {e["id"]: e for e in get_tree(page)}
    # boxes moved (reflow) but the same ids name the same elements
    assert tree_after[before["b2"]]["name"] == "Two"
    assert tree_after[before["b2"]]["box"][1] > tree_before[before["b2"]]["box"][1]


def test_new_nodes_fresh_ids_clones_new_ids_no_reuse(page):
    goto(page, str(FIX / "dynamic.html"))
    before = ids_by_dom_id(page)
    max_before = page.evaluate("() => window.__parallax.counter()")
    page.evaluate("() => addNew()")
    page.evaluate("() => clone()")
    after = ids_by_dom_id(page)
    assert after["b4"] > max_before
    assert after["b1clone"] > max_before
    assert after["b1clone"] != after["b1"]
    assert page.evaluate("() => document.getElementById('b1clone').getAttribute('data-pxid')") == str(after["b1clone"])
    # removing and adding again never reuses an id
    page.evaluate("() => document.getElementById('b4').remove()")
    page.evaluate("() => addNew()")
    again = ids_by_dom_id(page)
    assert again["b4"] > after["b1clone"]
    assert len(set(again.values())) == len(again)


def test_tree_entries_and_hidden_exclusion(page):
    goto(page, str(FIX / "static.html"))
    tree = get_tree(page)
    assert all({"id", "role", "name", "states", "box", "depth"} <= set(e) for e in tree)
    by_name = {(e["role"], e["name"]): e for e in tree}
    assert ("button", "Place order") in by_name
    assert ("button", "Cancel") in by_name
    assert ("link", "Home") in by_name
    assert ("textbox", "Email address") in by_name
    assert ("checkbox", "Gift wrap") in by_name and "checked" in by_name[("checkbox", "Gift wrap")]["states"]
    assert ("combobox", "Shipping") in by_name
    assert ("button", "Close dialog") in by_name
    assert "disabled" in by_name[("button", "Disabled")]["states"]
    assert ("heading", "Checkout") in by_name
    assert ("img", "Product photo") in by_name
    assert ("listitem", "Item one") in by_name
    names = {e["name"] for e in tree}
    assert "Hidden button" not in names and "Invisible button" not in names and "Aria hidden" not in names
    text = render_tree(tree)
    assert '[%d] button "Place order"' % by_name[("button", "Place order")]["id"] in text
    ids = [e["id"] for e in tree]
    assert len(ids) == len(set(ids))


def test_overlay_labels_match_boxes_and_pixels(page):
    goto(page, str(FIX / "static.html"))
    n = draw_overlay(page)
    tree = get_tree(page)
    assert n == len(tree) > 5
    labels = {l["id"]: l["box"] for l in page.evaluate("() => window.__parallax.labelBoxes()")}
    assert set(labels) == {e["id"] for e in tree}
    for e in tree:
        lx, ly, _, _ = labels[e["id"]]
        assert abs(lx - e["box"][0]) <= 4 and abs(ly - e["box"][1]) <= 4, e
    pe = page.evaluate("() => getComputedStyle(document.getElementById('__parallax_overlay')).pointerEvents")
    assert pe == "none"
    png = page.screenshot()
    im = Image.open(io.BytesIO(png)).convert("RGB")
    assert im.size == (VIEWPORT["width"], VIEWPORT["height"])
    yellowish = 0
    for e in tree:
        x, y = e["box"][0] + 3, e["box"][1] + 3
        if 0 <= x < im.width and 0 <= y < im.height:
            r, g, b = im.getpixel((x, y))
            if r > 200 and g > 200 and b < 120:
                yellowish += 1
    assert yellowish >= 0.9 * len(tree), (yellowish, len(tree))
    # overlay is not part of the tree
    assert all(e["role"] != "generic" or not e["name"].isdigit() for e in tree)


def _fidelity_counts(page, browser):
    tree = get_tree(page)
    cdp = browser.context.new_cdp_session(page)
    cdp.send("DOM.enable")
    cdp.send("Accessibility.enable")
    doc = cdp.send("DOM.getDocument", {"depth": -1, "pierce": True})["root"]
    backend_to_pxid = {}

    def walk(n):
        attrs = n.get("attributes") or []
        for i in range(0, len(attrs) - 1, 2):
            if attrs[i] == "data-pxid":
                backend_to_pxid[n["backendNodeId"]] = int(attrs[i + 1])
        for c in n.get("children", []) + n.get("shadowRoots", []):
            walk(c)
        if "contentDocument" in n:
            walk(n["contentDocument"])

    walk(doc)
    ax = cdp.send("Accessibility.getFullAXTree")["nodes"]
    ax_by_pxid = {}
    for node in ax:
        if node.get("ignored"):
            continue
        bid = node.get("backendDOMNodeId")
        if bid in backend_to_pxid:
            ax_by_pxid[backend_to_pxid[bid]] = (
                (node.get("role") or {}).get("value"),
                (node.get("name") or {}).get("value", ""),
            )
    interactive = {"button", "link", "textbox", "checkbox", "combobox", "radio", "searchbox", "spinbutton"}
    checked = matched = 0
    for e in tree:
        if e["role"] not in interactive:
            continue
        checked += 1
        role, name = ax_by_pxid.get(e["id"], (None, None))
        norm = lambda t: " ".join((t or "").split())
        if role == e["role"] and norm(name) == norm(e["name"]):
            matched += 1
        else:
            print("MISMATCH", page.url, e, role, name)
    cdp.detach()
    return matched, checked


def test_fidelity_against_cdp_ax_tree(browser):
    """Roles+names of interactive entries match Chrome's own AX tree on >= 95% across the fixtures and a generated page."""
    from parallax.pages.loader import DEV_DIR
    pages = [FIX / n for n in ("static.html", "dynamic.html", "overlay.html", "reflow.html")]
    pages += sorted(DEV_DIR.glob("*/page.html"))[:6]
    matched = checked = 0
    for path in pages:
        page = browser.new_page()
        goto(page, str(path))
        m, c = _fidelity_counts(page, browser)
        page.close()
        matched += m
        checked += c
    assert checked >= 60
    assert matched / checked >= 0.95, (matched, checked)


def test_ids_are_universal(page):
    """Every element (outside the overlay) has a stable id, not only tree entries."""
    goto(page, str(FIX / "static.html"))
    draw_overlay(page)
    assert page.evaluate("() => [...document.querySelectorAll('*')].filter(e => !e.closest('#__parallax_overlay')).every(e => window.__parallax.idOf(e) !== null)")
    n_elements = page.evaluate("() => document.querySelectorAll('*').length - document.querySelectorAll('#__parallax_overlay, #__parallax_overlay *').length")
    assert n_elements > len(get_tree(page))
    page.evaluate("() => { const d = document.createElement('div'); d.innerHTML = '<span><i>late</i></span>'; document.body.appendChild(d); }")
    assert page.evaluate("() => window.__parallax.idOf(document.querySelector('i')) !== null")


def test_pinned_browser_build(page, browser):
    goto(page, str(FIX / "static.html"))
    product = browser.context.new_cdp_session(page).send("Browser.getVersion")["product"]
    assert "Chrome/151.0.7922.34" in product, product


def test_elementFor_survives_cloned_mirror_attribute(page):
    goto(page, str(FIX / "dynamic.html"))
    b1 = page.evaluate("() => window.__parallax.idFor('#b1')")
    # clone with a stale mirror attribute inserted BEFORE the original in document order
    page.evaluate("() => { const c = document.getElementById('b1').cloneNode(true); c.id = 'pre'; document.getElementById('list').prepend(c); }")
    assert page.evaluate("id => window.__parallax.boxFor(id) !== null", b1)
    assert page.evaluate("id => window.__parallax.describe(id).name", b1) == "One"
    assert page.evaluate("() => window.__parallax.idFor('#pre')") != b1
