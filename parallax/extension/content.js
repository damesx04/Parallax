// Parallax Annotator content script (MAIN world, document_start).
//
// 1. Stable node IDs: every element gets an integer ID the first time it is
//    observed (document order), kept in a WeakMap keyed by the Element object
//    and mirrored as data-pxid. IDs are never reused within a document.
// 2. Annotated accessibility tree: window.__parallax.tree() returns entries
//    {id, role, name, states, box, depth} computed from ARIA rules.
// 3. Overlay: window.__parallax.overlay(true) draws a label with the same ID
//    at the top-left of each entry's box in a pointer-events:none,
//    aria-hidden layer so screenshots show the IDs.
// 4. Click recorder: capture-phase listener records every click's recipient.
(function () {
  if (window.__parallax) return;

  const OVERLAY_ID = "__parallax_overlay";
  const ids = new WeakMap();
  let counter = 0;
  const clicks = [];
  const hiddenState = new Map();

  function isOverlayNode(el) {
    return el.id === OVERLAY_ID || (el.closest && el.closest("#" + OVERLAY_ID) !== null);
  }

  function assign(el) {
    let id = ids.get(el);
    if (id === undefined) {
      id = ++counter;
      ids.set(el, id);
    }
    // Mirror; corrects clones that copied another element's attribute.
    if (el.getAttribute("data-pxid") !== String(id)) el.setAttribute("data-pxid", String(id));
    return id;
  }

  function scan() {
    const root = document.body || document.documentElement;
    if (!root) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
    let node = walker.currentNode;
    while (node) {
      if (node.nodeType === 1 && !isOverlayNode(node)) assign(node);
      node = walker.nextNode();
    }
  }

  const observer = new MutationObserver(function () { scan(); });
  observer.observe(document.documentElement, { childList: true, subtree: true });
  scan();

  // ---------- roles and names ----------
  const INPUT_ROLES = {
    text: "textbox", email: "textbox", password: "textbox", search: "searchbox",
    tel: "textbox", url: "textbox", number: "spinbutton", checkbox: "checkbox",
    radio: "radio", submit: "button", button: "button", reset: "button",
    range: "slider", image: "button",
  };
  const TAG_ROLES = {
    button: "button", textarea: "textbox", h1: "heading", h2: "heading", h3: "heading",
    h4: "heading", h5: "heading", h6: "heading", img: "img", li: "listitem", ul: "list",
    ol: "list", nav: "navigation", main: "main", header: "banner", footer: "contentinfo",
    form: "form", table: "table", dialog: "dialog", p: "paragraph", summary: "button",
    option: "option", hr: "separator", article: "article", section: "region",
    aside: "complementary", label: "label",
  };
  const INTERACTIVE_ROLES = new Set([
    "button", "link", "textbox", "searchbox", "spinbutton", "checkbox", "radio",
    "combobox", "listbox", "slider", "switch", "tab", "menuitem", "option",
    "menuitemcheckbox", "menuitemradio",
  ]);
  const STRUCT_ROLES = new Set(["heading", "img", "listitem", "dialog", "alertdialog", "alert"]);

  function roleOf(el) {
    const explicit = el.getAttribute("role");
    if (explicit) return explicit.trim().split(/\s+/)[0];
    const tag = el.tagName.toLowerCase();
    if (tag === "a") return el.hasAttribute("href") ? "link" : "generic";
    if (tag === "input") return INPUT_ROLES[(el.getAttribute("type") || "text").toLowerCase()] || "textbox";
    if (tag === "select") return el.multiple || el.size > 1 ? "listbox" : "combobox";
    return TAG_ROLES[tag] || "generic";
  }

  function textOf(el) {
    return (el.textContent || "").replace(/\s+/g, " ").trim();
  }

  function nameOf(el, role) {
    const aria = el.getAttribute("aria-label");
    if (aria && aria.trim()) return aria.trim();
    const labelledby = el.getAttribute("aria-labelledby");
    if (labelledby) {
      const parts = labelledby.split(/\s+/).map(function (id) {
        const ref = document.getElementById(id);
        return ref ? textOf(ref) : "";
      }).filter(Boolean);
      if (parts.length) return parts.join(" ");
    }
    const tag = el.tagName.toLowerCase();
    if (tag === "img" && el.hasAttribute("alt")) return el.getAttribute("alt").trim();
    if (tag === "input" || tag === "select" || tag === "textarea") {
      if (el.id) {
        const lab = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
        if (lab) return textOf(lab);
      }
      const wrap = el.closest("label");
      if (wrap) {
        const clone = wrap.cloneNode(true);
        clone.querySelectorAll("input,select,textarea").forEach(function (n) { n.remove(); });
        const t = textOf(clone);
        if (t) return t;
      }
      const t = el.getAttribute("type");
      if (tag === "input" && (t === "submit" || t === "button" || t === "reset")) {
        if (el.value) return el.value;
        return t === "submit" ? "Submit" : t === "reset" ? "Reset" : "";
      }
    }
    const title = el.getAttribute("title");
    if (title && title.trim()) return title.trim();
    const ph = el.getAttribute("placeholder");
    if (ph && ph.trim()) return ph.trim();
    if (INTERACTIVE_ROLES.has(role) || STRUCT_ROLES.has(role) || role === "paragraph" || role === "label") {
      return textOf(el);
    }
    // Generic text block: direct text only.
    let direct = "";
    for (const c of el.childNodes) if (c.nodeType === 3) direct += c.textContent;
    return direct.replace(/\s+/g, " ").trim();
  }

  function statesOf(el) {
    const s = [];
    if (el.disabled || el.getAttribute("aria-disabled") === "true") s.push("disabled");
    if (el.checked || el.getAttribute("aria-checked") === "true") s.push("checked");
    if (el.getAttribute("aria-expanded") === "true") s.push("expanded");
    if (el.selected || el.getAttribute("aria-selected") === "true") s.push("selected");
    if (el.required) s.push("required");
    return s;
  }

  function isHidden(el) {
    if (el.getAttribute("aria-hidden") === "true") return true;
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden") return true;
    return false;
  }

  function boxOf(el) {
    const r = el.getBoundingClientRect();
    return [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)];
  }

  function onScreen(b) {
    const W = window.innerWidth, H = window.innerHeight;
    return b[2] > 0 && b[3] > 0 && b[0] + b[2] > 0 && b[1] + b[3] > 0 && b[0] < W && b[1] < H;
  }

  function includes(el, role) {
    if (INTERACTIVE_ROLES.has(role)) return true;
    // explicitly roled or labelled elements are accessibility nodes (landmarks, regions, dialogs, widgets)
    const explicit = el.getAttribute("role");
    if (explicit && explicit !== "presentation" && explicit !== "none") return true;
    if (el.hasAttribute("aria-label") || el.hasAttribute("aria-labelledby")) return true;
    if (el.hasAttribute("tabindex") || el.hasAttribute("data-px-clickable")) return true;
    if (STRUCT_ROLES.has(role)) return true;
    if (role === "paragraph" || role === "label") return textOf(el).length > 0;
    if (role === "generic" || role === "region" || role === "article" || role === "listitem") {
      for (const c of el.childNodes) if (c.nodeType === 3 && c.textContent.trim()) return true;
    }
    return false;
  }

  // Returns the annotated accessibility tree as a flat list in document
  // order with depth = number of ancestor entries.
  function tree() {
    scan();
    const out = [];
    const root = document.body;
    if (!root) return out;
    function walk(el, depth, hiddenAncestor, insideInteractive) {
      if (isOverlayNode(el)) return;
      const hidden = hiddenAncestor || isHidden(el);
      const role = roleOf(el);
      let d = depth;
      if (!hidden) {
        const box = boxOf(el);
        const interactive = INTERACTIVE_ROLES.has(role) || el.hasAttribute("tabindex") || el.hasAttribute("data-px-clickable");
        if (!insideInteractive && includes(el, role) && onScreen(box)) {
          out.push({ id: assign(el), role: role, name: nameOf(el, role), states: statesOf(el), box: box, depth: depth });
          d = depth + 1;
          if (interactive) insideInteractive = true;
        }
      }
      for (const c of el.children) walk(c, d, hidden, insideInteractive);
    }
    walk(root, 0, false, false);
    return out;
  }

  function elementFor(id) {
    // the WeakMap is the source of truth; the mirror attribute is only an index
    const cands = document.querySelectorAll('[data-pxid="' + id + '"]');
    for (const el of cands) if (ids.get(el) === Number(id)) return el;
    return null;
  }

  function idOf(el) {
    while (el && el.nodeType === 1) {
      if (isOverlayNode(el)) return null;
      const id = ids.get(el);
      if (id !== undefined) return id;
      el = el.parentElement;
    }
    return null;
  }


  function pathOf(el) {
    const out = [];
    while (el && el.nodeType === 1) {
      if (isOverlayNode(el)) return out;
      const id = ids.get(el);
      if (id !== undefined) out.push(id);
      el = el.parentElement;
    }
    return out;
  }

  // Hit-test probe for verify_target: centre hit, grid of sample points,
  // occluders (elements above n at any sampled point) and a clear point.
  function probe(id, nx, ny) {
    const el = elementFor(id);
    if (!el) return { found: false };
    const box = boxOf(el);
    const W = window.innerWidth, H = window.innerHeight;
    const cx = box[0] + box[2] / 2, cy = box[1] + box[3] / 2;
    const chit = document.elementFromPoint(cx, cy);
    const center = { x: cx, y: cy, id: chit ? idOf(chit) : null, path: chit ? pathOf(chit) : [], tag: chit ? chit.tagName.toLowerCase() : null };
    const points = [], occluders = new Set();
    let clear = null, best = Infinity;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const x = box[0] + 2 + (box[2] - 4) * (nx === 1 ? 0.5 : i / (nx - 1));
      const y = box[1] + 2 + (box[3] - 4) * (ny === 1 ? 0.5 : j / (ny - 1));
      if (x < 0 || y < 0 || x >= W || y >= H) { points.push({ x: x, y: y, hit: false }); continue; }
      const stack = document.elementsFromPoint(x, y).filter(function (e) { return !isOverlayNode(e); });
      const top = stack[0];
      const hit = !!top && (top === el || el.contains(top));
      points.push({ x: x, y: y, hit: hit });
      if (!hit && top) {
        for (const s of stack) { if (s === el || el.contains(s)) break; if (!s.contains(el)) { const sid = ids.get(s); if (sid !== undefined) occluders.add(sid); } }
      }
      if (hit) { const d = (x - cx) * (x - cx) + (y - cy) * (y - cy); if (d < best) { best = d; clear = [Math.round(x), Math.round(y)]; } }
    }
    return { found: true, box: box, center: center, points: points, occluders: Array.from(occluders), clear_point: clear,
             role: roleOf(el), name: nameOf(el, roleOf(el)) };
  }

  function describe(id) {
    const el = elementFor(id); if (!el) return null;
    const role = roleOf(el); return { id: Number(id), role: role, name: nameOf(el, role), box: boxOf(el) };
  }

  // ---------- overlay ----------
  function overlay(on, onlyIds) {
    let layer = document.getElementById(OVERLAY_ID);
    if (!on) { if (layer) layer.remove(); return 0; }
    const keep = onlyIds ? new Set(onlyIds) : null;
    if (!layer) {
      layer = document.createElement("div");
      layer.id = OVERLAY_ID;
      layer.setAttribute("aria-hidden", "true");
      layer.style.cssText = "position:fixed;left:0;top:0;width:0;height:0;overflow:visible;pointer-events:none;z-index:2147483647;";
      (document.body || document.documentElement).appendChild(layer);
    }
    layer.textContent = "";
    const entries = tree().filter(function (e) { return !keep || keep.has(e.id); });
    for (const e of entries) {
      const lab = document.createElement("div");
      lab.setAttribute("data-px-label", String(e.id));
      lab.textContent = String(e.id);
      lab.style.cssText = "position:fixed;left:" + e.box[0] + "px;top:" + e.box[1] + "px;" +
        "font:bold 10px/12px monospace;color:#000;background:#ffeb3b;border:1px solid #000;" +
        "padding:0 2px;height:12px;box-sizing:content-box;pointer-events:none;white-space:nowrap;";
      layer.appendChild(lab);
    }
    return entries.length;
  }

  function labelBoxes() {
    const layer = document.getElementById(OVERLAY_ID);
    if (!layer) return [];
    return Array.from(layer.children).map(function (lab) {
      const r = lab.getBoundingClientRect();
      return { id: Number(lab.getAttribute("data-px-label")), box: [r.left, r.top, r.width, r.height] };
    });
  }

  // ---------- click recorder ----------
  window.addEventListener("click", function (ev) {
    clicks.push({
      id: idOf(ev.target), path: pathOf(ev.target), tag: ev.target && ev.target.tagName ? ev.target.tagName.toLowerCase() : null,
      isTrusted: ev.isTrusted, x: ev.clientX, y: ev.clientY, t: Date.now(),
    });
  }, true);

  window.__parallax = {
    version: "0.1.0",
    tree: tree,
    overlay: overlay,
    labelBoxes: labelBoxes,
    idFor: function (selector) { const el = document.querySelector(selector); return el ? assign(el) : null; },
    idOf: idOf,
    pathOf: function (id) { const el = elementFor(id); return el ? pathOf(el) : []; },
    probe: probe,
    describe: describe,
    setHiddenMany: function (idList, hidden) { idList.forEach(function (i) { window.__parallax.setHidden(i, hidden); }); },
    boxFor: function (id) { const el = elementFor(id); return el ? boxOf(el) : null; },
    elementAt: function (x, y) { const el = document.elementFromPoint(x, y); return el ? { id: idOf(el), tag: el.tagName.toLowerCase() } : null; },
    setHidden: function (id, hidden) {
      const el = elementFor(id); if (!el) return false;
      if (hidden) { hiddenState.set(el, el.style.visibility); el.style.visibility = "hidden"; }
      else { el.style.visibility = hiddenState.has(el) ? hiddenState.get(el) : ""; hiddenState.delete(el); }
      return true;
    },
    clicks: function () { return clicks.slice(); },
    clearClicks: function () { clicks.length = 0; },
    counter: function () { return counter; },
    scan: scan,
  };
})();
