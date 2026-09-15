// Parallax page runtime: a virtual-time scheduler. Pages schedule their
// dynamics (delayed reflow) with __px.after(ms, fn); only the environment
// advances time with __px.advance(ms). No tool advances time.
(function () {
  if (window.__px) return;
  var now = 0, queue = [], seq = 0;
  window.__px = {
    now: function () { return now; },
    after: function (ms, fn) { queue.push({ at: now + ms, seq: seq++, fn: fn }); },
    advance: function (ms) {
      var target = now + ms;
      while (true) {
        var due = queue.filter(function (q) { return q.at <= target; });
        if (!due.length) break;
        due.sort(function (a, b) { return a.at - b.at || a.seq - b.seq; });
        var q = due[0];
        queue.splice(queue.indexOf(q), 1);
        now = q.at;
        try { q.fn(); } catch (e) { console.error("__px callback failed", e); }
      }
      now = target;
      return now;
    },
    pending: function () { return queue.length; }
  };
})();
