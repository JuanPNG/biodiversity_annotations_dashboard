// assets/custom.js
(function () {
  console.log("[assets/custom.js] LinkRenderer ready");
  window.dashAgGridFunctions = window.dashAgGridFunctions || {};

  // Capitalized name; return a DOM element.
  window.dashAgGridFunctions.LinkRenderer = function (params) {
    let v = params && params.value;
    if (v === undefined || v === null) return "";

    // Coerce to string (handles pandas NA / Arrow scalar)
    v = String(v).trim();
    if (!v) return "";

    // If missing scheme, assume https
    let href = /^https?:\/\//i.test(v) ? v : "https://" + v;

    const a = document.createElement("a");
    a.href = href;
    a.target = "_blank";
    a.rel = "noopener noreferrer";

    // Label: hostname (fallback to href)
    try {
      const u = new URL(href);
      a.textContent = (u.hostname || href).replace(/^www\./, "");
    } catch {
      a.textContent = href;
    }

    a.title = v;                 // full original on hover
    a.style.color = "inherit";   // ensure visible in dark theme
    a.style.textDecoration = "underline";
    a.style.whiteSpace = "nowrap";
    a.style.overflow = "hidden";
    a.style.textOverflow = "ellipsis";

    console.debug("LinkRenderer value:", v);

    return a;
  };
})();
