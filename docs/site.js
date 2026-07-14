document.addEventListener("DOMContentLoaded", () => {
  const progress = document.querySelector(".reading-progress span");
  const backToTop = document.querySelector(".back-to-top");
  const updateScroll = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    if (progress) progress.style.width = `${max > 0 ? (window.scrollY / max) * 100 : 0}%`;
    backToTop?.classList.toggle("visible", window.scrollY > 900);
  };
  window.addEventListener("scroll", updateScroll, { passive: true });
  updateScroll();

  backToTop?.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
  const toggle = document.querySelector(".toc-toggle");
  const toc = document.querySelector(".toc");
  const articleLayout = document.querySelector(".article-layout");
  toggle?.addEventListener("click", () => {
    const visible = !articleLayout?.classList.toggle("toc-collapsed");
    if (toc) toc.hidden = !visible;
    toggle.setAttribute("aria-expanded", String(visible));
    toggle.textContent = visible ? "隐藏本文目录" : "显示本文目录";
  });

  const tocLinks = [...document.querySelectorAll(".toc a")];
  const headings = tocLinks.map((link) => document.querySelector(link.getAttribute("href"))).filter(Boolean);
  if ("IntersectionObserver" in window && headings.length) {
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        tocLinks.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`));
      }
    }, { rootMargin: "-15% 0px -75% 0px" });
    headings.forEach((heading) => observer.observe(heading));
  }
});

