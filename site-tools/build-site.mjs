import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { marked } = require("marked");

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");
const docsRoot = path.join(repoRoot, "docs");

const articles = [
  {
    slug: "kepler-dr25-guide",
    source: path.join(repoRoot, "site-content", "kepler-dr25-guide.md"),
    pdf: path.join(docsRoot, "pdf", "kepler-dr25-guide.pdf"),
    pdfName: "kepler-dr25-guide.pdf",
    eyebrow: "Kepler DR25 · 入门科普",
    summary: "从凌星法、FITS与TCE开始，逐步读懂Kepler光变曲线、候选体与假阳性。",
    meta: "M0阶段 · 100颗恒星 · 1,569个FITS",
    imageMap: new Map([
      ["../操作记录/M0远程报告_2026-07-14/m0_pilot100_download_overview.png", "../assets/guide/download-overview.png"],
      ["../../kepler_article_assets/kepler_original_field_schematic.png", "../assets/guide/kepler-field.png"],
      ["../../kepler_article_assets/kepler_catalog_to_planets_numbers.png", "../assets/guide/catalog-levels.png"],
      ["../结果图/M0首批100目标/pilot100_lightcurve_cards/kic_003120355_candidate.png", "../assets/guide/candidate-example.png"],
      ["../结果图/M0首批100目标/pilot100_lightcurve_cards/kic_002309910_false_positive.png", "../assets/guide/false-positive-example.png"],
    ]),
    assets: [
      [path.join(docsRoot, "assets", "guide", "download-overview.png"), "guide/download-overview.png"],
      [path.join(docsRoot, "assets", "guide", "kepler-field.png"), "guide/kepler-field.png"],
      [path.join(docsRoot, "assets", "guide", "catalog-levels.png"), "guide/catalog-levels.png"],
      [path.join(docsRoot, "assets", "guide", "candidate-example.png"), "guide/candidate-example.png"],
      [path.join(docsRoot, "assets", "guide", "false-positive-example.png"), "guide/false-positive-example.png"],
    ],
  },
  {
    slug: "astronet-training",
    source: path.join(repoRoot, "site-content", "astronet-training.md"),
    pdf: path.join(docsRoot, "pdf", "astronet-training.pdf"),
    pdfName: "astronet-training.pdf",
    eyebrow: "AstroNet · 深度学习复现",
    summary: "完整解释双分支1D-CNN、数据预处理、训练划分、Robovetter对照与M2实验结果。",
    meta: "M2阶段 · 1,000颗恒星 · 1,164个TCE",
    imageMap: new Map(),
    assets: fs.readdirSync(path.join(docsRoot, "assets", "astronet"))
      .filter((name) => name.toLowerCase().endsWith(".png"))
      .map((name) => [path.join(docsRoot, "assets", "astronet", name), `astronet/${name}`]),
  },
];

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function copyFile(source, destination) {
  if (!fs.existsSync(source)) throw new Error(`Missing source asset: ${source}`);
  if (path.resolve(source) === path.resolve(destination)) return;
  ensureDir(path.dirname(destination));
  fs.copyFileSync(source, destination);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function plainText(html) {
  return html.replace(/<[^>]+>/g, "").replaceAll("&amp;", "&").replaceAll("&quot;", '"').trim();
}

function extractTitle(markdown) {
  const title = markdown.match(/^#\s+(.+)$/m)?.[1]?.trim() ?? "文章";
  let body = markdown.replace(/^#\s+.+\r?\n/, "").trimStart();
  const subtitle = body.match(/^##\s+(.+)$/m)?.[1]?.trim() ?? "";
  body = body.replace(/^##\s+.+\r?\n/, "").trimStart();
  return { title, subtitle, body };
}

function renderMarkdown(markdown, article) {
  let body = markdown.replaceAll("<!-- pdf-pagebreak -->", "");
  for (const [from, to] of article.imageMap) body = body.replaceAll(`](${from})`, `](${to})`);
  if (article.slug === "astronet-training") body = body.replaceAll("](assets/", "](../assets/astronet/");

  let html = marked.parse(body, { gfm: true });
  const headings = [];
  let headingIndex = 0;
  html = html.replace(/<h([2-4])>([\s\S]*?)<\/h\1>/g, (_, level, inner) => {
    headingIndex += 1;
    const id = `section-${headingIndex}`;
    headings.push({ level: Number(level), id, text: plainText(inner) });
    return `<h${level} id="${id}">${inner}<a class="heading-anchor" href="#${id}" aria-label="链接到本节">#</a></h${level}>`;
  });
  html = html.replace(/<p><img src="([^"]+)" alt="([^"]*)"\s*><\/p>/g, (_, src, alt) => (
    `<figure><img src="${src}" alt="${alt}" loading="lazy"><figcaption>${alt}</figcaption></figure>`
  ));
  html = html.replaceAll("<table>", '<div class="table-wrap"><table>').replaceAll("</table>", "</table></div>");
  html = html.replace(/<a href="(https?:\/\/[^\"]+)"/g, '<a href="$1" target="_blank" rel="noopener noreferrer"');
  return { html, headings };
}

function tocHtml(headings) {
  return headings
    .filter((heading) => heading.level <= 3)
    .map((heading) => `<a class="toc-level-${heading.level}" href="#${heading.id}">${escapeHtml(heading.text)}</a>`)
    .join("\n");
}

function pageShell({ title, description, body, pathPrefix = "", article = false }) {
  const canonical = `https://sutony1.github.io/astronet-dr25-pytorch/${pathPrefix}`;
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="${escapeHtml(description)}">
  <meta name="theme-color" content="#17324d">
  <meta property="og:type" content="${article ? "article" : "website"}">
  <meta property="og:title" content="${escapeHtml(title)}">
  <meta property="og:description" content="${escapeHtml(description)}">
  <meta property="og:url" content="${canonical}">
  <meta property="og:image" content="https://sutony1.github.io/astronet-dr25-pytorch/assets/astronet/08_astronet_layer_by_layer.png">
  <link rel="canonical" href="${canonical}">
  <link rel="icon" type="image/png" href="${article ? "../" : ""}assets/guide/kepler-field.png">
  <link rel="stylesheet" href="${article ? "../" : ""}styles.css">
  <script defer src="${article ? "../" : ""}site.js"></script>
  <title>${escapeHtml(title)} · Kepler系外行星机器学习</title>
</head>
<body${article ? ' class="article-page"' : ""}>
${body}
</body>
</html>`;
}

function articlePage(article, title, subtitle, rendered, previous, next) {
  const navItem = (item, label) => item
    ? `<a href="${item.slug}.html"><span>${label}</span><strong>${escapeHtml(item.title)}</strong></a>`
    : "<span></span>";
  const body = `
<div class="reading-progress" aria-hidden="true"><span></span></div>
<header class="site-header">
  <a class="brand" href="../index.html"><span class="brand-mark">K</span><span>Kepler系外行星机器学习</span></a>
  <nav aria-label="主导航"><a href="../index.html">首页</a><a href="https://github.com/sutony1/astronet-dr25-pytorch">源代码</a></nav>
</header>
<main>
  <section class="article-hero">
    <div class="article-hero-inner">
      <p class="eyebrow">${escapeHtml(article.eyebrow)}</p>
      <h1>${escapeHtml(title)}</h1>
      <p class="subtitle">${escapeHtml(subtitle)}</p>
      <p class="article-meta">${escapeHtml(article.meta)} · 2026-07-14研究记录</p>
      <div class="hero-actions">
        <a class="button primary" href="../pdf/${article.pdfName}" download>下载PDF</a>
        <a class="button secondary" href="https://github.com/sutony1/astronet-dr25-pytorch">查看GitHub</a>
      </div>
    </div>
  </section>
  <div class="toc-toolbar">
    <button class="toc-toggle" type="button" aria-expanded="false" aria-controls="article-toc">显示本文目录</button>
  </div>
  <div class="article-layout toc-collapsed">
    <aside class="toc-panel">
      <nav class="toc" id="article-toc" aria-label="本文目录" hidden><p>本文目录</p>${tocHtml(rendered.headings)}</nav>
    </aside>
    <article class="prose">${rendered.html}</article>
  </div>
  <nav class="article-neighbors" aria-label="文章导航">${navItem(previous, "上一篇")}${navItem(next, "下一篇")}</nav>
</main>
<footer class="site-footer"><p>Kepler DR25 × AstroNet PyTorch · 数据来自NASA公开档案</p><a href="../index.html">返回首页</a></footer>
<button class="back-to-top" type="button" aria-label="返回顶部">↑</button>`;
  return pageShell({
    title,
    description: article.summary,
    body,
    pathPrefix: `articles/${article.slug}.html`,
    article: true,
  });
}

function homePage(metadata) {
  const [guide, training] = metadata;
  const body = `
<header class="site-header home-header">
  <a class="brand" href="index.html"><span class="brand-mark">K</span><span>Kepler系外行星机器学习</span></a>
  <nav aria-label="主导航"><a href="#articles">文章</a><a href="#results">研究结果</a><a href="https://github.com/sutony1/astronet-dr25-pytorch">GitHub</a></nav>
</header>
<main>
  <section class="home-hero">
    <div class="home-hero-copy">
      <p class="eyebrow">KEPLER DR25 · ASTRONET · PYTORCH</p>
      <h1>从星光变暗，<br><span>走到神经网络。</span></h1>
      <p class="lead">在一台双RTX 3090工作站上，复现Kepler系外行星分类流程。这里把天文数据、1D-CNN、训练结果和公共数据库核对，讲成两篇任何人都能读懂的文章。</p>
      <div class="hero-actions"><a class="button primary" href="#articles">开始阅读</a><a class="button secondary" href="https://github.com/sutony1/astronet-dr25-pytorch">查看源代码</a></div>
    </div>
    <div class="hero-visual">
      <div class="orbit orbit-one"></div><div class="orbit orbit-two"></div><div class="star"></div><div class="planet"></div>
      <div class="signal-card"><span>相对亮度</span><div class="signal-line"></div><strong>周期性凌星</strong></div>
    </div>
  </section>
  <section class="metrics" id="results" aria-label="核心研究数据">
    <div><strong>1,000</strong><span>颗恒星</span></div><div><strong>1,164</strong><span>个TCE</span></div><div><strong>84.12%</strong><span>测试准确率</span></div><div><strong>79</strong><span>个当前确认行星信号</span></div>
  </section>
  <section class="articles-section" id="articles">
    <div class="section-heading"><p class="eyebrow">两篇完整长文</p><h2>从基础天文，到可复现的深度学习</h2><p>HTML保留Markdown全文、图表和代码，并提供原版PDF下载。</p></div>
    <div class="article-cards">
      <article class="article-card guide-card">
        <a class="card-image" href="articles/${guide.slug}.html"><img src="assets/guide/kepler-field.png" alt="Kepler原始视场示意图"></a>
        <div class="card-body"><p class="eyebrow">${escapeHtml(guide.eyebrow)}</p><h3><a href="articles/${guide.slug}.html">${escapeHtml(guide.title)}</a></h3><p>${escapeHtml(articles[0].summary)}</p><p class="card-meta">${escapeHtml(articles[0].meta)}</p><div class="card-actions"><a href="articles/${guide.slug}.html">阅读全文 →</a><a href="pdf/${articles[0].pdfName}" download>PDF</a></div></div>
      </article>
      <article class="article-card training-card">
        <a class="card-image" href="articles/${training.slug}.html"><img src="assets/astronet/08_astronet_layer_by_layer.png" alt="AstroNet逐层网络架构图"></a>
        <div class="card-body"><p class="eyebrow">${escapeHtml(training.eyebrow)}</p><h3><a href="articles/${training.slug}.html">${escapeHtml(training.title)}</a></h3><p>${escapeHtml(articles[1].summary)}</p><p class="card-meta">${escapeHtml(articles[1].meta)}</p><div class="card-actions"><a href="articles/${training.slug}.html">阅读全文 →</a><a href="pdf/${articles[1].pdfName}" download>PDF</a></div></div>
      </article>
    </div>
  </section>
  <section class="method-section">
    <div><p class="eyebrow">完整研究链路</p><h2>不是把FITS当图片，而是把星光变成可学习的序列</h2></div>
    <ol><li><span>01</span><strong>NASA目录与FITS</strong><p>连接TCE、KOI、恒星参数和季度光变。</p></li><li><span>02</span><strong>相位折叠</strong><p>生成2001点global和201点local视图。</p></li><li><span>03</span><strong>双分支1D-CNN</strong><p>单张RTX 3090训练AstroNet基线。</p></li><li><span>04</span><strong>公共目录核对</strong><p>区分模型候选、当前确认行星与假阳性。</p></li></ol>
  </section>
</main>
<footer class="site-footer"><p>Kepler DR25 × AstroNet PyTorch · 研究与科普记录</p><a href="https://github.com/sutony1/astronet-dr25-pytorch">GitHub公开仓库</a></footer>`;
  return pageShell({
    title: "Kepler系外行星机器学习",
    description: "Kepler DR25与AstroNet PyTorch复现项目：两篇中文科普长文、完整图表与公开源代码。",
    body,
    pathPrefix: "",
  });
}

ensureDir(path.join(docsRoot, "articles"));
ensureDir(path.join(docsRoot, "assets"));
ensureDir(path.join(docsRoot, "pdf"));
fs.writeFileSync(path.join(docsRoot, ".nojekyll"), "", "utf8");

const metadata = [];
for (let index = 0; index < articles.length; index += 1) {
  const article = articles[index];
  const markdown = fs.readFileSync(article.source, "utf8");
  const extracted = extractTitle(markdown);
  const rendered = renderMarkdown(extracted.body, article);
  const meta = { ...article, title: extracted.title, subtitle: extracted.subtitle };
  metadata.push(meta);
  for (const [source, relativeDestination] of article.assets) {
    copyFile(source, path.join(docsRoot, "assets", relativeDestination));
  }
  copyFile(article.pdf, path.join(docsRoot, "pdf", article.pdfName));
  const previous = index > 0 ? { slug: articles[index - 1].slug, title: metadata[index - 1].title } : null;
  const nextArticle = index + 1 < articles.length ? articles[index + 1] : null;
  const next = nextArticle ? { slug: nextArticle.slug, title: extractTitle(fs.readFileSync(nextArticle.source, "utf8")).title } : null;
  fs.writeFileSync(path.join(docsRoot, "articles", `${article.slug}.html`), articlePage(article, extracted.title, extracted.subtitle, rendered, previous, next), "utf8");
}

fs.writeFileSync(path.join(docsRoot, "index.html"), homePage(metadata), "utf8");
fs.writeFileSync(path.join(docsRoot, "404.html"), `<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=/astronet-dr25-pytorch/"><title>返回首页</title>`, "utf8");
console.log(`Built ${metadata.length} articles in ${docsRoot}`);
