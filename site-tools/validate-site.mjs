import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "docs");
const htmlFiles = [];

function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const target = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(target);
    else if (entry.name.endsWith(".html")) htmlFiles.push(target);
  }
}

walk(root);
const missing = [];
let checkedLinks = 0;
for (const htmlFile of htmlFiles) {
  const html = fs.readFileSync(htmlFile, "utf8");
  for (const match of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    const value = match[1];
    if (/^(?:https?:|mailto:|#|\/)/.test(value)) continue;
    const clean = value.split("#")[0].split("?")[0];
    if (!clean) continue;
    checkedLinks += 1;
    const target = path.resolve(path.dirname(htmlFile), clean);
    if (!fs.existsSync(target)) missing.push(`${path.relative(root, htmlFile)} -> ${value}`);
  }
}

const articleFiles = fs.readdirSync(path.join(root, "articles")).filter((name) => name.endsWith(".html"));
const imageFiles = [];
walkImages(path.join(root, "assets"));
function walkImages(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const target = path.join(dir, entry.name);
    if (entry.isDirectory()) walkImages(target);
    else if (entry.name.toLowerCase().endsWith(".png")) imageFiles.push(target);
  }
}
const pdfFiles = fs.readdirSync(path.join(root, "pdf")).filter((name) => name.endsWith(".pdf"));

if (articleFiles.length !== 2) throw new Error(`Expected 2 articles, found ${articleFiles.length}`);
if (imageFiles.length !== 15) throw new Error(`Expected 15 images, found ${imageFiles.length}`);
if (pdfFiles.length !== 2) throw new Error(`Expected 2 PDFs, found ${pdfFiles.length}`);
if (missing.length) throw new Error(`Broken local links:\n${missing.join("\n")}`);

console.log(`Validated ${htmlFiles.length} HTML files, ${checkedLinks} local links, ${imageFiles.length} images and ${pdfFiles.length} PDFs.`);

