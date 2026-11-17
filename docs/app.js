if (window.mermaid) {
  mermaid.initialize({
    startOnLoad: false,
    theme: 'base',
    themeVariables: {
      background: '#fdfcf8',
      primaryColor: '#f4f6fb',
      primaryTextColor: '#1e242b',
      primaryBorderColor: '#1b1f8a',
      lineColor: '#2b2b2b',
      secondaryColor: '#fbf7ef',
      tertiaryColor: '#f3f8f8',
      fontFamily: '"Source Sans 3", system-ui, sans-serif'
    }
  });
}

const navEl = document.getElementById('toc-nav');
const contentEl = document.getElementById('chapter-content');
const assetBarEl = document.getElementById('asset-bar');

function renderMarkdownWithMath(md) {
  const mathBlocks = [];
  let protectedMd = md.replace(/\$\$[\s\S]*?\$\$|\$[^\n$]+\$/g, (match) => {
    const id = `@@MATH_${mathBlocks.length}@@`;
    mathBlocks.push(match);
    return id;
  });
  protectedMd = protectedMd.replace(/```mermaid\n([\s\S]*?)```/g, (_, code) => {
    return `<div class="mermaid">\n${code}\n</div>`;
  });
  let html = marked.parse(protectedMd);
  mathBlocks.forEach((block, i) => {
    html = html.replace(`@@MATH_${i}@@`, () => block);
  });
  return html;
}

function decorateCallouts(container) {
  container.querySelectorAll('p').forEach((p) => {
    const firstStrong = p.querySelector('strong:first-child');
    if (!firstStrong) return;
    const txt = firstStrong.textContent.trim();
    let cls = null;
    if (/^(Def\.|Prop\.|Thm\.|Lemma|Corollary)/i.test(txt)) cls = 'callout';
    else if (/^(Example|Exercise)/i.test(txt)) cls = 'callout example';
    else if (/^(Remark|Note)/i.test(txt)) cls = 'callout remark';
    else if (/^(Coding|Algorithm)/i.test(txt)) cls = 'callout coding';
    if (cls) {
      const div = document.createElement('div');
      div.className = cls;
      p.parentNode.insertBefore(div, p);
      div.appendChild(p);
    }
  });
}

function buildNav() {
  let currentPart = '';
  BOOK_SECTIONS.forEach((sec, idx) => {
    if (sec.part !== currentPart) {
      currentPart = sec.part;
      const pDiv = document.createElement('div');
      pDiv.className = 'nav-part';
      pDiv.textContent = currentPart;
      navEl.appendChild(pDiv);
    }
    const btn = document.createElement('button');
    btn.className = 'nav-item' + (idx === 0 ? ' active' : '');
    btn.dataset.id = sec.id;
    btn.innerHTML = `<span>${sec.label}</span><span class="badge ${sec.verified ? 'badge-ready' : 'badge-draft'}">${sec.verified ? 'Verified' : 'Outline'}</span>`;
    btn.onclick = () => selectSection(sec.id);
    navEl.appendChild(btn);
  });
}

async function selectSection(id) {
  const idx = BOOK_SECTIONS.findIndex(s => s.id === id);
  const sec = idx >= 0 ? BOOK_SECTIONS[idx] : BOOK_SECTIONS[0];
  const activeIdx = idx >= 0 ? idx : 0;

  document.querySelectorAll('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.id === sec.id));
  window.location.hash = sec.id;

  let barHtml = `<strong>Section:</strong> ${sec.label}`;
  if (sec.notebook) {
    barHtml += `<a class="asset-pill" href="../notebooks/${sec.notebook}" download>Colab Notebook (.ipynb)</a>`;
  }
  if (sec.skill) {
    barHtml += `<a class="asset-pill" href="../skills/${sec.skill}/SKILL.md" target="_blank">Skill (SKILL.md)</a>`;
  }
  if (sec.chapterFile) {
    barHtml += `<a class="asset-pill" href="../chapters/${sec.chapterFile}" target="_blank">Raw Markdown</a>`;
  }
  assetBarEl.innerHTML = barHtml;

  const runningHead = `<div class="running-head">${sec.part} &middot; ${sec.label}</div>`;
  let html = renderMarkdownWithMath(sec.markdown);
  if (sec.diagramKey && window.KC_DIAGRAMS[sec.diagramKey]) {
    html = window.KC_DIAGRAMS[sec.diagramKey] + html;
  }

  const prevSec = activeIdx > 0 ? BOOK_SECTIONS[activeIdx - 1] : null;
  const nextSec = activeIdx < BOOK_SECTIONS.length - 1 ? BOOK_SECTIONS[activeIdx + 1] : null;
  const footerNav = `
    <div class="chapter-nav-footer">
      <div>${prevSec ? `<button class="chapter-nav-btn" onclick="selectSection('${prevSec.id}')">&larr; ${prevSec.label}</button>` : ''}</div>
      <div>${nextSec ? `<button class="chapter-nav-btn" onclick="selectSection('${nextSec.id}')">${nextSec.label} &rarr;</button>` : ''}</div>
    </div>`;

  contentEl.innerHTML = runningHead + html + footerNav;
  decorateCallouts(contentEl);

  if (window.mermaid) {
    await mermaid.run({ nodes: contentEl.querySelectorAll('.mermaid') });
  }
  if (window.MathJax && window.MathJax.typesetPromise) {
    await MathJax.typesetPromise([contentEl]);
  }
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

buildNav();
const initialId = window.location.hash ? window.location.hash.slice(1) : BOOK_SECTIONS[0].id;
selectSection(initialId);
