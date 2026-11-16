if (window.mermaid) {
  mermaid.initialize({
    startOnLoad: false,
    theme: 'base',
    themeVariables: {
      primaryColor: '#eef0f9',
      primaryTextColor: '#1e242b',
      primaryBorderColor: '#1b1f8a',
      lineColor: '#525d6a',
      secondaryColor: '#f6f4ee',
      tertiaryColor: '#fdfcf8',
      fontFamily: 'Source Sans 3, sans-serif'
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
  const sec = BOOK_SECTIONS.find(s => s.id === id) || BOOK_SECTIONS[0];
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

  let html = renderMarkdownWithMath(sec.markdown);
  if (sec.diagramKey && window.KC_DIAGRAMS[sec.diagramKey]) {
    html = window.KC_DIAGRAMS[sec.diagramKey] + html;
  }
  contentEl.innerHTML = html;

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
